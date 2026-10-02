"""PennyLane differentiable VQC -- the second SDK from the Quant-A-thon primer,
and the one it points to for quantum ML (slide 12: "gradient descent, on a
qubit"). Our Qiskit VQC (vqc.py / vqc_v2.py) trained with GRADIENT-FREE
optimizers (COBYLA / SPSA) and underperformed -- ROC-AUC ~0.63, near random.
PennyLane's whole pitch is that the circuit is a differentiable function, so we
can train the same kind of ansatz with true GRADIENT DESCENT (Adam) and give
the VQC its best honest shot.

Same setup as the Qiskit quantum experiments for a fair comparison: the 4
QUANTUM_FEATURES, drone-vs-not binary, stratified subsample (drone frac 0.30).
Circuit: AngleEmbedding (4 features -> 4 qubit rotations) + StronglyEntangling
Layers ansatz + expval(PauliZ(0)) read-out -- a QNode, differentiated end to
end. Reported multi-seed, honestly, next to a classical baseline and the Qiskit
VQC numbers.

Device-agnostic (primer slide 14): the SAME QNode runs on Qiskit/IBM hardware
by swapping the device to 'qiskit.aer' / 'qiskit.remote' (needs
pennylane-qiskit); default here is the fast local 'default.qubit'. That
one-line swap is the "prototype in PennyLane, deploy on Qiskit" workflow.
"""

import json

import numpy as np
import pandas as pd
import pennylane as qml
from pennylane import numpy as pnp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import MinMaxScaler

from radarsense import config as C
from radarsense.quantum import subsample

N_QUBITS = len(C.QUANTUM_FEATURES)
N_LAYERS = 3
N_TRAIN, N_TEST = 300, 300
EPOCHS, BATCH = 40, 25
SEEDS = [0, 1, 2]

dev = qml.device("default.qubit", wires=N_QUBITS)


@qml.qnode(dev, interface="autograd", diff_method="backprop")
def circuit(weights, X):
    qml.AngleEmbedding(X, wires=range(N_QUBITS), rotation="Y")
    qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
    return qml.expval(qml.PauliZ(0))


def classifier(weights, bias, X):
    return circuit(weights, X) + bias


def cost(weights, bias, X, Y):
    preds = classifier(weights, bias, X)
    return pnp.mean((Y - preds) ** 2)


def train_one(Xtr, ytr, seed):
    rng = np.random.default_rng(seed)
    shape = qml.StronglyEntanglingLayers.shape(n_layers=N_LAYERS, n_wires=N_QUBITS)
    weights = pnp.array(0.05 * rng.standard_normal(shape), requires_grad=True)
    bias = pnp.array(0.0, requires_grad=True)
    Y = pnp.array(1 - 2 * ytr, requires_grad=False)
    opt = qml.AdamOptimizer(0.08)
    n = len(Xtr)
    for _ in range(EPOCHS):
        idx = rng.permutation(n)
        for b in range(0, n, BATCH):
            bi = idx[b:b + BATCH]
            weights, bias = opt.step(lambda w, bb: cost(w, bb, Xtr[bi], Y[bi]), weights, bias)
    return weights, bias


def main():
    train = pd.read_parquet(C.DATA_DIR / "radar_train.parquet")
    test = pd.read_parquet(C.DATA_DIR / "radar_test.parquet")

    accs, recs, precs, roc, pr = [], [], [], [], []
    cls_roc, cls_pr = [], []
    for seed in SEEDS:
        Xtr_raw, ytr = subsample(train, N_TRAIN, np.random.RandomState(seed))
        Xte_raw, yte = subsample(test, N_TEST, np.random.RandomState(100 + seed))
        scaler = MinMaxScaler((0, np.pi)).fit(Xtr_raw)
        Xtr = pnp.array(scaler.transform(Xtr_raw), requires_grad=False)
        Xte = pnp.array(scaler.transform(Xte_raw), requires_grad=False)

        weights, bias = train_one(Xtr, ytr, seed)
        scores = np.array(classifier(weights, bias, Xte))
        drone_score = -scores
        pred_drone = (drone_score > 0).astype(int)

        tp = int(((pred_drone == 1) & (yte == 1)).sum())
        fp = int(((pred_drone == 1) & (yte == 0)).sum())
        fn = int(((pred_drone == 0) & (yte == 1)).sum())
        accs.append(float((pred_drone == yte).mean()))
        recs.append(tp / (tp + fn) if tp + fn else 0.0)
        precs.append(tp / (tp + fp) if tp + fp else 0.0)
        roc.append(roc_auc_score(yte, drone_score))
        pr.append(average_precision_score(yte, drone_score))

        lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr_raw, ytr)
        p = lr.predict_proba(Xte_raw)[:, 1]
        cls_roc.append(roc_auc_score(yte, p)); cls_pr.append(average_precision_score(yte, p))
        print(f"  seed {seed}: PennyLane VQC roc_auc={roc[-1]:.3f} pr_auc={pr[-1]:.3f} acc={accs[-1]:.3f} "
              f"| classical roc_auc={cls_roc[-1]:.3f}")

    def ms(v):
        return {"mean": float(np.mean(v)), "std": float(np.std(v))}

    print("\n=== PennyLane differentiable VQC (Adam gradient descent), 3 seeds ===")
    print(f"  ROC-AUC : {np.mean(roc):.4f} +/- {np.std(roc):.4f}")
    print(f"  PR-AUC  : {np.mean(pr):.4f} +/- {np.std(pr):.4f}")
    print(f"  accuracy: {np.mean(accs):.4f}   recall: {np.mean(recs):.4f}   precision: {np.mean(precs):.4f}")
    print("\n--- comparison ---")
    print(f"  Qiskit VQC (COBYLA, vqc.py)   : ROC-AUC ~0.50, PR-AUC 0.483  (gradient-free)")
    print(f"  Qiskit VQC (SPSA, vqc_v2.py)  : ROC-AUC 0.629, PR-AUC 0.394  (gradient-free)")
    print(f"  PennyLane VQC (Adam, this)    : ROC-AUC {np.mean(roc):.3f}, PR-AUC {np.mean(pr):.3f}  (GRADIENT descent)")
    print(f"  classical LogisticRegression  : ROC-AUC {np.mean(cls_roc):.3f}, PR-AUC {np.mean(cls_pr):.3f}")

    out = {
        "sdk": "PennyLane 0.45 (default.qubit); same circuit runs on Qiskit via pennylane-qiskit",
        "circuit": {"embedding": "AngleEmbedding(RY)", "ansatz": "StronglyEntanglingLayers",
                    "n_layers": N_LAYERS, "qubits": N_QUBITS, "optimizer": "Adam (gradient descent)",
                    "epochs": EPOCHS, "n_train": N_TRAIN, "n_test": N_TEST},
        "pennylane_vqc": {"roc_auc": ms(roc), "pr_auc": ms(pr), "accuracy": ms(accs),
                          "recall_drone": ms(recs), "precision_drone": ms(precs)},
        "classical_baseline_same_features": {"roc_auc": ms(cls_roc), "pr_auc": ms(cls_pr)},
        "qiskit_vqc_reference": {"cobyla_pr_auc": 0.483, "spsa_roc_auc": 0.629},
    }
    (C.REPORTS_DIR / "vqc_pennylane_results.json").write_text(json.dumps(out, indent=2))
    print(f"\nresults -> {C.REPORTS_DIR / 'vqc_pennylane_results.json'}")


if __name__ == "__main__":
    main()
