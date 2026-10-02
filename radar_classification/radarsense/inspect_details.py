import json
import os
from pathlib import Path

def print_section(title):
    print("=" * 60)
    print(title)
    print("=" * 60)

print_section("1. Baseline & Optimization Results (11-feature)")
with open("artifacts/reports/baseline_results.json") as f:
    b_res = json.load(f)

cb_test = b_res["catboost"]["test"]
cb_cm = b_res["catboost"]["test_confusion_matrix"]

print("CatBoost Baseline Test Metrics (11-feature):")
print(f"  Accuracy: {cb_test['accuracy']:.6f} ({cb_test['accuracy']*100:.2f}%)")
print(f"  Drone Recall: {cb_test['recall_drone']:.6f}")
print(f"  Drone Precision: {cb_test['precision_drone']:.6f}")
print(f"  Drone PR-AUC: {cb_test['pr_auc_drone']:.6f}")
print("CatBoost Baseline Test Confusion Matrix:")
print(cb_cm)
total_samples = sum(sum(row) for row in cb_cm)
correct_samples = sum(cb_cm[i][i] for i in range(len(cb_cm)))
print(f"  CM total: {total_samples}, correct: {correct_samples}, accuracy = {correct_samples/total_samples:.6f} ({correct_samples/total_samples*100:.2f}%)")

print_section("2. CV Error Bars in baseline_results.json")
for m_name in ["catboost", "xgboost", "random_forest", "balanced_rf", "svm_rbf", "logreg"]:
    if m_name in b_res:
        pr_mean = b_res[m_name]["cv"]["pr_auc_drone"]["mean"]
        pr_std = b_res[m_name]["cv"]["pr_auc_drone"]["std"]
        print(f"  {m_name:18s}: mean = {pr_mean:.3f}, std = {pr_std:.3f}")

print_section("3. ONNX File Sizes")
onnx_dir = Path("artifacts/onnx")
if onnx_dir.exists():
    for f_path in onnx_dir.glob("*"):
        size_bytes = f_path.stat().st_size
        print(f"  {f_path.name}: {size_bytes} bytes ({size_bytes/1024:.2f} KiB / {size_bytes/(1024*1024):.2f} MB)")

print_section("4. Quantum Scaling Study Numbers")
with open("artifacts/reports/quantum_results.json") as f:
    q_res = json.load(f)

for r in q_res.get("results", []):
    print(r)

