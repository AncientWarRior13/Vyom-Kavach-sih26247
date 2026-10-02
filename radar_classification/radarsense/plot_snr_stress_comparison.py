"""Script to generate a beautiful, presentation-ready comparison graph of the SNR Noise Stress-Test results.

It plots:
1. Binary Classification: Quantum vs Classical vs Stacking Ensemble Accuracy under SNR Noise Stress.
2. Multi-Class Architectures: 11-Feature vs 31-Feature vs 42-Feature ONNX Model Accuracy under SNR Noise Stress.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

plt.style.use('default')
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#CBD5E1'
plt.rcParams['axes.linewidth'] = 1.2

NAVY = '#1E293B'
BLUE = '#2563EB'
TEAL = '#0D9488'
AMBER = '#D97706'
RED = '#DC2626'
PURPLE = '#7C3AED'
GRAY = '#64748B'

REPORTS_DIR = Path("artifacts/reports")
PLOTS_DIR = Path("artifacts/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

ENSEMBLE_JSON_PATH = REPORTS_DIR / "snr_stress_test_ensemble_summary.json"
MULTICLASS_JSON_PATH = REPORTS_DIR / "snr_stress_test_multiclass_summary.json"
OUTPUT_PLOT_PATH = PLOTS_DIR / "snr_stress_test_comparison.png"

def main():
    print("[1/3] Loading stress test JSON summaries...")
    
    with open(ENSEMBLE_JSON_PATH, "r") as f:
        ensemble_data = json.load(f)
        
    with open(MULTICLASS_JSON_PATH, "r") as f:
        multiclass_data = json.load(f)

    snr_levels = ensemble_data["snr_levels_db"]
    ensemble_results = ensemble_data["results"]
    multiclass_results = multiclass_data["results"]
    
    binary_snr = [r["snr_db"] for r in ensemble_results]
    q_acc = [r["quantum_accuracy"] * 100 for r in ensemble_results]
    c_acc = [r["classical_accuracy"] * 100 for r in ensemble_results]
    ens_acc = [r["ensemble_accuracy"] * 100 for r in ensemble_results]
    
    q_recall = [r["quantum_recall_drone"] * 100 for r in ensemble_results]
    c_recall = [r["classical_recall_drone"] * 100 for r in ensemble_results]
    ens_recall = [r["ensemble_recall_drone"] * 100 for r in ensemble_results]
    
    multi_snr = [r["snr_db"] for r in multiclass_results]
    m11_acc = [r["m11_onnx_acc"] * 100 for r in multiclass_results]
    m31_acc = [r["m31_onnx_acc"] * 100 for r in multiclass_results]
    m42_acc = [r["m42_onnx_acc"] * 100 for r in multiclass_results]
    
    print("[2/3] Generating comparison charts...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), dpi=300)
    
    ax1.plot(binary_snr, ens_acc, 'o-', color=TEAL, linewidth=2.5, markersize=7, label='Stacking Ensemble (LogReg Meta)')
    ax1.plot(binary_snr, c_acc, 's--', color=BLUE, linewidth=2.0, markersize=6, label='Classical SVM (RBF Kernel)')
    ax1.plot(binary_snr, q_acc, '^-.', color=RED, linewidth=2.0, markersize=6, label='Quantum SVC (ZZFeatureMap)')
    
    ax1.set_xlabel('Signal-to-Noise Ratio (SNR) in dB', fontsize=11, fontweight='bold', color=NAVY)
    ax1.set_ylabel('Inference Accuracy (%)', fontsize=11, fontweight='bold', color=NAVY)
    ax1.set_title('Panel A: Binary Drone Detection Modes under Noise\n(Stacking Ensemble vs Classical SVM vs Quantum SVC)', 
                 fontsize=12, fontweight='bold', color=NAVY, pad=12)
    ax1.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax1.set_xlim(max(binary_snr) + 1, min(binary_snr) - 1)
    ax1.set_ylim(60, 102)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax1.legend(loc='lower left', frameon=True, fontsize=9.5)
    
    ax1.annotate('Classical SVM & Ensemble\nstay robust (>91% accuracy)', 
                 xy=(-25, 92), xytext=(-18, 80),
                 arrowprops=dict(facecolor=NAVY, shrink=0.08, width=1.0, headwidth=5),
                 fontsize=9, color=NAVY, bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#CBD5E1", lw=0.8))
                 
    ax1.annotate('Quantum SVC accuracy\ndegrades below 84%', 
                 xy=(-30, 83.3), xytext=(-26, 70),
                 arrowprops=dict(facecolor=RED, shrink=0.08, width=1.0, headwidth=5),
                 fontsize=9, color=RED, bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#CBD5E1", lw=0.8))

    ax2.plot(multi_snr, m42_acc, 'o-', color=PURPLE, linewidth=2.5, markersize=7, label='42-Feature Combined Model')
    ax2.plot(multi_snr, m11_acc, 's--', color=BLUE, linewidth=2.0, markersize=6, label='11-Feature Physics-Only Model')
    ax2.plot(multi_snr, m31_acc, '^-.', color=AMBER, linewidth=2.0, markersize=6, label='31-Feature Waveform-Only Model')
    
    ax2.set_xlabel('Signal-to-Noise Ratio (SNR) in dB', fontsize=11, fontweight='bold', color=NAVY)
    ax2.set_ylabel('4-Class Multi-Class Accuracy (%)', fontsize=11, fontweight='bold', color=NAVY)
    ax2.set_title('Panel B: Multi-Class Architectures under Noise Sweep\n(11 Physics vs 31 Waveform vs 42 Combined Features)', 
                 fontsize=12, fontweight='bold', color=NAVY, pad=12)
    ax2.grid(True, linestyle=':', alpha=0.6, color='#CBD5E1')
    ax2.set_xlim(max(multi_snr) + 1, min(multi_snr) - 1)
    ax2.set_ylim(55, 102)
    ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax2.legend(loc='lower left', frameon=True, fontsize=9.5)

    ax2.annotate('42-Feature Combined model\nachieves top clean accuracy (97.2%)', 
                 xy=(-5.0, 97.2), xytext=(-18, 93),
                 arrowprops=dict(facecolor=PURPLE, shrink=0.08, width=1.0, headwidth=5),
                 fontsize=9, color=PURPLE, bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#CBD5E1", lw=0.8))

    ax2.annotate('31-Feature Waveform model\ndegrades rapidly to 63.2%', 
                 xy=(-30, 63.2), xytext=(-26, 75),
                 arrowprops=dict(facecolor=AMBER, shrink=0.08, width=1.0, headwidth=5),
                 fontsize=9, color=AMBER, bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#CBD5E1", lw=0.8))

    ax2.annotate('11-Feature Physics model\nretains 80.0% accuracy', 
                 xy=(-30, 80.0), xytext=(-20, 85),
                 arrowprops=dict(facecolor=BLUE, shrink=0.08, width=1.0, headwidth=5),
                 fontsize=9, color=BLUE, bbox=dict(boxstyle="round,pad=0.3", fc="#F8FAFC", ec="#CBD5E1", lw=0.8))

    fig.suptitle('RadarSense Noise Stress-Test Results: Performance under Critical SNR Conditions', 
                 fontsize=15, fontweight='bold', color=NAVY, y=1.01)
    plt.tight_layout()
    
    plt.savefig(OUTPUT_PLOT_PATH, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"[3/3] Chart successfully saved -> {OUTPUT_PLOT_PATH}")

if __name__ == "__main__":
    main()
