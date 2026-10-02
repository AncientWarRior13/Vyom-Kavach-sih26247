import re

with open('Final_report.md', 'r', encoding='utf-8') as f:
    text = f.read()

forbidden_patterns = [
    (r'91\.8%\s*\(as SAAB zero-shot\)', "91.8% zero-shot SAAB"),
    (r'0\.084', "0.084 raw ECE"),
    (r'0\.742', "0.742 LogReg PR-AUC"),
    (r'0\.105', "0.105 cost-optimal threshold"),
    (r'14\.2%', "14.2% zero-shot DIAT-uSAT"),
    (r'94\.8%', "94.8% retrained DIAT-uSAT")
]

clean = True
for pat, label in forbidden_patterns:
    matches = list(re.finditer(pat, text))
    if matches:
        print(f"[FAIL] Found stale/wrong value for '{label}': {len(matches)} occurrences")
        clean = False
    else:
        print(f"[PASS] Checked '{label}': Clean!")

if clean:
    print("\n[SUCCESS] Final_report.md passed all ground truth self-checks!")
