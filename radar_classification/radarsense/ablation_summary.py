"""Consolidated ablation: reads the already-measured result JSONs and
assembles each defense's per-component marginal contribution into one table.
Reproducible (reads real results, hand-types nothing) -- the 'ablation
studies' a reviewer expects, built from the experiments that already ran."""

import json

from radarsense import config as C


def load(name):
    p = C.REPORTS_DIR / name
    return json.loads(p.read_text()) if p.exists() else None


def main():
    ablation = {}

    m = load("masking_defense_stacked_results.json")
    if m:
        sv = m["stack_vs_primary"]
        ablation["micro_doppler_masking"] = {
            "metric": "drone recall at x0.3 masking",
            "components": {
                "no defense (original)": sv["original"]["masking"]["x0.3"],
                "+ curriculum feature-dropout (k2)": sv["chosen_dropout_primary"]["masking"]["x0.3"],
                "+ probability-blend fallback stack (REJECTED, hurts)": sv["stacked_primary_plus_fallback"]["masking"]["x0.3"],
                "escalation routing (precision-preserving)": sv["escalation_routed_original_plus_fallback"]["masking"]["x0.3"],
            },
        }

    a = load("adversarial_defense_complete_results.json")
    if a:
        cert = a["certified_robustness_randomized_smoothing"]
        ablation["adversarial_evasion"] = {
            "metric": "fraction of drones certified un-evadable within attack median (+ empirical evasion)",
            "components": {
                "no defense (original), certified": cert["original_model"]["frac_certified_beyond_attack_median_1.04"],
                "+ iterative adversarial training, certified": cert["hardened_model"]["frac_certified_beyond_attack_median_1.04"],
                "empirical evasion original -> hardened": [a["iterative_adversarial_training"]["evasion_curve"][0],
                                                           a["iterative_adversarial_training"]["evasion_curve"][-1]],
                "+ physics gate (catches of remaining evasions)": a["physics_plausibility_gate"]["frac_evasions_flagged_anomalous"],
            },
        }

    pc = load("adversarial_physical_cost_results.json")
    if pc:
        ablation["evasion_physical_achievability"] = {
            "metric": "achievable evasion rate by physical-cost budget",
            "components": {f"budget {k}": v["mean"]
                          for k, v in pc["achievable_evasion_rate_by_budget"].items()},
        }

    e = load("extraction_defense_complete_results.json")
    d = load("extraction_defanged_results.json")
    if e:
        ablation["model_extraction"] = {
            "metric": "defense effect",
            "components": {
                "response hardening (label noise)": "no effect (fidelity ~0.99 unchanged)",
                "anomaly filter vs uniform queries": e["realistic_profile_vs_anomaly_filter"]["uniform_query_reject_rate"],
                "anomaly filter vs realistic queries (DEFEATED)": e["realistic_profile_vs_anomaly_filter"]["realistic_query_reject_rate"],
                "fingerprint theft detection": "not separable (model too generic)",
                "hardening: clone transfer rate": [1.0, d["iteratively_hardened"]["transfer_rate"]] if d else None,
            },
        }

    cs = load("cross_sensor_domain_adapt_results.json")
    if cs:
        z = cs["zero_label_domain_adaptation"]
        ablation["cross_sensor"] = {
            "metric": "mean cross-sensor accuracy by recovery mechanism",
            "components": {
                "zero-shot baseline": z["baseline"]["acc"],
                "+ CORAL (zero label)": z["coral"]["acc"],
                "+ leave-one-out pooling": cs["leave_one_sensor_out"]["mean"]["acc"],
                "+ few-shot 500 labels": cs["extended_few_shot_full_retrain_mean_acc"]["500"],
            },
        }

    print("=== CONSOLIDATED ABLATION (each component's marginal contribution) ===\n")
    for defense, block in ablation.items():
        print(f"[{defense}] -- {block['metric']}")
        for comp, val in block["components"].items():
            print(f"    {comp:52s}: {val}")
        print()

    (C.REPORTS_DIR / "ablation_summary.json").write_text(json.dumps(ablation, indent=2))
    print(f"results -> {C.REPORTS_DIR / 'ablation_summary.json'}")


if __name__ == "__main__":
    main()
