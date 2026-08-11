"""Create the final condition-wise comparison table for the cloud-teacher gate."""
import argparse
import json
import os


CONDITIONS = ("clean", "illumination_1.0", "motion_blur_0.8", "sensor_noise_0.6")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--adapter-dir", required=True)
    p.add_argument("--head-metrics", required=True)
    p.add_argument("--output-json", required=True)
    p.add_argument("--output-md", required=True)
    p.add_argument("--min-cloud-gain", type=float, default=.01)
    p.add_argument("--min-unlabeled-gain", type=float, default=.01)
    p.add_argument("--max-unlabeled-gap-to-label", type=float, default=.02)
    return p.parse_args()


def load_manifest(root, name):
    path = os.path.join(root, name, "manifest.json")
    with open(path) as handle: payload = json.load(handle)
    scores = payload.get("validation_accuracy")
    if not isinstance(scores, dict) or any(key not in scores for key in CONDITIONS):
        raise ValueError(f"{path} lacks condition-wise validation_accuracy")
    return {key: float(scores[key]) for key in CONDITIONS}


def drift_mean(scores):
    return sum(scores[key] for key in CONDITIONS[1:]) / 3


def main():
    a = parse_args()
    with open(a.head_metrics) as handle: head_metrics = json.load(handle)
    selected = head_metrics.get("selected")
    if not selected:
        raise ValueError("head metrics has no selected candidate")
    teacher = head_metrics["results"][selected]["accuracy"]
    models = {
        "Edge baseline": {"clean": .9383667, "illumination_1.0": .8181818,
                          "motion_blur_0.8": .8459168, "sensor_noise_0.6": .7950693},
        "Cloud teacher": {key: float(teacher[key]) for key in CONDITIONS},
        "Label-only Adapter": load_manifest(a.adapter_dir, "label_only_control"),
        "Cloud hybrid Adapter": load_manifest(a.adapter_dir, "cloud_hybrid"),
        "Cloud unlabeled Adapter": load_manifest(a.adapter_dir, "cloud_unlabeled"),
    }
    for scores in models.values(): scores["mean_drift"] = drift_mean(scores)
    hybrid_gain = models["Cloud hybrid Adapter"]["mean_drift"] - models["Label-only Adapter"]["mean_drift"]
    hybrid_no_regression = all(
        models["Cloud hybrid Adapter"][key] + .005 >= models["Label-only Adapter"][key]
        for key in CONDITIONS)
    unlabeled_gain = models["Cloud unlabeled Adapter"]["mean_drift"] - models["Edge baseline"]["mean_drift"]
    unlabeled_gap = models["Label-only Adapter"]["mean_drift"] - models["Cloud unlabeled Adapter"]["mean_drift"]
    unlabeled_clean_preserved = (models["Cloud unlabeled Adapter"]["clean"] + .005
                                 >= models["Edge baseline"]["clean"])
    unlabeled_no_drift_regression = all(
        models["Cloud unlabeled Adapter"][key] + .005 >= models["Edge baseline"][key]
        for key in CONDITIONS[1:])
    hybrid_accepted = hybrid_gain >= a.min_cloud_gain and hybrid_no_regression
    unlabeled_accepted = (unlabeled_gain >= a.min_unlabeled_gain
                          and unlabeled_gap <= a.max_unlabeled_gap_to_label
                          and unlabeled_clean_preserved and unlabeled_no_drift_regression)
    verdict = {
        "hybrid_additive_test": {
            "gain_over_label_only": hybrid_gain,
            "passes_gain_gate": hybrid_gain >= a.min_cloud_gain,
            "passes_no_condition_regression_gate": hybrid_no_regression,
            "accepted": hybrid_accepted,
        },
        "unlabeled_refresh_test": {
            "gain_over_edge_baseline": unlabeled_gain,
            "gap_to_label_only_upper_bound": unlabeled_gap,
            "passes_gain_gate": unlabeled_gain >= a.min_unlabeled_gain,
            "passes_upper_bound_gap_gate": unlabeled_gap <= a.max_unlabeled_gap_to_label,
            "passes_clean_preservation_gate": unlabeled_clean_preserved,
            "passes_no_drift_regression_gate": unlabeled_no_drift_regression,
            "accepted": unlabeled_accepted,
        },
        "primary_system_gate": "PASS" if unlabeled_accepted else "FAIL",
    }
    result = {"selected_head": selected, "models": models, "verdict": verdict}
    os.makedirs(os.path.dirname(os.path.abspath(a.output_json)), exist_ok=True)
    with open(a.output_json, "w") as handle: json.dump(result, handle, indent=2)
    header = "| Model | Clean | Illumination | Blur | Noise | Mean drift |\n|---|---:|---:|---:|---:|---:|\n"
    rows = "".join(f"| {name} | {scores['clean']:.2%} | {scores['illumination_1.0']:.2%} | "
                   f"{scores['motion_blur_0.8']:.2%} | {scores['sensor_noise_0.6']:.2%} | "
                   f"{scores['mean_drift']:.2%} |\n" for name, scores in models.items())
    conclusion = (f"\nHybrid - label-only mean drift: {hybrid_gain:+.2%} "
                  f"(**{'PASS' if hybrid_accepted else 'FAIL'}** additive test).  \n"
                  f"Unlabeled - Edge mean drift: {unlabeled_gain:+.2%}; gap to label-only: "
                  f"{unlabeled_gap:+.2%}. Primary unlabeled refresh gate: "
                  f"**{'PASS' if unlabeled_accepted else 'FAIL'}**.\n")
    with open(a.output_md, "w") as handle: handle.write("# Cloud teacher pipeline result\n\n" + header + rows + conclusion)
    print(json.dumps(verdict, indent=2))


if __name__ == "__main__": main()
