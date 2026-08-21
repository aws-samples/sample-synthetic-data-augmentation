"""Extract per-class mAP breakdown from YOLO model evaluation.

This is the key analysis: it shows whether synthetic people actually improve
*person* detection specifically.

Usage:
    python extract_per_class.py --data-yaml ./test_data/val_dataset.yaml
    python extract_per_class.py --data-yaml ./test_data/val_dataset.yaml --output results.json
"""
import json
import os
from pathlib import Path

import click
from ultralytics import YOLO

DEFAULT_MODELS = {
    "baseline": "best_baseline.pt",
    "synthetic": "best_synthetic.pt",
}


def parse_models_spec(spec: str) -> dict[str, str]:
    """Parse comma-separated name:path pairs into a dict.

    Example: "baseline:/path/to/model.pt,synthetic:/other/model.pt"
    """
    models = {}
    for pair in spec.split(","):
        pair = pair.strip()
        if ":" not in pair:
            raise click.BadParameter(
                f"Invalid model spec '{pair}'. Expected format: name:path"
            )
        name, path = pair.split(":", 1)
        models[name.strip()] = path.strip()
    return models


@click.command()
@click.option("--data-yaml", required=True, help="Path to validation dataset YAML")
@click.option("--models-dir", default=None, help="Directory containing model .pt files (used with DEFAULT_MODELS)")
@click.option("--models", default=None, help="Comma-separated name:path pairs (e.g. baseline:/path/model.pt,synthetic:/path/model2.pt)")
@click.option("--iou", default=0.25, help="IoU threshold for NMS (must match run_eval.py for comparable numbers)")
@click.option("--device", default=None, help="Device (auto-detected if unset)")
@click.option("--output", default=None, help="Optional JSON output path")
def main(data_yaml: str, models_dir: str | None, models: str | None, iou: float, device: str | None, output: str | None) -> None:
    """Run eval on all models and print per-class mAP breakdown."""
    data_path = Path(data_yaml).resolve()
    output_path = Path(output).resolve() if output else None
    all_results: dict[str, dict] = {}

    # Determine which models to evaluate. Resolve all model paths to absolute
    # BEFORE the chdir below, so relative paths (e.g. "./models/best_baseline.pt")
    # are interpreted against the current working directory, not the YAML's dir.
    if models:
        model_dict = {name: str(Path(path).resolve()) for name, path in parse_models_spec(models).items()}
    elif models_dir:
        models_path = Path(models_dir).resolve()
        model_dict = {name: str(models_path / filename) for name, filename in DEFAULT_MODELS.items()}
    else:
        raise click.UsageError("Either --models or --models-dir must be provided.")

    # chdir so "path: ." in YAML resolves correctly
    os.chdir(data_path.parent)

    for name, model_path_str in model_dict.items():
        model_file = Path(model_path_str)
        if not model_file.exists():
            print(f"Skipping {name}: {model_file} not found")
            continue

        model = YOLO(str(model_file))
        results = model.val(data=data_path.name, iou=iou, device=device, verbose=False)

        class_names = results.names
        # ultralytics reports per-class metrics as positional arrays ordered by
        # `ap_class_index` (the class ids actually present in the eval), NOT indexed
        # by class id. `class_result(pos)` returns (precision, recall, ap50, ap) for
        # that position. Classes with no predictions/labels are absent from
        # `ap_class_index`; we still list them so the table has a stable column set.
        per_class = {cls_name: {"mAP50": None, "mAP50-95": None, "precision": None, "recall": None}
                     for cls_name in class_names.values()}
        for pos, class_id in enumerate(results.box.ap_class_index):
            cls_name = class_names[int(class_id)]
            p, r, ap50, ap = results.box.class_result(pos)
            per_class[cls_name] = {
                "mAP50": float(ap50),
                "mAP50-95": float(ap),
                "precision": float(p),
                "recall": float(r),
            }

        all_results[name] = {
            "aggregate": {
                "mAP50": float(results.box.map50),
                "mAP50-95": float(results.box.map),
                "precision": float(results.box.mp),
                "recall": float(results.box.mr),
            },
            "per_class": per_class,
        }

    if not all_results:
        raise click.ClickException(
            "No models were evaluated (all paths missing?). Nothing to report."
        )

    # Class column order, taken from the first evaluated model.
    class_cols = list(next(iter(all_results.values()))["per_class"].keys())

    def print_metric(title: str, agg_key: str, per_key: str) -> None:
        header = f"{'Model':<15} {'Aggregate':<12} " + " ".join(f"{c:<12}" for c in class_cols)
        print(f"\n{title}")
        print("-" * len(header))
        print(header)
        print("-" * len(header))
        for name, res in all_results.items():
            agg = f"{res['aggregate'][agg_key]:.4f}"
            cells = []
            for c in class_cols:
                v = res["per_class"][c][per_key]
                cells.append(f"{v:.4f}" if v is not None else "N/A")
            per = " ".join(f"{cell:<12}" for cell in cells)
            print(f"{name:<15} {agg:<12} {per}")

    print("\n" + "=" * 80)
    print("PER-CLASS mAP@0.5 BREAKDOWN (aggregate column is dataset-wide mAP@0.5)")
    print("=" * 80)
    print_metric("PER-CLASS mAP@0.5", "mAP50", "mAP50")
    print_metric("PER-CLASS mAP@0.5:0.95", "mAP50-95", "mAP50-95")
    print_metric("PER-CLASS PRECISION", "precision", "precision")
    print_metric("PER-CLASS RECALL", "recall", "recall")

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(all_results, indent=2))
        print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
