"""Run YOLO model evaluation and save output plots."""
import os
from pathlib import Path

import click
from ultralytics import YOLO


@click.command()
@click.option("--model-path", default="./models/best_baseline.pt", help="Path to pretrained YOLO model (.pt file)")
@click.option("--data-yaml", required=True, help="Path to validation dataset YAML")
@click.option("--imgsz", default=640, help="Image size for evaluation")
@click.option("--batch", default=16, help="Batch size")
@click.option("--conf", default=0.001, help="Confidence threshold")
@click.option("--iou", default=0.25, help="IoU threshold for NMS")
@click.option("--device", default=None, help="Device to use (e.g., 'cpu', '0', 'mps'); auto-detected if unset")
@click.option("--plots/--no-plots", default=True, help="Generate evaluation plots")
def main(
    model_path: str,
    data_yaml: str,
    imgsz: int,
    batch: int,
    conf: float,
    iou: float,
    device: str | None,
    plots: bool,
) -> None:
    """Load a pretrained YOLO model and run validation with standard plots."""
    model_file = Path(model_path)
    if not model_file.exists():
        raise click.ClickException(f"Model not found: {model_path}")
    
    # Create output directory based on model name
    model_name = model_file.stem
    output_dir = Path(f"{model_name}_output")
    
    # Resolve all paths to absolute before changing directory
    model_path_abs = Path(model_path).resolve()
    data_yaml_path = Path(data_yaml).resolve()
    output_dir_abs = output_dir.resolve()

    print(f"Loading model: {model_path_abs}")
    model = YOLO(str(model_path_abs))

    # Change to YAML's directory so "path: ." resolves correctly
    os.chdir(data_yaml_path.parent)

    print(f"Running validation on: {data_yaml_path.name}")
    print(f"Output directory: {output_dir_abs}")

    # Run validation
    results = model.val(
        data=data_yaml_path.name,
        imgsz=imgsz,
        batch=batch,
        conf=conf,
        iou=iou,
        device=device,
        plots=plots,
        project=str(output_dir_abs.parent),
        name=output_dir_abs.name,
    )
    
    print("\nEvaluation complete!")
    print(f"Results saved to: {output_dir_abs}")
    print("\nMetrics:")
    print(f"  mAP50: {results.box.map50:.4f}")
    print(f"  mAP50-95: {results.box.map:.4f}")
    print(f"  Precision: {results.box.mp:.4f}")
    print(f"  Recall: {results.box.mr:.4f}")


if __name__ == "__main__":
    main()
