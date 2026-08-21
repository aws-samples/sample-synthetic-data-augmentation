"""SageMaker training script for YOLO model training."""
import os
import argparse
import shutil

import torch
import yaml
from ultralytics import YOLO



def _str2bool(value):
    """Parse a boolean flag that may be bare (``--flag``) or valued (``--flag true``).

    SageMaker always emits hyperparameters as ``--key value`` (never a bare flag),
    passing booleans as ``"true"``/``"false"`` strings, while a local CLI run uses the
    bare flag. Both are supported: a bare flag yields ``True`` (via ``const``), and a
    value is interpreted here (an empty string also means ``True``).
    """
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in ("", "1", "true", "yes", "y")


def get_device():
    """Auto-detect best available device."""
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        if count > 1:
            return list(range(count))  # Multi-GPU: [0, 1, 2, 3]
        return 0  # Single GPU
    if torch.backends.mps.is_available():
        return "mps"  # Apple Silicon
    return "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    
    # Hyperparameters
    parser.add_argument("--model-name", type=str, default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--resume", nargs="?", const=True, default=False, type=_str2bool,
                        help="Resume from the run's last.pt checkpoint")
    parser.add_argument("--dataset-yaml", type=str, default="data.yaml",
                        help="Path to dataset YAML relative to data-dir")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    # SageMaker specific arguments
    parser.add_argument("--model-dir", type=str,
                        default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--data-dir", type=str,
                        default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--output-dir", type=str,
                        default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    
    return parser.parse_args()


def update_dataset_yaml_path(yaml_path: str, new_path: str) -> None:
    """Update the 'path' entry in a YOLO dataset YAML file."""
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    config["path"] = new_path

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"Updated dataset YAML path to: {new_path}")


def main():
    args = parse_args()

    # Update dataset YAML to point to SageMaker data directory
    dataset_yaml_path = os.path.join(args.data_dir, args.dataset_yaml)
    update_dataset_yaml_path(dataset_yaml_path, args.data_dir)

    # Initialize model. Ultralytics `resume=True` restores optimizer/epoch state
    # from the checkpoint the model was loaded FROM, so on resume we must load the
    # run's last.pt rather than a fresh pretrained weight.
    last_ckpt = os.path.join(args.output_dir, "yolo_training", "weights", "last.pt")
    if args.resume and os.path.exists(last_ckpt):
        print(f"Resuming from checkpoint: {last_ckpt}")
        model = YOLO(last_ckpt)
    else:
        if args.resume:
            print(f"resume requested but no checkpoint at {last_ckpt}; starting fresh")
        model = YOLO(args.model_name)

    # Auto-detect device
    device = get_device()
    print(f"Using device: {device}")

    # Train
    results = model.train(
        data=dataset_yaml_path,
        epochs=args.epochs,
        imgsz=args.imgsz,
        device=device,
        resume=args.resume and os.path.exists(last_ckpt),
        rect=True,
        project=args.output_dir,
        name="yolo_training",
        patience=args.patience,
        seed=args.seed,
    )
    
    # Save the best model to model_dir for SageMaker model artifacts
    best_model_path = os.path.join(args.output_dir, "yolo_training", "weights", "best.pt")
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, os.path.join(args.model_dir, "best.pt"))
    
    # Also save last checkpoint
    last_model_path = os.path.join(args.output_dir, "yolo_training", "weights", "last.pt")
    if os.path.exists(last_model_path):
        shutil.copy(last_model_path, os.path.join(args.model_dir, "last.pt"))
    
    print(f"Training complete. Model saved to {args.model_dir}")
    return results


if __name__ == "__main__":
    main()
