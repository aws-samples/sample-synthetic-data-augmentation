"""Launch SageMaker training job for YOLO model."""
from pathlib import Path

from sagemaker.pytorch import PyTorch

_THIS_DIR = str(Path(__file__).resolve().parent)


def launch_training_job(
    s3_training_data: str,
    role: str,
    s3_output_path: str | None = None,
    instance_type: str = "ml.g5.12xlarge",
    model_name: str = "yolo11n.pt",
    epochs: int = 500,
    imgsz: int = 640,
    resume: bool = False,
    job_name_prefix: str = "yolo-training",
    dataset_yaml: str = "data.yaml",
    patience: int = 50,
    seed: int = 42,
):
    """
    Launch a SageMaker training job for YOLO.
    
    Args:
        s3_training_data: S3 URI to training data (should contain dataset.yaml and images)
        role: SageMaker execution role ARN
        s3_output_path: S3 URI for model artifacts (optional, uses default bucket if not provided)
        instance_type: EC2 instance type
        model_name: YOLO model name (e.g., yolo11n.pt, yolo11s.pt)
        epochs: Number of training epochs
        imgsz: Image size for training
        resume: Whether to resume from checkpoint
        job_name_prefix: Prefix for the training job name
        dataset_yaml: Path to dataset YAML relative to training data root
        patience: Early stopping patience (epochs without improvement)
        seed: Random seed for reproducibility
    """
    
    # Define hyperparameters (must be strings for SageMaker). SageMaker always emits
    # "--key value", so the boolean is passed as an explicit "true"/"false" string
    # that train.py parses with _str2bool (a bare "--resume" flag would be rejected).
    hyperparameters: dict[str, str] = {
        "model-name": model_name,
        "epochs": str(epochs),
        "imgsz": str(imgsz),
        "dataset-yaml": dataset_yaml,
        "patience": str(patience),
        "seed": str(seed),
        "resume": str(resume).lower(),
    }
    
    # Create PyTorch estimator
    estimator_kwargs = {
        "entry_point": "train.py",
        "source_dir": _THIS_DIR,
        "role": role,
        "instance_type": instance_type,
        "instance_count": 1,
        "framework_version": "2.1.0",
        "py_version": "py310",
        "hyperparameters": hyperparameters,
        "base_job_name": job_name_prefix,
        "dependencies": [str(Path(_THIS_DIR) / "requirements.txt")],
    }
    if s3_output_path:
        estimator_kwargs["output_path"] = s3_output_path

    estimator = PyTorch(**estimator_kwargs)  # type: ignore[arg-type]
    
    # Start training
    estimator.fit(
        inputs={"training": s3_training_data},
        wait=True,
        logs="All",
    )
    
    return estimator


if __name__ == "__main__":
    import argparse

    # The training data in S3 should look like:
    #
    #   s3://amzn-s3-demo-bucket/datasets/openimages_subset/
    #   ├── data.yaml
    #   ├── images/
    #   │   └── ...
    #   └── labels/
    #       └── ...

    parser = argparse.ArgumentParser()
    parser.add_argument("--s3-data", required=True, help="S3 URI to training data")
    parser.add_argument("--role", required=True, help="SageMaker execution role ARN")
    parser.add_argument("--s3-output", default=None, help="S3 URI for output artifacts")
    parser.add_argument("--instance-type", default="ml.g5.12xlarge")
    parser.add_argument("--model-name", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dataset-yaml", default="data.yaml",
                        help="Path to dataset YAML relative to training data")
    parser.add_argument("--patience", type=int, default=50,
                        help="Early stopping patience (epochs without improvement)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    launch_training_job(
        s3_training_data=args.s3_data,
        role=args.role,
        s3_output_path=args.s3_output,
        instance_type=args.instance_type,
        model_name=args.model_name,
        epochs=args.epochs,
        imgsz=args.imgsz,
        resume=args.resume,
        dataset_yaml=args.dataset_yaml,
        patience=args.patience,
        seed=args.seed,
    )
