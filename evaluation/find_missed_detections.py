"""Find images where the model missed detections for a specific class."""
from pathlib import Path

import click
import cv2
import numpy as np
import yaml
from ultralytics import YOLO


def load_ground_truth_boxes(label_path: Path, class_id: int) -> list[tuple[float, ...]]:
    """Load ground truth boxes for a specific class (YOLO format: cx, cy, w, h normalized)."""
    boxes = []
    if not label_path.exists():
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if parts and int(parts[0]) == class_id:
                boxes.append(tuple(float(x) for x in parts[1:5]))
    return boxes


def yolo_to_pixel(box: tuple[float, ...], img_w: int, img_h: int) -> tuple[int, int, int, int]:
    """Convert YOLO normalized box (cx, cy, w, h) to pixel coords (x1, y1, x2, y2)."""
    cx, cy, w, h = box
    x1 = int((cx - w / 2) * img_w)
    y1 = int((cy - h / 2) * img_h)
    x2 = int((cx + w / 2) * img_w)
    y2 = int((cy + h / 2) * img_h)
    return x1, y1, x2, y2


def get_predictions(results, class_id: int | None, conf_thresh: float, names: dict) -> list[dict]:
    """Get prediction boxes. If class_id is None, return all classes."""
    preds = []
    boxes = results.boxes
    for xyxy, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        cls_int = int(cls)
        if float(conf) >= conf_thresh and (class_id is None or cls_int == class_id):
            preds.append({
                "box": tuple(int(x) for x in xyxy),
                "conf": float(conf),
                "cls": cls_int,
                "name": names.get(cls_int, str(cls_int)),
            })
    return preds


def draw_annotated_image(
    img_path: Path,
    gt_boxes: list[tuple[float, ...]],
    predictions: list[dict],
    all_predictions: list[dict],
    class_name: str,
) -> np.ndarray:
    """Draw ground truth (green), target predictions (blue), other predictions (red)."""
    img = cv2.imread(str(img_path))
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")
    img_h, img_w = img.shape[:2]

    # Draw ground truth boxes in GREEN (what should be detected)
    for box in gt_boxes:
        x1, y1, x2, y2 = yolo_to_pixel(box, img_w, img_h)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(img, f"GT:{class_name}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw OTHER class predictions in RED
    for pred in all_predictions:
        if pred["name"] != class_name:
            x1, y1, x2, y2 = pred["box"]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{pred['name']}:{pred['conf']:.2f}", (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    # Draw TARGET class predictions in BLUE
    for pred in predictions:
        x1, y1, x2, y2 = pred["box"]
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
        cv2.putText(img, f"Pred:{pred['conf']:.2f}", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Add legend
    legend = f"GREEN=GT({class_name}), BLUE=Pred({class_name}), RED=Other"
    cv2.putText(img, legend, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img


@click.command()
@click.option("--model-path", default="./models/best_baseline.pt", help="Path to YOLO model")
@click.option("--data-yaml", default="./test_data/val_dataset.yaml", help="Path to dataset YAML")
@click.option("--class-name", default="person", help="Class name to check for missed detections")
@click.option("--conf", default=0.1, help="Confidence threshold for predictions")
@click.option("--output-dir", default="./missed_detections", help="Directory to save annotated images")
@click.option("--min-gt", default=1, help="Minimum ground truth count to consider")
@click.option("--max-pred-ratio", default=0.5, help="Max pred/gt ratio to count as 'missed' (0.5 = missed >50%)")
@click.option("--device", default=None, help="Device to use (auto-detected if unset)")
@click.option("--limit", default=50, help="Max images to save")
def main(
    model_path: str,
    data_yaml: str,
    class_name: str,
    conf: float,
    output_dir: str,
    min_gt: int,
    max_pred_ratio: float,
    device: str | None,
    limit: int,
) -> None:
    """Find and annotate images where model missed detections for a class."""
    # Load dataset config
    yaml_path = Path(data_yaml).resolve()
    with open(yaml_path) as f:
        config = yaml.safe_load(f)

    # Find class ID
    names = config.get("names", {})
    class_id = None
    for cid, cname in names.items():
        if cname == class_name:
            class_id = int(cid)
            break

    if class_id is None:
        raise click.ClickException(f"Class '{class_name}' not found in {names}")

    # Setup paths. The val/test entry may be a single string (as in the
    # download_testdata.py output) or a list of dirs (as in the generated
    # data-with-synthetic.yaml); take the first entry in the list case.
    data_root = yaml_path.parent / config.get("path", ".")
    val_entry = config.get("val", config.get("test", "images"))
    if isinstance(val_entry, (list, tuple)):
        val_entry = val_entry[0]
    images_dir = data_root / val_entry
    labels_dir = data_root / "labels" / images_dir.name

    print(f"Looking for missed '{class_name}' (class {class_id}) detections")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")

    # Load model
    model = YOLO(model_path)

    # Find image files
    image_files = list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png"))
    print(f"Found {len(image_files)} images")

    # Output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    missed_images = []

    for img_path in image_files:
        # Get ground truth boxes
        label_path = labels_dir / f"{img_path.stem}.txt"
        gt_boxes = load_ground_truth_boxes(label_path, class_id)
        gt_count = len(gt_boxes)

        if gt_count < min_gt:
            continue

        # Run inference
        results = model(str(img_path), device=device, verbose=False)[0]
        predictions = get_predictions(results, class_id, conf, names)
        all_predictions = get_predictions(results, None, conf, names)
        pred_count = len(predictions)

        # Check if missed
        ratio = pred_count / gt_count if gt_count > 0 else 1.0
        if ratio <= max_pred_ratio:
            missed_images.append({
                "path": img_path,
                "label_path": label_path,
                "gt_boxes": gt_boxes,
                "predictions": predictions,
                "all_predictions": all_predictions,
                "gt": gt_count,
                "pred": pred_count,
                "ratio": ratio,
            })

    # Sort by worst misses
    missed_images.sort(key=lambda x: x["ratio"])

    print(f"\nFound {len(missed_images)} images with missed {class_name} detections")

    # Save annotated images
    for i, item in enumerate(missed_images[:limit]):
        annotated = draw_annotated_image(
            item["path"], item["gt_boxes"], item["predictions"],
            item["all_predictions"], class_name
        )
        dest = out_path / f"{i:03d}_gt{item['gt']}_pred{item['pred']}_{item['path'].name}"
        cv2.imwrite(str(dest), annotated)
        other_count = len(item["all_predictions"]) - item["pred"]
        print(f"  {item['path'].name}: GT={item['gt']}, Pred={item['pred']}, Other={other_count}")

    print(f"\nSaved {min(len(missed_images), limit)} annotated images to {out_path}")


if __name__ == "__main__":
    main()
