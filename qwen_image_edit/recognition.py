"""Amazon Rekognition pseudo-labeling for generated images.

Runs person detection on an edited image and reduces overlapping detections
to a clean set of YOLO-ready boxes via greedy NMS.
"""

PERSON_LABELS = {
    "Person", "Human", "Man", "Woman", "Boy", "Girl", "Child", "Adult", "People"
}


def extract_person_boxes(labels: list[dict]) -> list[dict]:
    """Pull person bounding-box instances out of a Rekognition DetectLabels response."""
    boxes = []
    for label in labels:
        names = {label["Name"]} | {alias["Name"] for alias in label.get("Aliases", [])}
        if names & PERSON_LABELS:
            for instance in label.get("Instances", []):
                boxes.append({
                    "label": label["Name"],
                    "confidence": instance["Confidence"],
                    "bbox": instance["BoundingBox"],
                })
    return boxes


def iou(box1: dict, box2: dict) -> float:
    """Intersection-over-union for two Rekognition boxes (Left/Top/Width/Height)."""
    x1 = max(box1["Left"], box2["Left"])
    y1 = max(box1["Top"], box2["Top"])
    x2 = min(box1["Left"] + box1["Width"], box2["Left"] + box2["Width"])
    y2 = min(box1["Top"] + box1["Height"], box2["Top"] + box2["Height"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = box1["Width"] * box1["Height"]
    area2 = box2["Width"] * box2["Height"]
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def dedupe_person_boxes(person_boxes: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Greedy NMS: keep highest-confidence boxes, drop overlaps above the threshold."""
    sorted_boxes = sorted(person_boxes, key=lambda x: x["confidence"], reverse=True)
    kept = []
    for box in sorted_boxes:
        if not any(iou(box["bbox"], k["bbox"]) > iou_threshold for k in kept):
            kept.append({
                "label": "Person",
                "confidence": box["confidence"],
                "bbox": box["bbox"],
            })
    return kept
