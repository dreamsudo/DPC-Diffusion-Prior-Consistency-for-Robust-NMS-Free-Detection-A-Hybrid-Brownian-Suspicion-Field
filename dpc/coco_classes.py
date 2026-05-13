"""COCO class id ↔ name lookup, used by adversarial_class_table and the
confusion matrix.

There are 80 COCO classes used by YOLO26 (the standard 80-class list, not the
91-id internal-COCO list). Index into COCO_CLASSES with the YOLO output
class index 0..79.
"""

from __future__ import annotations

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird",
    "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut",
    "cake", "chair", "couch", "potted plant", "bed",
    "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]

assert len(COCO_CLASSES) == 80, f"expected 80 classes, got {len(COCO_CLASSES)}"

CLASS_NAME_BY_ID = {i: name for i, name in enumerate(COCO_CLASSES)}
CLASS_ID_BY_NAME = {name: i for i, name in enumerate(COCO_CLASSES)}


def class_name(class_id: int) -> str:
    """Return COCO class name for a YOLO class id, or 'unknown' if out of range."""
    if 0 <= class_id < len(COCO_CLASSES):
        return COCO_CLASSES[class_id]
    return f"unknown(id={class_id})"

# ─── COCO category_id (1..90 with gaps) → dense class_id (0..79) ──────────────
# COCO's instances_*.json uses category_id 1..90 but only 80 are populated;
# the 11 missing ids are: 12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91.
# YOLO26 (and most COCO-pretrained detectors) emit dense class indices 0..79.
# The mapping below is the standard COCO 80-class convention.
#
# Source: https://github.com/cocodataset/cocoapi (lvis_v1_train.json categories,
# and standard COCO 2017 references). Verified against ultralytics yolo data
# config "coco.yaml".

# COCO category_id → dense class_id
COCO_CATEGORY_ID_TO_CLASS_ID = {
    1: 0,   2: 1,   3: 2,   4: 3,   5: 4,   6: 5,   7: 6,   8: 7,
    9: 8,   10: 9,  11: 10, 13: 11, 14: 12, 15: 13, 16: 14, 17: 15,
    18: 16, 19: 17, 20: 18, 21: 19, 22: 20, 23: 21, 24: 22, 25: 23,
    27: 24, 28: 25, 31: 26, 32: 27, 33: 28, 34: 29, 35: 30, 36: 31,
    37: 32, 38: 33, 39: 34, 40: 35, 41: 36, 42: 37, 43: 38, 44: 39,
    46: 40, 47: 41, 48: 42, 49: 43, 50: 44, 51: 45, 52: 46, 53: 47,
    54: 48, 55: 49, 56: 50, 57: 51, 58: 52, 59: 53, 60: 54, 61: 55,
    62: 56, 63: 57, 64: 58, 65: 59, 67: 60, 70: 61, 72: 62, 73: 63,
    74: 64, 75: 65, 76: 66, 77: 67, 78: 68, 79: 69, 80: 70, 81: 71,
    82: 72, 84: 73, 85: 74, 86: 75, 87: 76, 88: 77, 89: 78, 90: 79,
}
assert len(COCO_CATEGORY_ID_TO_CLASS_ID) == 80
assert set(COCO_CATEGORY_ID_TO_CLASS_ID.values()) == set(range(80))


def coco_category_to_class(cat_id: int) -> int:
    """Map a COCO category_id (1..90 with gaps) to YOLO26 class_id (0..79).

    Raises KeyError on unknown id; caller should treat that as "skip this
    annotation" (it indicates an annotation outside the 80-class subset, which
    is rare but possible in older COCO splits).
    """
    return COCO_CATEGORY_ID_TO_CLASS_ID[cat_id]
