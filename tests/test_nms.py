"""Test NMS — class-aware NMS keeps class-disjoint boxes, suppresses
class-equal overlapping boxes."""

import pytest

torch = pytest.importorskip("torch")


def test_box_iou_basic():
    from dpc.nms import box_iou

    a = torch.tensor([[0, 0, 100, 100]], dtype=torch.float)
    b = torch.tensor([
        [0, 0, 100, 100],     # full overlap → IoU 1
        [50, 50, 150, 150],   # partial
        [200, 200, 300, 300], # zero
    ], dtype=torch.float)
    iou = box_iou(a, b)
    assert iou.shape == (1, 3)
    assert torch.isclose(iou[0, 0], torch.tensor(1.0), atol=1e-5)
    # Partial: inter = 50*50 = 2500; union = 100*100 + 100*100 - 2500 = 17500
    assert torch.isclose(iou[0, 1], torch.tensor(2500/17500), atol=1e-3)
    assert torch.isclose(iou[0, 2], torch.tensor(0.0), atol=1e-5)


def test_box_iou_empty():
    from dpc.nms import box_iou

    a = torch.zeros((0, 4))
    b = torch.tensor([[0, 0, 10, 10]], dtype=torch.float)
    iou = box_iou(a, b)
    assert iou.shape == (0, 1)


def test_nms_suppresses_overlapping():
    from dpc.nms import nms

    boxes = torch.tensor([
        [0, 0, 100, 100],     # high score
        [10, 10, 110, 110],   # high IoU with #0, lower score
        [200, 200, 300, 300], # disjoint
    ], dtype=torch.float)
    scores = torch.tensor([0.9, 0.8, 0.7])
    keep = nms(boxes, scores, iou_threshold=0.5)
    keep_set = set(keep.tolist())
    assert 0 in keep_set  # highest score always kept
    assert 1 not in keep_set  # suppressed by #0
    assert 2 in keep_set  # disjoint, kept


def test_nms_keeps_disjoint():
    from dpc.nms import nms

    boxes = torch.tensor([
        [0, 0, 50, 50],
        [100, 100, 150, 150],
        [200, 200, 250, 250],
    ], dtype=torch.float)
    scores = torch.tensor([0.5, 0.8, 0.3])
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert sorted(keep.tolist()) == [0, 1, 2]


def test_nms_empty():
    from dpc.nms import nms

    boxes = torch.zeros((0, 4))
    scores = torch.zeros((0,))
    keep = nms(boxes, scores, iou_threshold=0.5)
    assert keep.shape == (0,)


def test_class_aware_nms_keeps_class_disjoint_overlap():
    """Two overlapping boxes with DIFFERENT classes → both kept."""
    from dpc.nms import class_aware_nms

    boxes = torch.tensor([
        [0, 0, 100, 100],
        [10, 10, 110, 110],   # IoU > 0.5 with #0, but DIFFERENT class
    ], dtype=torch.float)
    scores = torch.tensor([0.9, 0.8])
    classes = torch.tensor([5, 7])  # different classes
    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.5)
    assert sorted(keep.tolist()) == [0, 1]


def test_class_aware_nms_suppresses_same_class_overlap():
    """Two overlapping boxes with SAME class → lower-scored is suppressed."""
    from dpc.nms import class_aware_nms

    boxes = torch.tensor([
        [0, 0, 100, 100],
        [10, 10, 110, 110],   # same class as #0
    ], dtype=torch.float)
    scores = torch.tensor([0.9, 0.8])
    classes = torch.tensor([5, 5])
    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.5)
    assert keep.tolist() == [0]


def test_class_aware_nms_multiple_classes_with_intra_class_dups():
    from dpc.nms import class_aware_nms

    boxes = torch.tensor([
        [0, 0, 100, 100],     # class 5
        [10, 10, 110, 110],   # class 5 (suppressed by #0)
        [200, 200, 300, 300], # class 7
        [205, 205, 305, 305], # class 7 (suppressed by #2)
        [400, 400, 500, 500], # class 7 (kept — disjoint)
    ], dtype=torch.float)
    scores = torch.tensor([0.9, 0.8, 0.85, 0.7, 0.6])
    classes = torch.tensor([5, 5, 7, 7, 7])
    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.5)
    # Should keep 0, 2, 4
    assert sorted(keep.tolist()) == [0, 2, 4]


def test_class_aware_nms_empty():
    from dpc.nms import class_aware_nms

    boxes = torch.zeros((0, 4))
    scores = torch.zeros((0,))
    classes = torch.zeros((0,), dtype=torch.long)
    keep = class_aware_nms(boxes, scores, classes, iou_threshold=0.5)
    assert keep.shape == (0,)
