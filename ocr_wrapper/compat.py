"""
Functions for converting between the old and new way that OCR results are stored
"""

from __future__ import annotations
from ocr_wrapper.bbox import BBox


def bboxs2dicts(bboxes: list[BBox], confidences: list) -> list[dict]:
    assert len(bboxes) == len(confidences)
    res = []

    for bb, confidence in zip(bboxes, confidences):
        assert not bb.in_pixels
        res.append(
            {
                "bbox": bb,
                "text": bb.text,
                "confidence": confidence,
            }
        )

    return res


def dicts2bboxs(dicts: list[dict]) -> tuple[list[BBox], list]:
    bboxes = []
    confidences = []

    for d in dicts:
        bboxes.append(d["bbox"])
        confidences.append(d["confidence"])

    return bboxes, confidences
