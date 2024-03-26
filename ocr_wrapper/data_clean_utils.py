from __future__ import annotations

from .bbox import BBox
import re
from .bbox_utils import split_bbox
from typing import Optional


def split_date_boxes(bboxes: list[BBox], confidences: Optional[list] = None) -> list[BBox]:
    """
    Splits date boxes that contain a date range of the format "dd/mm/yyyy - dd/mm/yyyy" into three separate boxes.

    Args:
        bboxes (list[BBox]): The bboxes to filter.

    Returns:
        list[BBox]: The filtered bboxes.
    """
    if confidences is not None and len(bboxes) != len(confidences):
        raise ValueError("The length of the bboxes and confidences lists must be equal.")

    # Create dummy confidences if none are given. Makes the rest of the code more consistent
    if confidences is None:
        working_confidences = [0 for i in range(len(bboxes))]
    else:
        working_confidences = confidences

    date_range_pattern = (
        r"^\s*\d{1,2}\s*[/\.]\s*\d{1,2}\s*[/\.]\s*\d{4}\s*-\s*\d{1,2}\s*[/\.]\s*\d{1,2}\s*[/\.]\s*\d{4}\s*$"
    )

    filtered_bboxes = []
    new_confidences = []
    for bbox, confidence in zip(bboxes, working_confidences):
        text = bbox.text
        if text is not None and re.match(date_range_pattern, text):
            date1, date2 = text.split("-")
            date1, date2 = date1.strip(), date2.strip()
            # Info: The split points have been determined empirically
            bbox1, bbox2 = split_bbox(bbox, 0.49)
            bbox1_2, bbox2_2 = split_bbox(bbox2, 0.07)  # Split the second bbox again to get a box for the "-"
            bbox1.text = date1
            bbox1_2.text = "-"
            bbox2_2.text = date2
            filtered_bboxes.append(bbox1)
            filtered_bboxes.append(bbox1_2)
            filtered_bboxes.append(bbox2_2)
            # Confidences are just repeated three times for the three new boxes
            new_confidences.extend([confidence, confidence, confidence])
        else:
            filtered_bboxes.append(bbox)
            new_confidences.append(confidence)

    if confidences is None:
        return filtered_bboxes
    else:
        return filtered_bboxes, new_confidences
