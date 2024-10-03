"""
Additional functions for working with BBox objects.
Mainly here for compatibilty with the new ocr_wrapper package.
"""

from __future__ import annotations

import math
from functools import lru_cache

import rtree
from opentelemetry import trace

from .bbox import BBox
from .bbox_order import get_ordered_bboxes_idxs

tracer = trace.get_tracer(__name__)


def _interpolate_point(A: tuple[float, float], B: tuple[float, float], ratio: float) -> tuple[float, float]:
    """Returns the point along the line AB at the given ratio"""
    return (A[0] + ratio * (B[0] - A[0]), A[1] + ratio * (B[1] - A[1]))


def split_bbox(bbox: BBox, ratio: float) -> tuple[BBox, BBox]:
    """
    Splits a bounding box along the longer edge at the given ratio.

    Args:
        bbox: The bounding box.
        ratio: The ratio at which to split the bounding box.

    Returns:
        A tuple containing the two resulting bounding boxes. Text, label, and is_pixels are copied
        from the original bounding box.
    """
    # Calculate lengths of top and side edges
    top_length = math.sqrt((bbox.TRx - bbox.TLx) ** 2 + (bbox.TRy - bbox.TLy) ** 2)
    side_length = math.sqrt((bbox.BLx - bbox.TLx) ** 2 + (bbox.BLy - bbox.TLy) ** 2)

    # Determine longer edge and split points
    if top_length >= side_length:
        # Splitting along the top edge
        new_top_point = _interpolate_point((bbox.TLx, bbox.TLy), (bbox.TRx, bbox.TRy), ratio)
        new_bottom_point = _interpolate_point((bbox.BLx, bbox.BLy), (bbox.BRx, bbox.BRy), ratio)
        bbox1 = BBox(
            bbox.TLx,
            bbox.TLy,
            new_top_point[0],
            new_top_point[1],
            new_bottom_point[0],
            new_bottom_point[1],
            bbox.BLx,
            bbox.BLy,
        )
        bbox2 = BBox(
            new_top_point[0],
            new_top_point[1],
            bbox.TRx,
            bbox.TRy,
            bbox.BRx,
            bbox.BRy,
            new_bottom_point[0],
            new_bottom_point[1],
        )
    else:
        # Splitting along the side edge
        new_left_point = _interpolate_point((bbox.TLx, bbox.TLy), (bbox.BLx, bbox.BLy), ratio)
        new_right_point = _interpolate_point((bbox.TRx, bbox.TRy), (bbox.BRx, bbox.BRy), ratio)
        bbox1 = BBox(
            bbox.TLx,
            bbox.TLy,
            new_right_point[0],
            new_right_point[1],
            bbox.TRx,
            bbox.TRy,
            new_left_point[0],
            new_left_point[1],
        )
        bbox2 = BBox(
            new_left_point[0],
            new_left_point[1],
            new_right_point[0],
            new_right_point[1],
            bbox.BRx,
            bbox.BRy,
            bbox.BLx,
            bbox.BLy,
        )

    return bbox1, bbox2


@lru_cache(maxsize=16000)
def bbox_intersection_area_ratio(bb1: BBox, bb2: BBox) -> float:
    """Returns the area of the intersection of BBox `bb1` with BBox `bb2` as a ratio of the area of `bb1`.

    i.e. max is 1.0, min is 0.0
    """
    self_poly = bb1.get_shapely_polygon()
    that_poly = bb2.get_shapely_polygon()
    # Sometimes the polygons are invalid (usually because of self-intersection), in which case we return 0.0. We should have a closer look why this happens, but for now it seems to be a rare occurence and this is a quick fix that doesn't seem to affect the results much.
    if not self_poly.is_valid or not that_poly.is_valid:
        return 0.0
    if self_poly.intersects(that_poly):
        inter_poly = self_poly.intersection(that_poly)
        return inter_poly.area / self_poly.area
    else:
        return 0.0


def _find_overlapping_bboxes(bbox: BBox, bboxes: list[BBox], idx: rtree.index.Index, threshold: float) -> list[BBox]:
    """
    This function takes a bbox, a list of bboxes, an rtree index, and a threshold value as input.

    It returns a list of bboxes that have overlapping areas with the input bbox, based on the threshold value (either A overlaps B or B overlaps A (or both) by more than the threshold).
    """
    overlapping_bboxes = [bbox]

    polygon = bbox.get_shapely_polygon()
    potential_matches = [bboxes[pos] for pos in idx.intersection(polygon.bounds)]
    for mtch in potential_matches:
        if bbox == mtch:  # don't compare a bbox with itself
            continue

        overlap1 = bbox_intersection_area_ratio(bbox, mtch)
        overlap2 = bbox_intersection_area_ratio(mtch, bbox)
        if overlap1 > threshold or overlap2 > threshold:
            overlapping_bboxes.append(mtch)

    return overlapping_bboxes


def group_overlapping_bboxes(bboxes: list[BBox], threshold: float) -> list[list[BBox]]:
    """
    Group the bounding boxes that have an overlap greater than the given threshold.

    A bounding box counts as an overlap if either A overlaps B or B overlaps A (or both) by more than the threshold.

    Args:
        bboxes: The bounding boxes.
        threshold (float): The threshold for the percentage of overlap.

    Returns:
        A list of lists containing the groups of overlapping bounding boxes. Be aware: A bounding box can occur multiple times in different groups.
    """
    # Create an rtree index for the polygons so we can quickly find intersecting polygons
    idx = rtree.index.Index()
    bbox2treeid = {}  # Needed so we can delete bboxes from the index`
    for i, bbox in enumerate(bboxes):
        idx.insert(i, bbox.get_shapely_polygon().bounds)
        bbox2treeid[bbox] = i

    working_bboxes = bboxes.copy()

    groups = []
    while len(working_bboxes) > 0:
        # Find overlapping bboxes for the first bbox in the list
        bbox = working_bboxes.pop(0)
        overlapping_bboxes = _find_overlapping_bboxes(bbox, bboxes, idx, threshold)
        groups.append(overlapping_bboxes)

        # Remove the bboxes from the index
        for bbox in overlapping_bboxes:
            idx.delete(bbox2treeid[bbox], bbox.get_shapely_polygon().bounds)
            del bbox2treeid[bbox]

        # Remove the bboxes from the list
        for bbox in overlapping_bboxes:
            working_bboxes.remove(bbox)

    return groups


@tracer.start_as_current_span("merge_bbox_lists")
def merge_bbox_lists(
    bboxes_a: list[BBox],
    bboxes_b: list[BBox],
    document_width: int,
    document_height: int,
) -> list[BBox]:
    """
    Given the list of bboxes_a as well as bboxes_b, inserts the bboxes_b into the bboxes_a list at the correct position.

    For this, the order of bboxes_a are used as the reference. The position of the bboxes_b are determined by
    merging the two lists and sorting them using the order_bboxes function, which returns indexes of a fully sorted list.
    This sorting is not used to sort the bboxes, but to determine the position of the azure bboxes in the bboxes_a list.
    """
    bboxes_a_idxs = list(range(len(bboxes_a)))
    bboxes_b_idxs = list(range(len(bboxes_a), len(bboxes_a) + len(bboxes_b)))

    merged_bboxes = bboxes_a + bboxes_b
    sorted_idxs = get_ordered_bboxes_idxs(
        merged_bboxes, document_width=document_width, document_height=document_height
    )
    merged_bbox_idxs = merge_idx_lists(bboxes_a_idxs, bboxes_b_idxs, sorted_idxs)
    merged_bboxes = [merged_bboxes[i] for i in merged_bbox_idxs]

    return merged_bboxes


@tracer.start_as_current_span("merge_bbox_lists_with_confidences")
def merge_bbox_lists_with_confidences(
    bboxes_a: list[BBox],
    confidences_a: list[float],
    bboxes_b: list[BBox],
    confidences_b: list[float],
    document_width: int,
    document_height: int,
) -> tuple[list[BBox], list[float]]:
    """
    Given the list of bboxes_a as well as bboxes_b, and their corresponding confidences confidences_a and confidences_b, inserts the bboxes_b into the bboxes_a list at the correct position and also returned the merged confidences.

    For this, the order of bboxes_a are used as the reference. The position of the bboxes_b are determined by
    merging the two lists and sorting them using the order_bboxes function, which returns indexes of a fully sorted list.
    This sorting is not used to sort the bboxes, but to determine the position of the azure bboxes in the bboxes_a list.
    """
    assert len(bboxes_a) == len(confidences_a)
    assert len(bboxes_b) == len(confidences_b)

    bboxes_a_idxs = list(range(len(bboxes_a)))
    bboxes_b_idxs = list(range(len(bboxes_a), len(bboxes_a) + len(bboxes_b)))

    merged_bboxes = bboxes_a + bboxes_b
    merged_confidences = confidences_a + confidences_b

    sorted_idxs = get_ordered_bboxes_idxs(
        merged_bboxes, document_width=document_width, document_height=document_height
    )
    merged_idxs = merge_idx_lists(bboxes_a_idxs, bboxes_b_idxs, sorted_idxs)
    merged_bboxes = [merged_bboxes[i] for i in merged_idxs]
    merged_confidences = [merged_confidences[i] for i in merged_idxs]

    return merged_bboxes, merged_confidences


def merge_idx_lists(raw_a, raw_b, sorted_ab):
    """
    We merge two lists of indexes, raw_a and raw_b, into a single list. The order of the indexes in raw_a follow the
    order given in raw_a, but elements from raw_b can be inserted in between the elements of raw_a. The order of the
    elements in raw_b is determined by the order of the elements in sorted_ab.
    """
    assert len(raw_a) + len(raw_b) == len(sorted_ab)

    if len(sorted_ab) == 0:
        return []

    result = []
    raw_a_set = set(raw_a)
    raw_b_set = set(raw_b)
    raw_a_left = raw_a.copy()
    raw_a_left.reverse()

    # Create a map of each element in sorted_ab to the one following it
    # e.g. [1, 2, 3, 4] -> {1: 2, 2: 3, 3: 4}
    next_sorted_map = {sorted_ab[i]: sorted_ab[i + 1] for i in range(len(sorted_ab) - 1)}

    # Select the first element to add
    if sorted_ab[0] in raw_b_set:  # If the first element in sorted_ab is in raw_b, we start with that
        last_added = sorted_ab[0]
        raw_b_set.remove(last_added)
    else:  # Otherwise, we start with the first element in raw_a
        last_added = raw_a[0]
        raw_a_set.remove(last_added)
        raw_a_left.pop()
    result.append(last_added)

    # Add all the other elements
    while len(raw_a_set) != 0 or len(raw_b_set) != 0:
        next_in_sorted = next_sorted_map.get(last_added, -1)
        if next_in_sorted in raw_b_set:  # If the next element in sorted_ab is in raw_b, we follow the sorted order ...
            last_added = next_in_sorted
            raw_b_set.remove(last_added)
        else:  # ... otherwise we keep the order given in raw_a
            last_added = raw_a_left.pop()
            raw_a_set.remove(last_added)

        result.append(last_added)

    assert len(result) == len(raw_a) + len(raw_b)

    return result
