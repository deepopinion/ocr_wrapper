"""
This module contains functions to determine the order of bounding boxes in a document.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Literal, Union
from unicodedata import bidirectional as char_type

import numpy as np

import ocr_wrapper

DEGREE2RADIAN = 2 * np.pi / 360
#  We allow a certain tilt angle. However, the recently added tilt correction executed before the OCR-Scan removed the need
# of a wide range scan of tilt angles. So, we only check with a the range of - 1° ... + 1°.
# The MAX_TILT_ANGLE is measured in degrees. (circumference = 360°)
MAX_TILT_ANGLE = 11
# Next we determine how many intermediate positions we test between -MAX_TILT_FRACTION to +MAX_TILT_FRACTION
# since we only scan the range of - 1° ... + 1°, we set NB_TILT to 21 (i.e. steps of 0.1°)
NB_TILT = 21
MAX_TILT_FRACTION = np.tan(MAX_TILT_ANGLE * DEGREE2RADIAN)

# To determine the writing direction of a document, we use the codes of char_type (unicodedata.bidirectional).
# The bidirectional codes are:
# "L" for Left-to-Right
# "R" for Right-to-Left
# "AL" for Arabic Letter
# "EN" for European Number
# "AN" for Arabic Number
# "ET" for European Number Terminator
# "ON" for Other Neutrals
# The remaining code words can not appear in a OCR scan and if for any reason they do, we treat them as "ON".

# The following numbers are chosen such that the most dominant type gets the lowest number.
INITIAL_TYPE_CODES = {"ON": 5, "ET": 4, "AN": 3, "EN": 2, "R": 1, "AL": 1, "L": 0}
TYPE_CODES = defaultdict(lambda: 5, INITIAL_TYPE_CODES)


class BoxStatus(Enum):
    """
    A box starts out as with the status REGULAR and stays it, except it is identified as an OCR error - then its
    status turns into IGNORE. However, an IGNORE box might be split into new boxes and added to the other boxes. Such a box is
    marked as ADDED. Hence, the status ADDED indicates a box which is not part of the OCR scan result.
    """

    REGULAR = "regular"
    IGNORE = "ignore"
    ADDED = "added"


@dataclass
class DocumentProperties:
    full_height = None
    full_width = None
    half_width = None
    tilt_factor = None
    median_height = None  # height of average bounding box
    median_char_width = None
    char_space = None  # estimated space (width) occupied by an average character
    space_width = None  # estimated width of average space character
    space_width_height_ratio = None  # estimated value of character space width relative to bounding box height
    line_center = None  # array of line_centers
    # A list of boxes with the status ignore
    ignore_box_list = []
    # The dominant writing direction of the document: left-to-right (ltr) or right-to-left (rtl)
    writing_direction: Literal["ltr", "rtl"] = "ltr"
    # The "writing_direction" is chosen as the dominant one. It doesn't exclude that the inverse direction occurs, too.
    # The "is_bidirectional" flag is set to True, if the document contains text in both directions.
    is_bidirectional: bool = False


@dataclass
class BoxProperties:
    id = None  # order of appearance in sample
    text = None
    left = None
    right = None
    width = None
    top = None
    bottom = None
    height = None
    tilted_top = None
    tilted_bottom = None
    x = None  # average x - i.e. center of box
    x_center = None  # box center distance to page center
    y = None  # average y - i.e center
    tilted_y = None  # y after tilt correction
    int_top = None  # integer value for top (number of pixel pixel)
    int_bottom = None
    int_height = None
    is_word = None  # = is made letters only and has at least two of them
    # Once the lines are found, we set the following values
    line_id = None
    pos_in_line = None
    # We might declare a box as OCR artefact and split it. We use the status REGULAR, IGNORE, ADDED to describe a box.
    # The status ADDED describes a box which is added as part of an error correction and not part of the original OCR scan -
    # it has no id. IGNORE is for boxes, which are OCR artefacts
    status = BoxStatus.REGULAR
    # The type of the box is determined by the characters it contains. The type is used to determine the writing direction.
    # In case a box contains a mix of different types, the most dominant type is used. The numbers are chosen such that the
    # most dominant type gets the lowest number - so we can adopt the minimal character value as the type of the box.
    # 0 for L, 1 for R, 2 for EN, 3 for AN, 4 for ET, 5 for ON
    type: Literal[0, 1, 2, 3, 4, 5] = None


def _harmonize_bbox(
    ocr_item: Union[ocr_wrapper.BBox, dict[str, Any]],
    document_width: int,
    document_height: int,
) -> tuple[ocr_wrapper.BBox, str]:
    """Adapter function to bring both v0 and v1 OCR results to the same format.

    The output object is a tuple of:
      1. ocr_wrapper.BBox object, guaranteed to be in normalized coordinates
      2. The text of the OCR result
    """
    if isinstance(ocr_item, dict) and "bbox" in ocr_item and "text" in ocr_item:
        # A v1 OCR result
        box = ocr_item["bbox"]
        text = ocr_item["text"]
        return box, text
    if hasattr(ocr_item, "text") and hasattr(ocr_item, "to_normalized"):
        # A v0 OCR result
        ocr_item = ocr_item.to_normalized(document_width, document_height)
        return ocr_item, ocr_item.text or ""
    raise ValueError("Invalid OCR item type. Expected an output from OcrWrapper.ocr.")


def _extract_properties(
    ocr_bbox_lst: Union[list[ocr_wrapper.BBox], list[dict[str, Any]]],
    document_width: int,
    document_height: int,
) -> tuple[DocumentProperties, list[BoxProperties]]:
    """
    Extracts properties of the entire document as well as for each individual bounding box.

    The output of the ocr_wrapper serves as input.
    """
    harmonized_bboxes = [_harmonize_bbox(b, document_width, document_height) for b in ocr_bbox_lst]

    box_lst = []
    document_prop = DocumentProperties()
    document_prop.full_height = max(200, document_height)
    document_prop.full_width = max(200, document_width)
    document_prop.half_width = document_width / 2
    type_count = np.zeros(6, dtype=int)
    box_lst = []
    char_width_lst = []
    height_lst = []
    nb_words = 0
    for id, (bbx, text) in enumerate(harmonized_bboxes):
        box = BoxProperties()
        box_lst.append(box)
        box.id = id
        box.text = text
        box.left = document_prop.full_width * max(0, min(1, (bbx.TLx + bbx.BLx) / 2))
        box.right = document_prop.full_width * max(0, min(1, (bbx.TRx + bbx.BRx) / 2))
        box.width = max(1, box.right - box.left)
        box.x = (box.right + box.left) / 2
        # Now, we measure x form the center
        box.x_center = box.x - document_prop.half_width
        box.top = document_prop.full_height * max(0, min(1, (bbx.TLy + bbx.TRy) / 2))
        box.bottom = max(
            box.top + 1,
            document_prop.full_height * max(0, min(1, (bbx.BLy + bbx.BRy) / 2)),
        )
        box.height = max(1, box.bottom - box.top)
        box.y = (box.top + box.bottom) / 2
        box.int_top = int(round(box.top))
        box.int_bottom = int(round(box.bottom))
        box.int_height = max(1, box.int_bottom - box.int_top)
        # We need to know the writing direction of each box and the document. To do so, we look at the used characters.
        type_set = set(char_type(c) for c in box.text)
        # We defined the number-values of TYPE_CODES such that the minimum value is the most dominant type.
        box.type = min([TYPE_CODES[c] for c in type_set]) if len(type_set) > 0 else 0
        type_count[box.type] += 1
        # What comes below is needed for statistics and ultimately to estimate space character width
        # Assuming that two words are separated by a space is mostly correct, while for numbers, we might have tiny
        # spaces or noting, when a unit follows. So, we only use letter-words for space width estimation
        box.is_word = box.text.isalpha() and len(box.text) > 1
        if box.is_word:
            nb_words += 1
            height_lst.append(box.height)
            nb_char = len(box.text)
            avg_width = box.width / nb_char
            # When estimating the space character width, we might have outlier (e.g. headlines). Therefore, the median
            # seems more appropriate than the average. For the correct median, we have to add multiple "avg_width"
            # according to the number of characters in a word.
            char_width_lst.extend([avg_width] * nb_char)
    if nb_words > 10:  # cutoff for reliable statistic (arbitrary chosen)
        document_prop.median_height = np.median(height_lst)
        document_prop.median_char_width = np.median(char_width_lst)
    # else stays "None"
    # We determine the writing direction of the document as the writing direction of the majority of the boxes.
    # This is contrary to the standard bidi algorithm, which uses the first strong character to determine the
    # writing direction.
    document_prop.writing_direction = "rtl" if type_count[TYPE_CODES["R"]] > type_count[TYPE_CODES["L"]] else "ltr"
    # While Arabic numbers in Latin script has no special treatment, Latin numbers in Arabic script are treated special.
    ltr = type_count[TYPE_CODES["L"]] + type_count[TYPE_CODES["EN"]] + type_count[TYPE_CODES["ET"]]
    rtl = type_count[TYPE_CODES["R"]]
    # For documents with unique writing direction, we can allow at maximally one box with opposite direction.
    # This single exception can be tolerated, because the OCR scanner returns the letters of each box in the correct order.
    # We need at least two words in opposite direction to make a mistake neglecting the correct order of words.
    document_prop.is_bidirectional = min(ltr, rtl) > 1

    # We sort the boxes once according to the horizontal position (we know the original position thanks to box.id).
    # This one sort frees us from later sorts of the boxes on the found text lines. We simply fill the lines in the order
    # of the new box_lst and with that guarantee that the boxes in each line are ordered, too.
    # In case of a dominant right-to-left writing direction, we sort the boxes in reverse order.
    box_lst = sorted(
        box_lst,
        key=lambda box: box.x,
        reverse=document_prop.writing_direction == "rtl",
    )

    return document_prop, box_lst


def _order_boxes_fast(
    document_prop: DocumentProperties,
    box_lst: list[BoxProperties],
    device: str = "cpu",
) -> tuple[list[list[BoxProperties]], float]:
    """This function searches for parallel text lines, which might be tilted.
    This is the fast, vectorized version. It's not as easy to understand as the old version, but it's much faster.
    Using this faster version has no downside compared to the old version. The only reason for keeping the old version is
    that this code is not easy to understand and the intermediate step provided by the easier old version might be a big help.

    Args:
        document_prop: As generated by `_extract_properties`.
        box_lst: As generated by `_extract_properties`.
        device: "cpu" or "cuda"

    Returns:
        A tuple with two elements:
            1. A list over the found lines, where each line is given as list over boxes.
                The lines and boxes are in reading order (top->down, left->right).
            2. The best tilt_angle
    """
    # To understand this function, it might be wise to study the function `_order_boxes` (without "..._fast")
    # first, which contains more comments and is easier to read.
    nb_boxes = len(box_lst)
    if nb_boxes == 0:
        return [], 0

    max_tilt_pixel = int(np.ceil(MAX_TILT_FRACTION * document_prop.half_width))
    # We generate an array of all tilt_angles to check. At the end, we take the best performing one.
    tilt_fraction = np.linspace(-MAX_TILT_FRACTION, MAX_TILT_FRACTION, NB_TILT, dtype=float)
    #
    # The basic idea is to calculate a gain for each pixel on the y-axis (height). However, we also have to accommodate the
    # tilt. To do this in a unified fashion, it helps extend the height by twice the value max_tilt_pixel (from minus to
    # plus max_tilt_pixel). So, we use a 2D array [number of tested tilt angles, prolonged y-axis]
    y_range = document_prop.full_height + 2 * max_tilt_pixel + 2
    all_gains = np.zeros((NB_TILT, y_range), dtype=float)
    x_center = np.array([box.x_center for box in box_lst], dtype=float)
    int_height = np.array([box.int_height for box in box_lst], dtype=int)
    int_top = np.array([box.int_top for box in box_lst], dtype=int)
    width = np.array([box.width for box in box_lst], dtype=float)
    half_height = int_height.astype(float) / 2
    if document_prop.median_height is not None:
        height_factor = np.minimum(1, int_height / document_prop.median_height)
    else:
        height_factor = np.ones_like(width)
    weight = width * height_factor
    nb_y = int_height + 1
    max_height = 1 + nb_y.max()
    parabolae = np.arange(max_height, dtype=float).reshape(1, -1).repeat(nb_boxes, axis=0)
    mask = parabolae <= int_height[:, np.newaxis]
    # We'll use this mask to map the 2D parabolae onto a compact 1D tensor. In this way, we loose track which value
    # belongs to which box. But we use the same mask to all other relevant values to map them in the same way onto a compact
    # 1D tensor
    parabolae = parabolae[mask]
    weight = weight.reshape(-1, 1).repeat(max_height, axis=1)[mask]
    half_height = half_height.reshape(-1, 1).repeat(max_height, axis=1)[mask]
    parabolae = (1 - ((parabolae - half_height) / np.maximum(0.5, half_height)) ** 2) * weight
    # We need to know which Y-pixel belong to the parabolae values. We start as for the parabolae value, but add start from
    # y_top
    y_pos = np.arange(max_height, dtype=int) + int_top[:, np.newaxis]
    # For the tilt, we need the x_center value - so we do the same map for x_center
    x_factor = x_center.reshape(-1, 1).repeat(max_height, axis=1)
    # Now comes the mask
    y_pos = y_pos[mask]
    x_factor = x_factor[mask]
    y_pos_tilt = (
        max_tilt_pixel  # offset
        + y_pos  # original y
        + np.outer(tilt_fraction, x_factor).astype(int)  # tilt component
    )
    all_gains = np.zeros((NB_TILT, y_range), dtype=float)
    # Now, we use scatter_add to add the parabolae values to the correct y-pixel coordinates
    np.add.at(all_gains, (np.arange(NB_TILT)[:, np.newaxis], y_pos_tilt), parabolae)
    # Now, we are looking for the best tilt - but how is "the best" defined? First, let us recap that for each tilt angle,
    # we distributed the same box-gains - only at different positions of tilted_y. So, if we would sum over all tilted_y
    # (and hence arise the information of the different positions), we would get the same value for each tilt angle.
    # So, we take the summed squares of the gain. Why is this useful? Remember that the center of the lines are supposed
    # to be maximal gain positions. So, the gain over the y-positions jagged function with several maxima (and hence,
    # minima too). The more pronounced a maximum is the surer we can be that this is a line center. So we look for the
    # most pronounced maxima. When squared, the highest maxima have the strongest impact. That's why the squared gain is a
    # measure have well our lines fit and it indicates the best tilt.
    best_tilt = np.argmax((all_gains**2).sum(axis=1))  # That's an array position
    # Now for the best tilt angle.
    tilt_factor = tilt_fraction[best_tilt]
    best_tilt_angle = (np.arctan(tilt_factor) / DEGREE2RADIAN).item()
    tilt_factor = tilt_factor.item()
    document_prop.tilt_factor = tilt_factor
    # We need the tilted_y of each box for the best tilt
    tilted_y = np.zeros(nb_boxes)
    # Add the tilted_y to the box properties
    for b, box in enumerate(box_lst):
        shift = box.x_center * tilt_factor
        box.tilted_y = box.y + shift
        tilted_y[b] = box.tilted_y
        box.tilted_top = box.top + shift
        box.tilted_bottom = box.bottom + shift
    # So, we have the best tilt. Time to search for the best lines. We start by getting the best gain. Here, "best"
    # refers to best of all tilts - not to be confused with the maxima we search now.
    best_gains = all_gains[best_tilt, :]
    # As argued on top of this function, each line should generate exactly one maximum - so let's search maxima.
    # We look for pixel with higher values than their neighbors. Since the border pixels miss a neighbor, we exclude them.
    # Due to limited resolution, a unique maximum can be represented by two neighboring pixels. To pick just one of them,
    # we use a combination of "<=" and "<".
    line_center_bool = (best_gains[:-2] <= best_gains[1:-1]) & (best_gains[2:] < best_gains[1:-1])
    # We have a bool-tensor which is True only at the positions of maxima. Let's collect the positions
    index = np.arange(1, len(best_gains) - 1, dtype=int)
    line_center_y = index[line_center_bool]  # We only collect "True" values
    if line_center_y.shape[0] == 0:
        # For defect OCR with all boxes without extension, the gain is the same for all y and no line is picked.
        # In this case, something serious is wrong, which can't be fixed here. We just need to ensure that the show goes on.
        # We need at least one line - don't care which value
        line_center_y = np.zeros(1, dtype=int)
    # The line_center_y arose from an array index, which had to be positive. To ensure positivity, we added "max_tilt_pixel"
    # above. We remove this shift now (alternatively, we could have added it to all "tilted" properties)
    line_center_y -= max_tilt_pixel
    # Having the center position of each line, we can calculate the distance of each box to each line center, giving
    # us a 2D matrix. We use Python broadcasting to calculate that matrix
    line_distance = np.abs(tilted_y[:, np.newaxis] - line_center_y[..., np.newaxis, :])
    # Having the distance of each box to each line, we simply look for the closest line
    box_line = line_distance.argmin(axis=1)
    # Before we go on, we check for empty lines. First, we check which lines are in box_line
    used_lines = np.unique(box_line)
    if used_lines.shape[0] != line_center_y.shape[0]:
        # We have unused lines. This should be a rare case
        sorted_lines = np.sort(used_lines)
        line_index = np.arange(sorted_lines.shape[0], dtype=int)
        # Now we map old line numbers to new ones.So, where does an old line goto?
        # Each old line goes to the index position where its old value is found in sorted_lines.
        # Here, we only care for used_lines. We don't care to where an unused line in mapped
        new_line_center_y = np.zeros_like(line_index)
        new_line_center_y[line_index] = line_center_y[sorted_lines[line_index]]
        line_center_y = new_line_center_y
        # For box_line, we need a map. Here, we don't care where unused lines are mapped to
        old2new = np.arange(sorted_lines[-1] + 1, dtype=int)
        old2new[sorted_lines[line_index]] = line_index
        box_line = old2new[box_line]
    # Now, we distribute the boxes to their lines. The "box_lst" contains the boxes sorted by x-position. So, we
    # automatically get the boxes in the correct x-order when we distribute the boxes to the lines.
    line_boxes = [[] for _ in line_center_y]
    for box, line_id in zip(box_lst, box_line):
        line_boxes[line_id].append(box)
        box.line_id = line_id
    document_prop.line_center = line_center_y
    # In the "old" code, we used the "_mean_line" algorithm at this place. However, this works slightly different and it's
    # not clear if this gives an improvement. But it uses time. So here, we kicked it out.
    return line_boxes, best_tilt_angle


def get_ordered_bboxes_idxs(
    ocr_bbox_lst: list[ocr_wrapper.BBox],
    document_width: int,
    document_height: int,
    device: str = "cpu",
) -> list[int]:
    if len(ocr_bbox_lst) == 0:
        return []
    document_prop, box_lst = _extract_properties(ocr_bbox_lst, document_width, document_height)
    line_boxes, _ = _order_boxes_fast(document_prop, box_lst, device)
    idxs = [box.id for line in line_boxes for box in line]
    return idxs
