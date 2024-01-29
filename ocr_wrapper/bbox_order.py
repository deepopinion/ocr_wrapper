"""
This module contains functions to determine the order of bounding boxes in a document.
"""

from __future__ import annotations
import ocr_wrapper
import torch
from dataclasses import dataclass
from enum import Enum
from typing import Type, Union, cast
import numpy as np

DEGREE2RADIAN = 2 * np.pi / 360
#  We allow a certain tilt angle. However, the recently added tilt correction executed before the OCR-Scan removed the need
# of a wide range scan of tilt angles. So, we only check with a the range of - 1° ... + 1°.
# The MAX_TILT_ANGLE is measured in degrees. (circumference = 360°)
MAX_TILT_ANGLE = 11
# Next we determine how many intermediate positions we test between -MAX_TILT_FRACTION to +MAX_TILT_FRACTION
# since we only scan the range of - 1° ... + 1°, we set NB_TILT to 21 (i.e. steps of 0.1°)
NB_TILT = 21
MAX_TILT_FRACTION = np.tan(MAX_TILT_ANGLE * DEGREE2RADIAN)


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
    space_width = None  # estimated width of average space character
    space_width_height_ratio = None  # estimated value of character space width relative to bounding box height
    line_center = None  # array of line_centers
    # A list of boxes with the status ignore
    ignore_box_list = []


@dataclass
class BoxProperties:
    id = None  # order of appearance in sample
    text = None
    left = None
    tilted_left = None  # left after tilt correction
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


def _extract_properties(
    ocr_bbox_lst: list[ocr_wrapper.BBox], document_width: int, document_height: int
) -> tuple[Type[DocumentProperties], list[Type[BoxProperties]]]:
    """
    Extracts properties of the entire document as well as for each individual bounding box.

    The output of the ocr_wrapper serves as input.
    """

    box_lst = []
    document_prop = DocumentProperties()
    document_prop.full_height = max(200, document_height)
    document_prop.full_width = max(200, document_width)
    document_prop.half_width = document_width / 2

    box_lst = []
    char_width_lst = []
    height_lst = []
    nb_words = 0
    for id, bbx in enumerate(ocr_bbox_lst):
        box = BoxProperties()
        box_lst.append(box)
        box.id = id
        box.text = bbx.text
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
        # What comes below is needed for statistics and ultimately to estimate space character width
        # Assuming that two words are separated by a space is mostly correct, while for numbers, we might have tiny
        # spaces or noting, when a unit follows. So, we only use letter-words for space width estimation
        box.is_word = bbx.text.isalpha() and len(bbx.text) > 1
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
    # We resort the boxes once according to the horizontal position (we know the original position thanks to box.id).
    # This one sort frees us from later sorts of the boxes on the found text lines. We simply fill the lines in the order
    # of the new box_lst and with that guarantee that the smallest x value comes first in each line
    box_lst = sorted(box_lst, key=lambda box: box.x)

    return document_prop, box_lst


def _order_boxes_fast(
    document_prop: Type[DocumentProperties],
    box_lst: list[Type[BoxProperties]],
    device: str = "cpu",
) -> tuple[list[list[Type[BoxProperties]]], float]:
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
    max_tilt_pixel = int(np.ceil(MAX_TILT_FRACTION * document_prop.half_width))
    # We generate an array of all tilt_angles to check. At the end, we take the best performing one.
    tilt_fraction = torch.linspace(-MAX_TILT_FRACTION, MAX_TILT_FRACTION, NB_TILT, dtype=float, device=device)
    #
    # The basic idea is to calculate a gain for each pixel on the y-axis (height). However, we also have to accommodate the
    # tilt. To do this in a unified fashion, it helps extend the height by twice the value max_tilt_pixel (from minus to
    # plus max_tilt_pixel). So, we use a 2D array [number of tested tilt angles, prolonged y-axis]
    y_range = document_prop.full_height + 2 * max_tilt_pixel + 2
    all_gains = torch.zeros(NB_TILT, y_range, dtype=torch.float, device=device)
    x_center = torch.tensor([box.x_center for box in box_lst], dtype=torch.float, device=device)
    int_height = torch.tensor([box.int_height for box in box_lst], dtype=torch.long, device=device)
    int_top = torch.tensor([box.int_top for box in box_lst], dtype=torch.long, device=device)
    width = torch.tensor([box.width for box in box_lst], dtype=torch.float, device=device)
    half_height = int_height.float() / 2
    if document_prop.median_height is not None:
        height_factor = torch.min(
            torch.tensor(1.0, device=device),
            int_height.float() / document_prop.median_height,
        )
    else:
        height_factor = torch.ones_like(width)
    weight = width * height_factor
    nb_y = int_height + 1
    max_height = 1 + nb_y.max()
    parabolae = torch.arange(max_height, dtype=torch.float, device=device).unsqueeze(0).expand(nb_boxes, -1)
    mask = parabolae <= int_height.view(-1, 1)
    # We'll use this mask to map the 2D parabolae onto a compact 1D tensor. In this way, we loose track which value
    # belongs to which box. But we use the same mask to all other relevant values to map them in the same way onto a compact
    # 1D tensor
    parabolae = parabolae[mask]
    weight = weight.unsqueeze(1).expand(-1, max_height)[mask]
    half_height = half_height.unsqueeze(1).expand(-1, max_height)[mask]
    parabolae = (
        1 - ((parabolae - half_height) / torch.max(torch.tensor(0.5, device=device), half_height)) ** 2
    ) * weight
    # We need to know which Y-pixel belong to the parabolae values. We start as for the parabolae value, but add start from
    # y_top
    y_pos = torch.arange(max_height, dtype=torch.long, device=device).unsqueeze(0).expand(
        nb_boxes, -1
    ) + int_top.unsqueeze(1)
    # For the tilt, we need the x_center value - so we do the same map for x_center
    x_factor = x_center.unsqueeze(1).expand(-1, max_height)
    # Now comes the mask
    y_pos = y_pos[mask]
    x_factor = x_factor[mask]
    y_pos_tilt = (
        max_tilt_pixel + y_pos.unsqueeze(0).expand(NB_TILT, -1) + torch.outer(tilt_fraction, x_factor).long()  # offset
    )
    all_gains = torch.zeros(NB_TILT, y_range, dtype=torch.float, device=device)
    # Now, we use scatter_add to add the parabolae values to the correct y-pixel coordinates
    all_gains.scatter_add_(1, y_pos_tilt, parabolae.unsqueeze(0).expand(NB_TILT, -1))
    # Now, we are looking for the best tilt - but how is "the best" defined? First, let us recap that for each tilt angle,
    # we distributed the same box-gains - only at different positions of tilted_y. So, if we would sum over all tilted_y
    # (and hence arise the information of the different positions), we would get the same value for each tilt angle.
    # So, we take the summed squares of the gain. Why is this useful? Remember that the center of the lines are supposed
    # to be maximal gain positions. So, the gain over the y-positions jagged function with several maxima (and hence,
    # minima too). The more pronounced a maximum is the surer we can be that this is a line center. So we look for the
    # most pronounced maxima. When squared, the highest maxima have the strongest impact. That's why the squared gain is a
    # measure have well our lines fit and it indicates the best tilt.
    best_tilt = (all_gains**2).sum(1).argmax()  # That's an array position
    # Now for the best tilt angle.
    tilt_factor = tilt_fraction[best_tilt]
    best_tilt_angle = (torch.arctan(tilt_factor) / DEGREE2RADIAN).item()
    tilt_factor = tilt_factor.item()
    document_prop.tilt_factor = tilt_factor
    # We need the tilted_y of each box for the best tilt
    tilted_y = torch.zeros(nb_boxes, device=device)
    # Two more things: add the tilted_y to the box properties and get the tilted_left value (needed for the correct line
    # indent)
    for b, box in enumerate(box_lst):
        shift = box.x_center * tilt_factor
        box.tilted_y = box.y + shift
        tilted_y[b] = box.tilted_y
        box.tilted_top = box.top + shift
        box.tilted_bottom = box.bottom + shift
        # So far, we calculated the tilt-effect on the y-coordinate. Now, we go for x. Here, the sign is opposite.
        # Further, we are only interested in relative effects (not absolute x values). We don't need a "box.y_center"
        box.tilted_left = box.left - tilt_factor * box.y
    # So, we have the best tilt. Time to search for the best lines. We start by getting the best gain. Here, "best"
    # refers to best of all tilts - not to be confused with the maxima we search now.
    best_gains = all_gains[best_tilt, :]
    # As argued on top of this function, each line should generate exactly one maximum - so let's search maxima.
    # We look for pixel with higher values than their neighbors. Since the border pixels miss a neighbor, we exclude them.
    # Due to limited resolution, a unique maximum can be represented by two neighboring pixels. To pick just one of them,
    # we use a combination of "<=" and "<".
    line_center_bool = (best_gains[:-2] <= best_gains[1:-1]) & (best_gains[2:] < (best_gains[1:-1]))
    # We have a bool-tensor which is True only at the positions of maxima. Let's collect the positions
    index = torch.arange(1, len(best_gains) - 1, dtype=torch.long, device=device)
    line_center_y = index[line_center_bool]  # We only collect "True" values
    if line_center_y.shape[0] == 0:
        # For defect OCR with all boxes without extension, the gain is the same for all y and no line is picked.
        # In this case, something serious is wrong, which can't be fixed here. We just need to ensure that the show goes on.
        # We need at least one line - don't care which value
        line_center_y = torch.zeros([1], dtype=torch.long, device=device)
    # The line_center_y arose from an array index, which had to be positive. To ensure positivity, we added "max_tilt_pixel"
    # above. We remove this shift now (alternatively, we could have added it to all "tilted" properties)
    line_center_y -= max_tilt_pixel
    # Having the center position of each line, we can calculate the distance of each box to each line center, giving
    # us a 2D matrix. We use Python broadcasting to calculate that matrix
    line_distance = (tilted_y.unsqueeze(-1) - line_center_y.unsqueeze(-2)).abs()
    # Having the distance of each box to each line, we simply look for the closest line
    box_line = line_distance.argmin(1)
    # Before we go on, we check for empty lines. First, we check which lines are in box_line
    used_lines = torch.unique(box_line, sorted=False)
    if used_lines.shape[0] != line_center_y.shape[0]:
        # We have unused lines. This should be a rare case
        sorted_lines = used_lines.sort()[0]
        line_index = torch.arange(sorted_lines.shape[0], dtype=torch.long, device=device)
        # Now we map old line numbers to new ones.So, where does an old line goto?
        # Each old line goes to the index position where its old value is found in sorted_lines.
        # Here, we only care for used_lines. We don't care to where an unused line in mapped
        new_line_center_y = torch.zeros_like(line_index)
        new_line_center_y[line_index] = line_center_y[sorted_lines[line_index]]
        line_center_y = new_line_center_y
        # For box_line, we need a map. Here, we don't care where unused lines are mapped to
        old2new = torch.arange(sorted_lines[-1] + 1, dtype=torch.long, device=device)
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
