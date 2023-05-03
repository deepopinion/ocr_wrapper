"""
This module contains functions for estimating the angle of rotation of a document.
"""

from __future__ import annotations

from collections import Counter

from math import atan2, degrees
import matplotlib.pyplot as plt
import numpy as np
from ocr_wrapper import BBox
from scipy.stats import gaussian_kde


def _bbox_angle(bbox: BBox) -> float:
    """Returns the rotation angle of a bbox in radians"""
    left_midpoint_x = (bbox.TLx + bbox.BLx) / 2
    left_midpoint_y = (bbox.TLy + bbox.BLy) / 2
    right_midpoint_x = (bbox.TRx + bbox.BRx) / 2
    right_midpoint_y = (bbox.TRy + bbox.BRy) / 2

    dx = right_midpoint_x - left_midpoint_x
    dy = right_midpoint_y - left_midpoint_y
    angle_rad = atan2(dy, dx)

    return angle_rad


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    """Returns the center of a bbox"""
    center_x = (bbox.TLx + bbox.TRx + bbox.BLx + bbox.BRx) / 4
    center_y = (bbox.TLy + bbox.TRy + bbox.BLy + bbox.BRy) / 4

    return center_x, center_y


def _two_bbox_angle(bbox1: BBox, bbox2: BBox) -> float:
    """Returns the angle of the line between the left midpoint of bbox1 and the right midpoint of bbox2 in radians"""
    bbox1_left_midline_x = (bbox1.TLx + bbox1.BLx) / 2
    bbox1_left_midline_y = (bbox1.TLy + bbox1.BLy) / 2
    bbox2_right_midline_x = (bbox2.TRx + bbox2.BRx) / 2
    bbox2_right_midline_y = (bbox2.TRy + bbox2.BRy) / 2

    dx = bbox2_right_midline_x - bbox1_left_midline_x
    dy = bbox2_right_midline_y - bbox1_left_midline_y
    angle_rad = atan2(dy, dx)

    return angle_rad


def _bbox_neighbor_angles(bboxes: list[BBox]) -> list[float]:
    """
    Given a list of bboxes, calculate the angle of the line between each bbox and its neighbor
    """
    angles = []

    def filter_bbox_pair(bbox1: BBox, bbox2: BBox) -> bool:
        """Filter bbox pairs that don't meet certain criteria"""
        # If they likely are not cosecutive in one line, don't use them
        bbox1_center_x, _ = _bbox_center(bbox1)
        bbox2_center_x, _ = _bbox_center(bbox2)
        if bbox1_center_x > bbox2_center_x:
            return True

        return False

    for bbox1, bbox2 in zip(bboxes, bboxes[1:]):
        if filter_bbox_pair(bbox1, bbox2):
            continue
        angle = _two_bbox_angle(bbox1, bbox2)
        angles.append(angle)

    return angles


def _get_angles(bboxes: list[BBox]) -> list[float]:
    angles = [degrees(_bbox_angle(bbox)) for bbox in bboxes]
    neighboring_bbox_angles = [degrees(a) for a in _bbox_neighbor_angles(bboxes)]

    return angles + neighboring_bbox_angles


def _kde_angle_estimation(angle_estimates, max_remove_zero_proportion=0.7, verbose=False):
    """
    Estimate the angle of rotation using a kernel density estimation.

    Args:
        angle_estimates: A list of angle estimates in degrees
        max_remove_zero_proportion: The maximum proportion of zero angles. If the proportion of zero
            angles is greater than this, the zero angles are completely removed from the KDE estimation
            Even with rotated images, there will be a lot of zero-angles, so we remove them. But if
            the document is really not rotated, removing all of them leads to a rotation being detected
            when there is none. So we only remove them if the proportion of zero angles is less than
            max_remove_zero_proportion. This will probably have to be set differently for each OCR engine.
        verbose: Whether to print the proportion of zero angles

    Returns:
        A tuple containing the peak angle, the peak value, and a figure object containing the KDE plot
    """
    # Short circuit some invalid inputs
    if len(angle_estimates) == 0:
        return 0, 0, None # no bboxes
    if len(set(angle_estimates)) == 1:
        return angle_estimates[0], 0, None # all bboxes are the same angle
    
    # Calc proportion of angles that are zero
    zero_proportion = np.sum(np.array(angle_estimates) == 0.0) / len(angle_estimates)
    if verbose:
        print(f"KDE: Proportion of zero angles: {zero_proportion:.2f}")
    if zero_proportion < max_remove_zero_proportion:
        if verbose:
            print("KDE: Removing 0.0 angles")
        angle_estimates = [angle for angle in angle_estimates if angle != 0]
    # Create a Gaussian KDE instance with our angle estimates
    kde = gaussian_kde(angle_estimates)

    # Define a range of angles for the resulting distribution
    angle_range = np.linspace(min(angle_estimates) - 5, max(angle_estimates) + 5, 1000)

    # Evaluate the KDE for each angle in the range
    kde_values = kde(angle_range)

    # Find the peak angle and its value
    peak_angle = angle_range[np.argmax(kde_values)]
    peak_value = np.max(kde_values)

    # Plot the KDE result
    fig, ax = plt.subplots()

    fig.set_size_inches(7, 1)
    ax.plot(angle_range, kde_values, label="KDE")
    ax.axvline(peak_angle, color="r", linestyle="--", label=f"Peak angle: {peak_angle:.2f}")
    # ax.legend()
    ax.set_xlabel("Angle")
    ax.set_ylabel("Density")
    ax.set_title("Kernel Density Estimation")

    return peak_angle, peak_value, fig


def _histogram_angle_estimation(
    angle_estimates: list[float],
    bin_resolution: float,
    max_remove_zero_proportion: float = 0.4,
    mean_bins: int = 4,
    verbose: bool = False,
) -> float:
    """
    Determine the dominant rotation angle given a list of angles and a bin resolution using a histogram.

    Args:
    angles (list of float): A list of rotation angles in degrees.
    bin_resolution (float): The bin resolution in degrees.
    max_remove_zero_proportion (float): The max proportion of angles that have to be in bin 0.0 until the 0.0 bin is ignored.
        For rotated documents, we often still have 0.0 as the dominant angle even though it's not the dominant angle.
        For documents that are actually not rotated, 0.0 has a very high proportion of the angles, so it makes sense to keep it
        in these cases.
    mean_bins (int): The number of bins to average over when computing the mean angle. The counts in the bins are used as weights.

    Returns:
    float: The dominant rotation angle in degrees.
    """
    # Short circuit some invalid inputs
    if len(angle_estimates) == 0:
        return 0 # no bboxes
    
    # Calc proportion of angles that are zero
    zero_proportion = np.sum(np.array(angle_estimates) == 0.0) / len(angle_estimates)
    if verbose:
        print(f"Histogram: Proportion of zero angles: {zero_proportion:.2f}")
    if zero_proportion < max_remove_zero_proportion:
        if verbose:
            print("Histogram: Removing 0.0 angles")
        angle_estimates = [angle for angle in angle_estimates if angle != 0]

    # Round the angles to the nearest bin
    rounded_angles = [round(angle / bin_resolution) * bin_resolution for angle in angle_estimates]

    # Count the occurrences of each angle
    angle_counts = Counter(rounded_angles)

    # Compute the mean angle of the mean_bins bins with the highest counts using the counts as weights
    bins = angle_counts.most_common(mean_bins)
    bin_angles = [bin[0] for bin in bins]
    bin_counts = [bin[1] for bin in bins]
    mean_angle = float(np.average(bin_angles, weights=bin_counts))

    return mean_angle
