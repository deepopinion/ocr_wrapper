"""
This module contains functions to straighten an image and its bounding boxes.
"""
from __future__ import annotations

from math import radians, sin, cos

from ocr_wrapper import BBox
from PIL import Image, ImageDraw

from .angle_estimation import _get_angles, _kde_angle_estimation, _histogram_angle_estimation


def _rotate_bboxes(image: Image.Image, bboxes: list[BBox], angle: float) -> tuple[Image.Image, list[BBox]]:
    """
    Rotate bounding boxes to match a rotated image.

    Args:
        image: The original image as a PIL.Image object
        bboxes: A list of BBox objects representing the bounding boxes
        angle: The angle of rotation in degrees (clockwise)

    Returns:
        The rotated image and a list of transformed BBox objects that match the image
    """
    # Rotate image
    width, height = image.size
    new_image = image.rotate(angle, expand=True)
    new_width, new_height = new_image.size
    angle_rad = radians(-angle)  # Convert to radians and negate angle for clockwise rotation

    def rotate_point(x, y, ox, oy, angle_rad):
        """
        Rotate a point (x, y) around a pivot point (ox, oy) by a given angle in radians.
        """
        rx = cos(angle_rad) * (x - ox) - sin(angle_rad) * (y - oy) + ox
        ry = sin(angle_rad) * (x - ox) + cos(angle_rad) * (y - oy) + oy
        return rx, ry

    # Rotate bounding boxes
    new_bboxes = []
    for bbox in bboxes:
        pixel_bbox = bbox.to_pixels(width, height)  # We do all the calculations in the pixel domain

        # Calculate the center of the original and rotated images
        cx, cy = (width // 2, height // 2)
        new_cx, new_cy = (new_width // 2, new_height // 2)
        del_cx, del_cy = new_cx - cx, new_cy - cy

        # Rotate each corner of the bounding box around the center of the original image
        new_TLx, new_TLy = rotate_point(pixel_bbox.TLx, pixel_bbox.TLy, cx, cy, angle_rad)
        new_TRx, new_TRy = rotate_point(pixel_bbox.TRx, pixel_bbox.TRy, cx, cy, angle_rad)
        new_BRx, new_BRy = rotate_point(pixel_bbox.BRx, pixel_bbox.BRy, cx, cy, angle_rad)
        new_BLx, new_BLy = rotate_point(pixel_bbox.BLx, pixel_bbox.BLy, cx, cy, angle_rad)

        # Compensate for the translation of the center of the image
        new_TLx += del_cx
        new_TLy += del_cy
        new_TRx += del_cx
        new_TRy += del_cy
        new_BRx += del_cx
        new_BRy += del_cy
        new_BLx += del_cx
        new_BLy += del_cy

        new_bbox = BBox(
            new_TLx,
            new_TLy,
            new_TRx,
            new_TRy,
            new_BRx,
            new_BRy,
            new_BLx,
            new_BLy,
            in_pixels=True,
            text=pixel_bbox.text,
            label=pixel_bbox.label,
        )

        # Renormalize the bounding box if it was originally normalized
        if not bbox.in_pixels:
            new_bbox = new_bbox.to_normalized(new_width, new_height)
        new_bboxes.append(new_bbox)

    return new_image, new_bboxes


def _draw_horizontal_lines(img: Image.Image, color="red", thickness: int = 2, amount=20):
    """
    Draws horizontal lines on the Pillow image

    Args:
        img: The Pillow image
        color: The color of the lines
        thickness: The thickness of the lines
        amount: The number of lines to draw
    """
    img_copy = img.copy()
    width, height = img_copy.size
    delta = round(height / amount)
    draw = ImageDraw.Draw(img_copy)
    for i in range(1, amount):
        draw.line((0, i * delta, width, i * delta), fill=color, width=thickness)

    return img_copy


def straighten_bboxes(
    image: Image.Image,
    bboxes: list[BBox],
    method: str = "kde",
    *,
    max_remove_zero_proportion: float,
    max_rotation: float = 5.0,
    bin_resolution: float = 0.15,
    mean_bins: int = 4,
    display_horizontal_lines: bool = False,
    verbose: bool = False,
) -> tuple[Image.Image, list[BBox], float, plt.figure.Figure]:
    """
    Straighten an image and its bounding boxes.

    Args:
        image: The original image as a PIL.Image object
        bboxes: A list of BBox objects representing the bounding boxes
        method: The method to use for straightening the image. Either 'kde' or 'histogram'
        max_remove_zero_proportion:
        max_rotation: The maximum angle of rotation in degrees
        bin_resolution: The resolution of the bins used to find the dominant rotation angle
        mean_bins: The number of bins to use when calculating the mean of the dominant rotation angle
        display_horizontal_lines: Whether to display horizontal lines on the returned image for easier
            visual inspection
        verbose: Whether to print the dominant rotation angle, show the KDE plot, ...
    Returns:
        The straightened image, the straightened bounding boxes, the dominant rotation angle, and the
        KDE plot if one was generated
    """
    assert method in ["kde", "histogram"], "Method must be either 'kde' or 'histogram'"
    fig = None

    # Get the angle of rotation from the image
    angles = _get_angles(bboxes)

    # Determine the angle of rotation
    if method == "kde":
        angle, value, fig = _kde_angle_estimation(
            angles,
            max_remove_zero_proportion=max_remove_zero_proportion,
            verbose=verbose,
        )
        if verbose:
            print(f"KDE: angle: {angle:.2f}")
            print(f"KDE: value: {value:.2f}")
            if fig:
                fig.show()
    elif method == "histogram":
        angle = _histogram_angle_estimation(
            angles,
            bin_resolution=bin_resolution,
            max_remove_zero_proportion=max_remove_zero_proportion,
            mean_bins=mean_bins,
            verbose=verbose,
        )
        if verbose:
            print(f"Histogram: angle: {angle:.2f}")
    else:
        raise ValueError(f"Unknown method: {method}")

    if angle == 0:
        return image, bboxes, 0, fig
    elif abs(angle) > max_rotation:  # If the angle is too large, we don't rotate for security reasons
        return image, bboxes, 0, fig
    else:
        # Rotate the image and bounding boxes
        new_image, new_bboxes = _rotate_bboxes(image, bboxes, angle)

        if display_horizontal_lines:
            new_image = _draw_horizontal_lines(new_image)

        return new_image, new_bboxes, angle, fig
