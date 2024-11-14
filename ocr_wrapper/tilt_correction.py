"""
This module provides functionality to correct the tilt of an image of a document page.

Which implementation of the tilt correction algorithm is used can be controlled by the environment variable `OCR_WRAPPER_NO_TORCH`.
"""

import os
import warnings

from opentelemetry import trace
from opentelemetry.metrics import get_meter
from PIL import Image

tracer = trace.get_tracer(__name__)
meter = get_meter(__name__)
tilt_histogram = meter.create_histogram(
    name="tilt_histogram", unit="degrees", description="Tilt angle of document images"
)

if os.getenv("OCR_WRAPPER_NO_TORCH"):
    USE_TORCH = False
    from .tilt_correction_numpy import DetectTilt
else:
    USE_TORCH = True
    from .tilt_correction_torch import DetectTilt


def _closest_90_degree_distance(angle: float) -> float:
    """
    Returns the smallest distance to the nearest multiple of 90 degrees.
    The distance is negative if the angle is below the nearest multiple of 90,
    and positive if it is above.
    """
    nearest_multiple_of_90 = round(angle / 90) * 90
    distance = angle - nearest_multiple_of_90
    return distance


@tracer.start_as_current_span("correct_tilt")
def correct_tilt(
    image: Image.Image, tilt_threshold: float = 10, min_rotation_threshold: float = 0.0
) -> tuple[Image.Image, float]:
    """
    Corrects the tilt (small rotations) of an image of a document page

    Args:
        image: Image to correct the tilt of
        tilt_threshold: The maximum tilt angle to correct. If the angle is larger than this, the image is not rotated at all.
        min_rotation_threshold: The minimum rotation angle to correct. If abs(angle) is smaller than this, the image is not rotated at all.

    Returns:
        The rotated image and the angle of rotation
    """
    span = trace.get_current_span()
    span.set_attribute("use_torch", USE_TORCH)

    detect_tilt = DetectTilt()
    try:
        angle = detect_tilt.find_angle(image)
    except Exception as e:
        warnings.warn(f"Error while detecting tilt: {e}")
        angle = 0.0

    angle = _closest_90_degree_distance(angle)  # We round to the nearest multiple of 90 degrees
    span.set_attribute("tilt", angle)
    tilt_histogram.record(angle)

    # We only rotate if the angle is small enough to prevent bugs introduced by the algorithm
    angle = angle if abs(angle) < tilt_threshold else 0.0
    with tracer.start_as_current_span("correct_tilt: rotate_image"):
        if abs(angle) < min_rotation_threshold:
            rotated_image = image.rotate(-angle, expand=True, fillcolor="white")
        else:
            rotated_image = image

    return rotated_image, angle
