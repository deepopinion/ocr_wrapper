"""
Function for pre-processing images to prepare them for OCR.
"""

import cv2
import numpy as np
from PIL import Image


def _pillow_to_opencv(pillow_image: Image.Image):
    """Convert the Pillow image to OpenCV format (numpy array)"""
    pillow_image = pillow_image.convert("RGB")
    cv_image = np.array(pillow_image)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
    return cv_image


def _opencv_to_pillow(cv_image) -> Image.Image:
    """Convert the OpenCV image to Pillow format"""
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pillow_image = Image.fromarray(cv_image)
    return pillow_image


# Pillow to opencv and back decorator
def _pillow_to_opencv_and_back(func):
    """Decorator to convert the first argument (a Pillow image) to OpenCV format (numpy array) before
    calling the function and convert the OpenCV image back to Pillow format after
    calling the function."""

    def wrapper(*args, **kwargs):
        # Convert the first argument (a Pillow image) to OpenCV format (numpy array)
        args = list(args)
        args[0] = _pillow_to_opencv(args[0])
        args = tuple(args)

        # Call the function
        result = func(*args, **kwargs)

        # Convert the OpenCV image back to Pillow format
        result = _opencv_to_pillow(result)

        return result

    return wrapper


@_pillow_to_opencv_and_back
def denoise_image_for_ocr(image):
    # Denoise the image
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)

    return denoised
