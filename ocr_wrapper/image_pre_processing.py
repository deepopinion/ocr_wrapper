"""
Function for pre-processing images to prepare them for OCR.
"""

import cv2
import numpy as np
from PIL import Image


def _pillow_to_opencv(pillow_image: Image.Image):
    """Convert the Pillow image to OpenCV format (numpy array)"""
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
    # Denoise the image using fastNlMeansDenoisingColored with the following settings:
    # See https://docs.opencv.org/4.7.0/d1/d79/group__photo__denoise.html#ga03aa4189fc3e31dafd638d90de335617
    # h : Parameter regulating filter strength for luminance component.
    #     Bigger h value perfectly removes noise but also removes image details, smaller h value preserves details but also preserves some noise
    # hForColorComponents : The same as h but for color components.
    #     For most images value equals 10 will be enough to remove colored noise and do not distort colors
    # templateWindowSize : Size in pixels of the template patch that is used to compute weights. Should be odd. Recommended value 7 pixels
    # searchWindowSize : Size in pixels of the window that is used to compute weighted average for given pixel.
    #     Should be odd. Affect performance linearly: greater searchWindowsSize - greater denoising time. Recommended value 21 pixels
    denoised = cv2.fastNlMeansDenoisingColored(image, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21)

    return denoised
