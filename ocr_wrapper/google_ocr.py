from __future__ import annotations

import functools
from time import sleep
import os
from typing import Optional, List

from PIL import Image
from .bbox import BBox
from .ocr_wrapper import OcrWrapper

try:
    from google.cloud import vision
except ImportError:
    _has_gcloud = False
else:
    _has_gcloud = True


def requires_gcloud(fn):
    @functools.wraps(fn)
    def wrapper_decocator(*args, **kwargs):
        if not _has_gcloud:
            raise ImportError('Google OCR requires missing "google-cloud-vision" package.')
        return fn(*args, **kwargs)

    return wrapper_decocator


# Define a list of languages which are written from right to left. This is needed to determine the rotation of the document
rtl_languages = ["ar", "arc", "dv", "fa", "ha", "he", "khw", "ks", "ku", "ps", "ur", "yi"]


def get_mean_symbol_deltas(response):
    """Given an ocr response, calculates the mean x and y deltas for the first and last symbol in all the words.

    The information in rtl_languages is used to compensate for the right to left reading direction of some languages.

    This information can be used to judge the rotation of the text
    """
    xdeltas, ydeltas = [], []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    # Get language code
                    language_code = ""
                    if len(word.property.detected_languages) > 0:
                        language_code = word.property.detected_languages[0].language_code
                    # Calculate deltas
                    first_symbol = word.symbols[0]
                    last_symbol = word.symbols[-1]
                    xdelta = last_symbol.bounding_box.vertices[1].x - first_symbol.bounding_box.vertices[0].x
                    ydelta = last_symbol.bounding_box.vertices[1].y - first_symbol.bounding_box.vertices[0].y
                    # Fix deltas for RTL languages
                    if abs(xdelta) > abs(ydelta):  # Horizontal word orientation
                        if language_code in rtl_languages:
                            xdelta = -xdelta
                    else:  # Vertical word orientation
                        if language_code in rtl_languages:
                            ydelta = -ydelta

                    xdeltas.append(xdelta)
                    ydeltas.append(ydelta)

    # Calculate the median deltas
    xmean_delta = sum(xdeltas) / len(xdeltas)
    ymean_delta = sum(ydeltas) / len(ydeltas)

    return xmean_delta, ymean_delta


def get_rotation(xmean_delta, ymean_delta):
    """Given the mean x and y deltas, calculates the rotation of the text at 0, 90, 180, or 270 degrees

    This assumes a reading direction of left to right, top to bottom
    """
    # Normalize deltas so the bigger number becomes 1 and the smaller 0, preserving the sign
    if abs(xmean_delta) > abs(ymean_delta):
        xmean_delta, ymean_delta = xmean_delta / abs(xmean_delta), 0
    else:
        xmean_delta, ymean_delta = 0, ymean_delta / abs(ymean_delta)

    # Define rotation dictionary
    rotation_dict = {(1, 0): 0, (0, 1): 90, (-1, 0): 180, (0, -1): 270, (0, 0): 0}

    return rotation_dict[(xmean_delta, ymean_delta)]


def get_word_and_language_codes(response):
    """Given an ocr response, returns a list of tuples of word and language codes"""
    word_and_language_codes = []
    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    if len(word.property.detected_languages) > 0:
                        word_and_language_codes.append(
                            (
                                "".join([s.text for s in word.symbols]),
                                word.property.detected_languages[0].language_code,
                            )
                        )
    return word_and_language_codes


def _get_words_bboxes_confidences(response):
    """Given an ocr response, returns a list of tuples of word bounding boxes and confidences"""
    words, bboxes, confidences = [], [], []

    for page in response.full_text_annotation.pages:
        for block in page.blocks:
            for paragraph in block.paragraphs:
                for word in paragraph.words:
                    word_text = "".join([symbol.text for symbol in word.symbols])
                    word_confidence = word.confidence
                    word_bounding_box = word.bounding_box.vertices

                    words.append(word_text)
                    bboxes.append(word_bounding_box)
                    confidences.append(word_confidence)

    return words, bboxes, confidences


class GoogleOCR(OcrWrapper):
    """
    A class that provides OCR functionality using Google Cloud Vision API.

    Inherits from the OcrWrapper base class and adds a client for communicating
    with the Google Cloud Vision API. Requires authentication credentials to be
    set as an environment variable or stored in a file in the default location.

    Args:
        cache_file (Optional[str]): Path to a file to use for caching OCR results.
            Defaults to None, which disables caching.
        max_size (Optional[int]): Maximum size of the image to send to the OCR, if
            larger the image will be resized. Defaults to None, which disables resizing.
        endpoint (Optional[str]): URL of the Google Cloud Vision API endpoint to use.
            Defaults to "eu-vision.googleapis.com". "us-vision.googleapis.com" is also
            available. If None, the default endpoint (global) will be used.
        auto_rotate (bool): Whether to automatically rotate the image to
            compensate for text orientation. Defaults to False. If True, the
            bounding boxes will be rotated to match a correctly rotated image. The correctly rotated
            image itself will be returned in the extra directory from result, if return_extra is True.
        verbose (bool): Whether to print debug information during OCR processing.
            Defaults to False.

    Attributes:
        client: A vision.ImageAnnotatorClient instance for communicating with the
            Google Cloud Vision API.
    """

    @requires_gcloud
    def __init__(
        self,
        *,
        cache_file: Optional[str] = None,
        max_size: Optional[int] = None,
        endpoint: Optional[str] = "eu-vision.googleapis.com",
        auto_rotate: bool = False,
        verbose: bool = False,
    ):
        super().__init__(cache_file=cache_file, max_size=max_size, auto_rotate=auto_rotate, verbose=verbose)
        # Get credentials from environment variable of the offered default locations
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            if os.path.isfile("/credentials.json"):
                credentials_path = "/credentials.json"
            else:
                credentials_path = "~/.config/gcloud/credentials.json"
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser(credentials_path)
        # Create the client with the specified endpoint
        self.client = vision.ImageAnnotatorClient(client_options={"api_endpoint": endpoint})

    @requires_gcloud
    def _get_ocr_response(self, img: Image.Image):
        """Gets the OCR response from the Google cloud. Uses cached response if a cache file has been specified and the
        document has been OCRed already"""
        # Pack image in correct format
        img_bytes = self._pil_img_to_png(img)
        vision_img = vision.Image(content=img_bytes)

        response = self._get_from_shelf(img)  # Try to get cached response
        if response is None:  # Not cached, get response from Google
            # If something goes wrong during GoogleOCR, we also try to repeat before failing. This sometimes happens when the
            # client loses connection
            nb_repeats = 2  # Try to repeat twice before failing
            while True:
                try:
                    response = self.client.document_text_detection(image=vision_img)
                    break
                except Exception:
                    if nb_repeats == 0:
                        raise
                    nb_repeats -= 1
                    sleep(1.0)
            self._put_on_shelf(img, response)
        return response

    @requires_gcloud
    def _convert_ocr_response(self, response) -> List[BBox]:
        """Converts the response given by Google OCR to a list of BBox"""
        # Iterate over all responses except the first. The first is for the whole document -> ignore
        bboxes = []
        confidences = []

        words, verts, confs = _get_words_bboxes_confidences(response)

        for text, bbox, confidence in zip(words, verts, confs):
            coords = [item for vert in bbox for item in [vert.x, vert.y]]
            bbox = BBox.from_float_list(coords, text=text, in_pixels=True)
            bboxes.append(bbox)
            confidences.append(confidence)

        self.extra.setdefault("confidences", []).append(confidences)

        # Determine the rotation of the document
        if len(bboxes) > 0:
            rotation = get_rotation(*get_mean_symbol_deltas(response))
            self.extra["document_rotation"] = rotation
        else:  # If there is no OCR text, assume rotation is 0
            self.extra["document_rotation"] = 0

        return bboxes
