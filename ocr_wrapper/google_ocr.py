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
            raise ImportError(
                'Google OCR requires missing "google-cloud-vision" package.'
            )
        return fn(*args, **kwargs)

    return wrapper_decocator


class GoogleOCR(OcrWrapper):
    @requires_gcloud
    def __init__(self, *, cache_file: Optional[str] = None, verbose: bool = False):
        super().__init__(cache_file=cache_file, verbose=verbose)
        if not os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
            if os.path.isfile("/credentials.json"):
                credentials_path = "/credentials.json"
            else:
                credentials_path = "~/.config/gcloud/credentials.json"
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.expanduser(
                credentials_path
            )
        self.client = vision.ImageAnnotatorClient()

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
                    response = self.client.text_detection(image=vision_img)
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
        bboxes = []
        # Iterate over all responses except the first. The first is for the whole document -> ignore
        for annotation in response.text_annotations[1:]:
            text = annotation.description
            coords = [
                item
                for vert in annotation.bounding_poly.vertices
                for item in [vert.x, vert.y]
            ]
            bbox = BBox.from_float_list(coords, text=text, in_pixels=True)
            bboxes.append(bbox)
        return bboxes
