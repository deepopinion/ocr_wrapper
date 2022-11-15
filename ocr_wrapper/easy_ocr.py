from __future__ import annotations

import functools
from typing import List, Optional, Union

import numpy as np
from PIL import Image

from .bbox import BBox
from .ocr_wrapper import OcrWrapper

try:
    import easyocr
except ImportError:
    _has_easyocr = False
else:
    _has_easyocr = True


def requires_easyocr(fn):
    @functools.wraps(fn)
    def wrapper_decocator(*args, **kwargs):
        if not _has_easyocr:
            raise ImportError('Easy OCR requires missing "easyocr" package.')
        return fn(*args, **kwargs)

    return wrapper_decocator


class EasyOCR(OcrWrapper):
    @requires_easyocr
    def __init__(
        self,
        *,
        cache_file: Optional[str] = None,
        languages: Union[str, list[str]],
        width_thr: float = 0.5,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Args:
            languages: A string or a list of languages to use for OCR from the list here: https://www.jaided.ai/easyocr/
            width_thr: Distance where bounding boxes are still getting merged into one"""
        super().__init__(cache_file=cache_file, verbose=verbose)
        self.languages = [languages] if isinstance(languages, str) else list(languages)
        self.width_thr = width_thr

        self.client = easyocr.Reader(self.languages, **kwargs)

    @requires_easyocr
    def _get_ocr_response(self, img: Image.Image):
        """Gets the OCR response from EasyOCR. Uses a cached response if a cache file has been specified and the
        document has been OCRed already"""
        # Pack image in correct format
        response = self._get_from_shelf(img)  # Try to get cached response
        if response is None:  # Not cached, get response from Google
            response = self.client.readtext(np.array(img), width_ths=self.width_thr)
            self._put_on_shelf(img, response)
        return response

    @requires_easyocr
    def _convert_ocr_response(self, response) -> List[BBox]:
        """Converts the response given by EasyOCR to a list of BBox"""
        bboxes = []
        # Iterate over all responses except the first. The first is for the whole document -> ignore
        for bbox, text, score in response:
            bbox = BBox.from_easy_ocr_output(bbox)
            bbox.text = text
            bboxes.append(bbox)
        return bboxes
