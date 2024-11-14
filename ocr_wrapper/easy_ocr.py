from __future__ import annotations

import functools
from typing import Any, List, Optional, Union

import numpy as np
from PIL import Image

from .bbox import BBox
from .ocr_wrapper import OcrWrapper, OcrCacheDisabled


class EasyOCR(OcrWrapper):
    def __init__(
        self,
        *,
        cache_file: Union[None, str, OcrCacheDisabled] = None,
        languages: Union[str, list[str]],
        width_thr: float = 0.5,
        max_size: Optional[int] = 1024,
        add_checkboxes: bool = False,
        add_qr_barcodes: bool = False,
        min_rotation_threshold: float = 0.0,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Args:
            languages: A string or a list of languages to use for OCR from the list here: https://www.jaided.ai/easyocr/
            width_thr: Distance where bounding boxes are still getting merged into one
        """
        try:
            import easyocr
        except ImportError:
            raise ImportError('EasyOCR requires missing "easyocr" package.')

        super().__init__(
            cache_file=cache_file,
            max_size=max_size,
            add_checkboxes=add_checkboxes,
            add_qr_barcodes=add_qr_barcodes,
            min_rotation_threshold=min_rotation_threshold,
            verbose=verbose,
        )
        self.languages = [languages] if isinstance(languages, str) else list(languages)
        self.width_thr = width_thr

        self.client = easyocr.Reader(self.languages, **kwargs)

    def _get_ocr_response(self, img: Image.Image):
        """Gets the OCR response from EasyOCR. Uses a cached response if a cache file has been specified and the
        document has been OCRed already"""
        # Pack image in correct format
        response = self._get_from_shelf(img)  # Try to get cached response
        if response is None:  # Not cached, get response from Google
            response = self.client.readtext(np.array(img), width_ths=self.width_thr)
            self._put_on_shelf(img, response)
        return response

    def _convert_ocr_response(self, response) -> tuple[List[BBox], dict[str, Any]]:
        """Converts the response given by EasyOCR to a list of BBox"""
        bboxes, confidences = [], []
        # Iterate over all responses except the first. The first is for the whole document -> ignore
        for bbox, text, score in response:
            bbox = BBox.from_easy_ocr_output(bbox)
            bbox.text = text
            bboxes.append(bbox)
            confidences.append(score)

        extra = {"confidences": confidences}
        return bboxes, extra
