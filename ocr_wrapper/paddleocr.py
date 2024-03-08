import functools
import numpy as np
from PIL import Image
from typing import Any, Optional, Tuple, List

from .bbox import BBox
from .ocr_wrapper import OcrWrapper

try:
    import paddleocr
except ImportError:
    _has_paddle = False
else:
    _has_paddle = True


def requires_paddle(fn):
    @functools.wraps(fn)
    def wrapper_decocator(*args, **kwargs):
        if not _has_paddle:
            raise ImportError('PaddleOCR requires missing "paddleocr" package.')
        return fn(*args, **kwargs)

    return wrapper_decocator


class PaddleOCR(OcrWrapper):
    @requires_paddle
    def __init__(
        self,
        *,
        cache_file: Optional[str] = None,
        max_size: Optional[int] = 1024,
        add_checkboxes: bool = False,
        verbose: bool = False
    ):
        super().__init__(cache_file=cache_file, max_size=max_size, add_checkboxes=add_checkboxes, verbose=verbose)
        self.client = paddleocr.PaddleOCR(
            use_angle_cls=True,
            show_log=False,
        )

    @staticmethod
    def _resize(img: Image.Image) -> Tuple[Image.Image, float]:
        img = img.copy()
        if img.mode != "RGB":
            img = img.convert("RGB")
        if img.size > (1700, 2338):  # largest known good image size for PaddleOCR
            resize_ratio = max(a / b for a, b in zip(img.size, (1700, 2338)))
            new_size = (
                int(img.size[0] / resize_ratio),
                int(img.size[1] / resize_ratio),
            )
            img = img.resize(new_size)
            return img, resize_ratio
        else:
            return img, 1.0

    @requires_paddle
    def _get_ocr_response(self, img: Image.Image):
        # Try to get cached response
        response = self._get_from_shelf(img)
        if response is None:
            # If that fails (no cache file, not yet cached, ...), get response from Google OCR
            resized_img, resize_ratio = self._resize(img)
            paddle_resp = self.client.ocr(np.asarray(resized_img))
            response = (paddle_resp, resize_ratio)
            self._put_on_shelf(img, response)
        return response

    @requires_paddle
    def _convert_ocr_response(self, response) -> tuple[List[BBox], dict[str, Any]]:
        """Converts the response given by Google OCR to a list of BBox"""
        paddle_resp, resize_ratio = response
        bboxes = []
        for corners, (text, confidence) in paddle_resp:
            coords = [item * resize_ratio for vert in corners for item in vert]

            bbox = BBox.from_float_list(coords, text=text, in_pixels=True)
            bboxes.append(bbox)
        return bboxes, {}
