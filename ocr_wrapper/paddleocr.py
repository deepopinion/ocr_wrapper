from typing import Any, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .bbox import BBox
from .ocr_wrapper import OcrCacheDisabled, OcrWrapper


class PaddleOCR(OcrWrapper):
    def __init__(
        self,
        *,
        cache_file: Union[None, str, OcrCacheDisabled] = None,
        max_size: Optional[int] = 1024,
        add_checkboxes: bool = False,
        add_qr_barcodes: bool = False,
        min_rotation_threshold: float = 0.0,
        verbose: bool = False
    ):
        try:
            import paddleocr
        except ImportError:
            raise ImportError('PaddleOCR requires missing "paddleocr" package.')

        super().__init__(
            cache_file=cache_file,
            max_size=max_size,
            add_checkboxes=add_checkboxes,
            add_qr_barcodes=add_qr_barcodes,
            min_rotation_threshold=min_rotation_threshold,
            verbose=verbose,
        )
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

    def _convert_ocr_response(self, response) -> tuple[List[BBox], dict[str, Any]]:
        """Converts the response given by Google OCR to a list of BBox"""
        paddle_resp, resize_ratio = response
        bboxes = []
        for corners, (text, confidence) in paddle_resp:
            coords = [item * resize_ratio for vert in corners for item in vert]

            bbox = BBox.from_float_list(coords, text=text, in_pixels=True)
            bboxes.append(bbox)
        return bboxes, {}
