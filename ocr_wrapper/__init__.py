from .aws import AwsOCR
from .azure import AzureOCR
from .bbox import BBox, draw_bboxes, get_label2color_dict
from .easy_ocr import EasyOCR
from .google_ocr import GoogleOCR
from .ocr_wrapper import OcrWrapper
from .paddleocr import PaddleOCR

# Important as last import, because it depends on the other modules
from .autoselect import autoselect_ocr_engine  # isort:skip

__all__ = [
    "AwsOCR",
    "AzureOCR",
    "GoogleOCR",
    "PaddleOCR",
    "EasyOCR",
    "BBox",
    "draw_bboxes",
    "get_label2color_dict",
    "autoselect_ocr_engine",
    "OcrWrapper",
]
