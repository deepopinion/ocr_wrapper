from .aws import AwsOCR
from .azure import AzureOCR
from .google_ocr import GoogleOCR
from .paddleocr import PaddleOCR
from .easy_ocr import EasyOCR
from .bbox import BBox, draw_bboxes, get_label2color_dict
from .ocr_wrapper import OcrWrapper
from .rotation_compensation import straighten_bboxes

__all__ = [
    "AwsOCR",
    "AzureOCR",
    "GoogleOCR",
    "PaddleOCR",
    "EasyOCR",
    "BBox",
    "draw_bboxes",
    "get_label2color_dict",
    "OcrWrapper",
    "straighten_bboxes",
]
