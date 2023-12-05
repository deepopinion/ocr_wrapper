import functools
import os
import json
import time
from io import BytesIO
from typing import Any, Optional, List

from PIL import Image
from .bbox import BBox
from .ocr_wrapper import OcrWrapper

try:
    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from azure.cognitiveservices.vision.computervision.models import (
        OperationStatusCodes,
    )
    from msrest.authentication import CognitiveServicesCredentials
except ImportError:
    _has_azure = False
else:
    _has_azure = True


def requires_azure(fn):
    @functools.wraps(fn)
    def wrapper_decocator(*args, **kwargs):
        if not _has_azure:
            raise ImportError('Azure Read requires missing "azure-cognitiveservices-vision-computervision" package.')
        return fn(*args, **kwargs)

    return wrapper_decocator


def _discretize_angle_to_90_deg(rotation: float) -> int:
    """Discretize an angle to the nearest 90 degrees"""
    return int(((rotation + 45) // 90 * 90) % 360)


def _determine_endpoint_and_key(endpoint: Optional[str], key: Optional[str]) -> tuple[str, str]:
    """Determine the endpoint and key to be used.

    If endpoint and key are both None, the values are looked up in the environment variables AZURE_OCR_ENDPOINT and
    AZURE_OCR_KEY. If these are not set, the values are read from the file ~/.config/azure/ocr_credentials.json.

    If only one of endpoint and key is None, only the other one is looked up in the environment variables and the file.
    """
    if endpoint is None:
        endpoint = os.environ.get("AZURE_OCR_ENDPOINT")
    if key is None:
        key = os.environ.get("AZURE_OCR_KEY")
    if endpoint is None or key is None:
        if os.path.exists(os.path.expanduser("~/.config/azure/ocr_credentials.json")):
            with open(os.path.expanduser("~/.config/azure/ocr_credentials.json")) as f:
                data = json.load(f)
                endpoint = endpoint or data.get("endpoint")
                key = key or data.get("key")
    if endpoint is None or key is None:
        raise Exception("Azure endpoint and key must be specified via some means")

    return endpoint, key


class AzureOCR(OcrWrapper):
    @requires_azure
    def __init__(
        self,
        *,
        cache_file: Optional[str] = None,
        max_size: Optional[int] = None,
        auto_rotate: bool = False,
        correct_tilt: bool = True,
        ocr_samples: int = 1,
        endpoint: Optional[str] = None,
        key: Optional[str] = None,
        verbose: bool = False
    ):
        super().__init__(
            cache_file=cache_file,
            max_size=max_size,
            auto_rotate=auto_rotate,
            correct_tilt=correct_tilt,
            ocr_samples=ocr_samples,
            supports_multi_samples=False,
            verbose=verbose,
        )
        endpoint, key = _determine_endpoint_and_key(endpoint, key)
        self.client = ComputerVisionClient(endpoint, CognitiveServicesCredentials(key))

    @requires_azure
    def _get_ocr_response(self, img: Image.Image):
        """Gets the OCR response from the Azure. Uses cached response if a cache file has been specified and the
        document has been OCRed already"""
        # Pack image in correct format
        img_bytes = self._pil_img_to_compressed(img, compression="png")
        img_stream = BytesIO(img_bytes)

        # Try to get cached response
        read_result = self._get_from_shelf(img_bytes)
        if read_result is None:
            # If that fails (no cache file, not yet cached, ...), get response from Azure
            read_response = self.client.read_in_stream(img_stream, raw=True)
            read_operation_location = read_response.headers["Operation-Location"]
            operation_id = read_operation_location.split("/")[-1]

            while True:
                read_result = self.client.get_read_result(operation_id)
                if read_result.status not in ["notStarted", "running"]:
                    break
                time.sleep(0.1)
            if read_result.status != OperationStatusCodes.succeeded:
                raise Exception("Azure operation returned error")
            self._put_on_shelf(img_bytes, read_result)
        return read_result

    @requires_azure
    def _convert_ocr_response(self, response, *, sample_nr: int = 0) -> tuple[List[BBox], dict[str, Any]]:
        """Converts the response given by Azure Read to a list of BBox"""
        bboxes = []
        confidences = []
        extra = {}

        # Iterate over all responses
        for annotation in response.analyze_result.read_results:
            for line in annotation.lines:
                for word in line.words:
                    bbox = BBox.from_float_list(word.bounding_box, text=word.text, in_pixels=True)
                    bboxes.append(bbox)
                    confidences.append(word.confidence)

        extra["confidences"] = confidences

        # Determine rotation of document
        page_rotation = response.analyze_result.read_results[0].angle
        extra["document_rotation"] = _discretize_angle_to_90_deg(page_rotation)

        return bboxes, extra
