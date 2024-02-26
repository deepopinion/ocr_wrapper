import functools
import json
import os
import random
import time
from io import BytesIO
from typing import List, Any, Optional

from PIL import Image

from .bbox import BBox
from .ocr_wrapper import OcrWrapper

try:
    from msrest.authentication import CognitiveServicesCredentials

    from azure.cognitiveservices.vision.computervision import ComputerVisionClient
    from azure.cognitiveservices.vision.computervision.models import (
        ComputerVisionOcrErrorException,
        OperationStatusCodes,
    )
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
        verbose: bool = False,
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
        # Try to get cached response
        read_result = self._get_from_shelf(img)

        if read_result is None:
            # If that fails (no cache file, not yet cached, ...), get response from Azure
            img_bytes = self._pil_img_to_compressed(img, compression="png")

            start = time.time()
            retries = 5
            delay = 0.5
            while retries > 0:
                try:
                    img_stream = BytesIO(img_bytes)
                    read_response = self.client.read_in_stream(img_stream, raw=True)
                    read_operation_location = read_response.headers["Operation-Location"]
                    operation_id = read_operation_location.split("/")[-1]
                except ComputerVisionOcrErrorException as e:  # Usually raised because of too many requests
                    # Retry with jitter and exponential backoff
                    jitter_delay = delay * (1 + 0.1 * (1 - 2 * random.random()))
                    if True:
                        print("Azure OCR failed with error", e)
                        print(f"Retrying... {retries} retries left.")
                        print(f"Jitter delay: {jitter_delay}")
                    time.sleep(jitter_delay)
                    delay *= 2
                    retries -= 1
                    if retries == 0:
                        raise
                else:
                    break

            while True:
                read_result = self.client.get_read_result(operation_id)
                if read_result.status not in ["notStarted", "running"]:
                    break
                time.sleep(0.1)
            if read_result.status != OperationStatusCodes.succeeded:
                raise Exception("Azure operation returned error")
            end = time.time()
            if self.verbose:
                print("Azure OCR took ", end - start, "seconds")
            self._put_on_shelf(img, read_result)
        return read_result

    @requires_azure
    def _convert_ocr_response(self, response) -> tuple[List[BBox], dict[str, Any]]:
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
