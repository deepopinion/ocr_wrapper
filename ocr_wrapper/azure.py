from __future__ import annotations

import functools
import os
import json
import time
from io import BytesIO
from typing import Optional, Union

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
            raise ImportError(
                'Azure Read requires missing "azure-cognitiveservices-vision-computervision" package.'
            )
        return fn(*args, **kwargs)

    return wrapper_decocator


class AzureOCR(OcrWrapper):
    @requires_azure
    def __init__(
        self,
        *,
        cache_file: Optional[str] = None,
        max_size: Optional[int] = 1024,
        ocr_samples: int = 1,
        verbose: bool = False,
    ):
        super().__init__(
            cache_file=cache_file,
            max_size=max_size,
            ocr_samples=ocr_samples,
            verbose=verbose,
        )
        keyfile = "~/.config/azure/ocr_credentials.json"
        with open(os.path.expanduser(keyfile), mode="r") as f:
            ocr_credentials = json.load(f)
        self.client = ComputerVisionClient(
            endpoint=ocr_credentials["endpoint"],
            credentials=CognitiveServicesCredentials(ocr_credentials["key1"]),
        )

    @requires_azure
    def _get_ocr_response(self, img: Image.Image):
        """Gets the OCR response from the Azure. Uses cached response if a cache file has been specified and the
        document has been OCRed already"""
        # Pack image in correct format
        img_bytes = self._pil_img_to_png(img)
        img_stream = BytesIO(img_bytes)

        # Try to get cached response
        read_result = self._get_from_shelf(img)
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
            self._put_on_shelf(img, read_result)
        return read_result

    @requires_azure
    def _convert_ocr_response(self, img, response) -> list[dict[str, Union[BBox, str]]]:
        """Converts the response given by Azure Read to a list of BBox"""
        result = []
        # Iterate over all responses
        for annotation in response.analyze_result.read_results:
            for line in annotation.lines:
                for word in line.words:
                    bbox = BBox.from_pixels(word.bounding_box, original_size=img.size)
                    result.append({"bbox": bbox, "text": word.text})
        return result
