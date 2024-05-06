import functools
from typing import Any, List, Optional, Union

from PIL import Image

from .bbox import BBox
from .ocr_wrapper import OcrCacheDisabled, OcrWrapper

try:
    import boto3
except ImportError:
    _has_boto3 = False
else:
    _has_boto3 = True


def requires_boto(fn):
    @functools.wraps(fn)
    def wrapper_decocator(*args, **kwargs):
        if not _has_boto3:
            raise ImportError('AWS Textract requires missing "boto3" package.')
        return fn(*args, **kwargs)

    return wrapper_decocator


class AwsOCR(OcrWrapper):
    @requires_boto
    def __init__(
        self,
        *,
        cache_file: Union[None, str, OcrCacheDisabled] = None,
        max_size: Optional[int] = 1024,
        add_checkboxes: bool = False,
        verbose: bool = False
    ):
        super().__init__(cache_file=cache_file, max_size=max_size, add_checkboxes=add_checkboxes, verbose=verbose)
        self.client = boto3.client("textract", region_name="eu-central-1")

    @requires_boto
    def _get_ocr_response(self, img: Image.Image):
        """Gets the OCR response from AWS. Uses cached response if a cache file has been specified and the
        document has been OCRed already"""
        # Pack image in correct format
        img_bytes = self._pil_img_to_compressed(img)

        # Try to get cached response
        response = self._get_from_shelf(img)
        if response is None:
            # If that fails (no cache file, not yet cached, ...), get response from AWS
            response = self.client.detect_document_text(Document={"Bytes": img_bytes})
            self._put_on_shelf(img, response)
        return response

    @requires_boto
    def _convert_ocr_response(self, response) -> tuple[List[BBox], dict[str, Any]]:
        """Converts the response given by Google OCR to a list of BBox"""
        bboxes = []
        # Iterate over all responses
        for block in response["Blocks"]:
            if block["BlockType"] != "WORD":
                continue
            coords = [item for vert in block["Geometry"]["Polygon"] for item in [vert["X"], vert["Y"]]]
            bbox = BBox.from_float_list(coords, text=block["Text"], in_pixels=False)
            bboxes.append(bbox)
        return bboxes, {}
