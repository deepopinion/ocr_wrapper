from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Union

from PIL import Image

import ocr_wrapper
from ocr_wrapper import BBox

from .utils import resize_image

# Load the Google Cloud Document AI client library globally only for type checking (needed for argument types)
if TYPE_CHECKING:
    from google.cloud import documentai


def _val_or_env(val: Optional[str], env: str, default: Optional[str] = None) -> Optional[str]:
    """Return val if not None, else return the value of the environment variable env, if that is set, else return default."""
    return val if val is not None else os.getenv(env, default)


def _visual_element_to_bbox(visual_element) -> tuple[BBox, float]:
    """
    Convert a Document AI visual element into a bounding box (BBox) object.

    Args:
        visual_element: A Document AI visual element (checkbox, etc.).

    Returns:
        A BBox object with the bounds of the visual element and the associated text character.
    """
    style2text = {  # Mqap the style to a fitting unicode character
        "filled_checkbox": "☑",
        "unfilled_checkbox": "☐",
    }

    vertices = visual_element.layout.bounding_poly.normalized_vertices
    vertices = [value for vertex in vertices for value in (vertex.x, vertex.y)]
    confidence = visual_element.layout.confidence
    bbox = BBox.from_float_list(vertices)
    bbox.text = style2text[visual_element.type_]

    return bbox, confidence


class GoogleDocumentOcrCheckboxDetector:
    def __init__(
        self,
        project_id: Optional[str] = None,
        location: Optional[str] = None,
        processor_id: Optional[str] = None,
        processor_version: Optional[str] = None,
        max_size: Optional[int] = 4096,
    ):
        try:
            from google.api_core.client_options import ClientOptions
            from google.cloud import documentai
        except ImportError:
            raise ImportError("GoogleDocumentOcrCheckboxDetector requires missing 'google-cloud-documentai' package.")

        self.project_id = _val_or_env(project_id, "GOOGLE_DOC_OCR_PROJECT_ID")
        self.location = _val_or_env(location, "GOOGLE_DOC_OCR_LOCATION", default="eu")
        self.processor_id = _val_or_env(processor_id, "GOOGLE_DOC_OCR_PROCESSOR_ID")
        self.processor_version = _val_or_env(
            processor_version,
            "GOOGLE_DOC_OCR_PROCESSOR_VERSION",
            default="pretrained-ocr-v2.0-2023-06-02",
        )
        self.max_size = max_size

        self.process_options = documentai.ProcessOptions(
            ocr_config=documentai.OcrConfig(
                premium_features=documentai.OcrConfig.PremiumFeatures(
                    enable_selection_mark_detection=True,
                ),
            )
        )

        self.client = documentai.DocumentProcessorServiceClient(
            client_options=ClientOptions(api_endpoint=f"{self.location}-documentai.googleapis.com")
        )
        if (
            self.project_id is None
            or self.location is None
            or self.processor_id is None
            or self.processor_version is None
        ):
            raise ValueError(
                "project_id and processor_id must be set, either as arguments or as environment variables."
            )
        self.resource_name = self.client.processor_version_path(
            self.project_id, self.location, self.processor_id, self.processor_version
        )

    def detect_checkboxes(self, page: Union[Image.Image, documentai.RawDocument]) -> tuple[list[BBox], list[float]]:
        try:
            from google.cloud import documentai
        except ImportError:
            raise ImportError("GoogleDocumentOcrCheckboxDetector requires missing 'google-cloud-documentai' package.")

        if isinstance(page, Image.Image):
            if self.max_size is not None:
                page = resize_image(img=page, max_size=self.max_size)
            img_byte_arr = ocr_wrapper.OcrWrapper._pil_img_to_compressed(image=page, compression="webp")
            raw_document = documentai.RawDocument(content=img_byte_arr, mime_type="image/webp")
        elif isinstance(page, documentai.RawDocument):
            raw_document = page
        else:
            raise ValueError("page should be of type Image.Image or documentai.types.RawDocument")

        # Execute the request with exponential backoff and retry
        request = documentai.ProcessRequest(
            name=self.resource_name,
            raw_document=raw_document,
            process_options=self.process_options,
        )
        result = self.client.process_document(request=request)

        result = [
            _visual_element_to_bbox(visual_element) for visual_element in result.document.pages[0].visual_elements
        ]
        # For some reason, the system generally returns exactly the same checkbox twice, so we have to get rid of the duplicates
        result = list(set(result))

        if len(result) == 0:
            bboxes, confidences = [], []
        else:
            # Separate bboxes and confidence tuples
            bboxes, confidences = zip(*result)

        return bboxes, confidences
