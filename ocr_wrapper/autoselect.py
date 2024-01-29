"""Implements functionality to automatically select the correct OCR engine"""

from __future__ import annotations

import os
from typing import Optional

from ocr_wrapper import AwsOCR, AzureOCR, EasyOCR, GoogleOCR, OcrWrapper, PaddleOCR, GoogleAzureOCR


class InvalidOcrProviderException(Exception):
    """Raised when an invalid OCR provider is selected"""

    pass


name2engine = dict[str, type[OcrWrapper]](
    google=GoogleOCR,
    azure=AzureOCR,
    googleazure=GoogleAzureOCR,  # type: ignore
    aws=AwsOCR,
    easy=EasyOCR,
    paddle=PaddleOCR,
    # For backwards compatibility
    easyocr=EasyOCR,
    paddleocr=PaddleOCR,
)


def autoselect_ocr_engine(name: Optional[str] = None) -> type[OcrWrapper]:
    """Automatically select the correct OCR engine based on the environment variable OCR_PROVIDER

    Returns:
        The OCR engine class (default if environment variable is not set: GoogleOCR)
    """
    if name is not None:
        provider = name
    else:
        provider = os.environ.get("OCR_PROVIDER", "google").lower()
    provider_cls = name2engine.get(provider)
    if provider_cls is None:
        raise InvalidOcrProviderException(f"Invalid OCR provider {provider}. Select one of {name2engine.keys()}")

    return provider_cls
