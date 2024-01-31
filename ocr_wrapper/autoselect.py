"""Implements functionality to automatically select the correct OCR engine"""

from __future__ import annotations

import os
from typing import Optional

from ocr_wrapper import AwsOCR, AzureOCR, EasyOCR, GoogleAzureOCR, GoogleOCR, OcrWrapper, PaddleOCR


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

    The environment variable OCR_PROVIDER_MAPPING can be used to override the default mapping between the OCR provider
    names and the OCR engine classes. For example, setting OCR_PROVIDER_MAPPING to "google=googleazure" will make
    autoselect_ocr_engine("google") return GoogleAzureOCR instead of GoogleOCR.

    Returns:
        The OCR engine class (default if environment variable is not set and no name is given: GoogleOCR)
    """
    # Use an override if it is set

    if name is not None:
        provider = name
    else:
        provider = os.environ.get("OCR_PROVIDER", "google").lower()
    provider_cls = _name2engine_with_override(provider)
    if provider_cls is None:
        raise InvalidOcrProviderException(f"Invalid OCR provider {provider}. Select one of {name2engine.keys()}")

    return provider_cls


def _parse_override(override: Optional[str]) -> dict[str, str]:
    """Parses the possible override string (e.g. "google=googleazure") into a dictionary"""
    if override is None or override == "":
        return {}
    res = {}
    for entry in override.split(","):
        key, value = entry.split("=")
        res[key] = value
    return res


def _name2engine_with_override(name: str) -> Optional[type[OcrWrapper]]:
    """Returns the OCR engine class for the given name, taking into account the OCR_PROVIDER_MAPPING environment
    variable
    """
    override = os.environ.get("OCR_PROVIDER_MAPPING")
    override_dict = _parse_override(override)
    name = override_dict.get(name, name)
    return name2engine.get(name, None)
