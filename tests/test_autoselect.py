import pytest
from ocr_wrapper import AwsOCR, AzureOCR, EasyOCR, GoogleOCR, PaddleOCR
from ocr_wrapper.autoselect import InvalidOcrProviderException, autoselect_ocr_engine


def test_default_ocr_engine(monkeypatch):
    # Unset the OCR_PROVIDER environment variable if set
    monkeypatch.delenv("OCR_PROVIDER", raising=False)

    # When OCR_PROVIDER is not set, should default to GoogleOCR
    assert autoselect_ocr_engine() is GoogleOCR


@pytest.mark.parametrize(
    "provider, ocr_class",
    [("google", GoogleOCR), ("azure", AzureOCR), ("aws", AwsOCR), ("easy", EasyOCR), ("paddle", PaddleOCR)],
)
def test_valid_ocr_provider(monkeypatch, provider, ocr_class):
    # Set the OCR_PROVIDER environment variable to a valid provider
    monkeypatch.setenv("OCR_PROVIDER", provider)

    # Check if the correct OCR engine is returned
    assert autoselect_ocr_engine() is ocr_class


def test_invalid_ocr_provider(monkeypatch):
    # Set the OCR_PROVIDER environment variable to an invalid provider
    monkeypatch.setenv("OCR_PROVIDER", "invalid_provider")

    # Expect InvalidOcrProviderException to be raised with an unknown provider and check the message
    with pytest.raises(InvalidOcrProviderException, match="Invalid OCR provider"):
        autoselect_ocr_engine()
