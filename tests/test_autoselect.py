import ocr_wrapper.autoselect as autoselect
import pytest
from ocr_wrapper import AwsOCR, AzureOCR, EasyOCR, GoogleAzureOCR, GoogleOCR, PaddleOCR


def test_default_ocr_engine(monkeypatch):
    # Unset the OCR_PROVIDER environment variable if set
    monkeypatch.delenv("OCR_PROVIDER", raising=False)
    monkeypatch.delenv("OCR_PROVIDER_MAPPING", raising=False)

    # When OCR_PROVIDER is not set, should default to GoogleOCR
    assert autoselect.autoselect_ocr_engine() is GoogleOCR


@pytest.mark.parametrize(
    "provider_name, ocr_class",
    [
        ("google", GoogleOCR),
        ("azure", AzureOCR),
        ("aws", AwsOCR),
        ("easy", EasyOCR),
        ("paddle", PaddleOCR),
        ("googleazure", GoogleAzureOCR),
    ],
)
def test_valid_ocr_provider_env_variable_selection(monkeypatch, provider_name, ocr_class):
    # Set the OCR_PROVIDER environment variable to a valid provider
    monkeypatch.setenv("OCR_PROVIDER", provider_name)
    monkeypatch.delenv("OCR_PROVIDER_MAPPING", raising=False)

    # Check if the correct OCR engine is returned
    assert autoselect.autoselect_ocr_engine() is ocr_class


@pytest.mark.parametrize(
    "provider_name, ocr_class",
    [
        ("google", GoogleOCR),
        ("azure", AzureOCR),
        ("aws", AwsOCR),
        ("easy", EasyOCR),
        ("paddle", PaddleOCR),
        ("googleazure", GoogleAzureOCR),
    ],
)
def test_valid_ocr_provider_argument_selection(provider_name, ocr_class, monkeypatch):
    monkeypatch.delenv("OCR_PROVIDER_MAPPING", raising=False)
    assert autoselect.autoselect_ocr_engine(name=provider_name) is ocr_class


def test_invalid_ocr_provider(monkeypatch):
    # Expect InvalidOcrProviderException to be raised with an unknown provider and check the message
    with pytest.raises(autoselect.InvalidOcrProviderException, match="Invalid OCR provider"):
        autoselect.autoselect_ocr_engine(name="invalid_provider")

    # Set the OCR_PROVIDER environment variable to an invalid provider
    monkeypatch.setenv("OCR_PROVIDER", "invalid_provider")
    # Expect InvalidOcrProviderException to be raised with an unknown provider and check the message
    with pytest.raises(autoselect.InvalidOcrProviderException, match="Invalid OCR provider"):
        autoselect.autoselect_ocr_engine()


def test_empty_selection_default_to_google(monkeypatch):
    monkeypatch.delenv("OCR_PROVIDER", raising=False)
    monkeypatch.delenv("OCR_PROVIDER_MAPPING", raising=False)
    assert autoselect.autoselect_ocr_engine() is GoogleOCR


@pytest.mark.parametrize(
    "inpt, expected_output",
    [
        ("", {}),
        ("google=googleazure", {"google": "googleazure"}),
        ("google=googleazure,aws=google", {"google": "googleazure", "aws": "google"}),
        ("google=googleazure,aws=google,easy=azure", {"google": "googleazure", "aws": "google", "easy": "azure"}),
    ],
)
def test_parse_override(inpt, expected_output):
    assert autoselect._parse_override(inpt) == expected_output


@pytest.mark.parametrize(
    "inpt, expected_output, env_setting",
    [
        ("google", GoogleOCR, ""),
        ("google", GoogleAzureOCR, "google=googleazure"),
        ("google", GoogleAzureOCR, "google=googleazure,aws=google"),
        ("easy", EasyOCR, "google=googleazure,aws=google"),
        ("azure", AzureOCR, ""),
        ("azure", AzureOCR, "google=googleazure"),
        ("azure", AzureOCR, "google=googleazure,aws=google"),
        ("azure", GoogleOCR, "google=googleazure,azure=google"),
    ],
)
def test_name2engine_with_override(inpt, expected_output, env_setting, monkeypatch):
    monkeypatch.setenv("OCR_PROVIDER_MAPPING", env_setting)
    assert autoselect._name2engine_with_override(inpt) == expected_output
