import json
import os
from unittest.mock import mock_open

import pytest
from ocr_wrapper import AzureOCR, qr_barcodes
from ocr_wrapper.azure import _determine_endpoint_and_key, _discretize_angle_to_90_deg
from PIL import Image

filedir = os.path.dirname(__file__)
DATA_DIR = os.path.join(filedir, "data")

# Define filename, rotation list
rotation_test_documents = [
    # English text
    ("ocr_test.png", 0),
    ("ocr_test_90deg.png", 90),
    ("ocr_test_180deg.png", 180),
    ("ocr_test_270deg.png", 270),
    # Purely arabic text
    ("pure_arabic.jpg", 0),
    ("pure_arabic_90deg.jpg", 90),
    ("pure_arabic_180deg.jpg", 180),
    ("pure_arabic_270deg.jpg", 270),
    # Mixed english/arabic text
    ("mixed_arabic.jpg", 0),
    ("mixed_arabic_90deg.jpg", 90),
    ("mixed_arabic_180deg.jpg", 180),
    ("mixed_arabic_270deg.jpg", 270),
]


@pytest.fixture
def ocr():
    return AzureOCR(ocr_samples=1, add_qr_barcodes=True)


@pytest.fixture
def ocr_with_auto_rotate():
    return AzureOCR(auto_rotate=True, ocr_samples=1)


# Fixture for unrotated bboxes
@pytest.fixture
def unrotated_bboxes(ocr):
    img = Image.open(os.path.join(DATA_DIR, "ocr_test.png"))
    return ocr.ocr(img)


def test_azure_ocr(ocr):
    img = Image.open(os.path.join(DATA_DIR, "ocr_test_big.png"))

    res, extra = ocr.ocr(img, return_extra=True)
    text = " ".join([str(r.text) for r in res])
    assert text == "This is a test."
    assert len(extra["confidences"][0]) == len(res)


def test_azure_ocr_single_sample():
    img = Image.open(os.path.join(DATA_DIR, "ocr_test_big.png"))
    ocr = AzureOCR(auto_rotate=True, ocr_samples=1)

    res, extra = ocr.ocr(img, return_extra=True)
    text = " ".join([str(r.text) for r in res])
    assert text == "This is a test."
    assert len(extra["confidences"][0]) == len(res)


def test_azure_qr(ocr):
    img = Image.open(os.path.join(DATA_DIR, "qr_code.png"))

    res = ocr.ocr(img, return_extra=False)

    # Assert that one of the returned bboxes is the QR code
    expected_text = "QRCODE[[http://en.m.wikipedia.org]]"
    assert any(r.text == expected_text for r in res)


@pytest.mark.parametrize("rotation_angle", [0.5, -2.2, 3.5, -9.5])
@pytest.mark.parametrize("ocr_system", ["ocr", "ocr_with_auto_rotate"])
def test_tilt_correction(rotation_angle, ocr_system, request):
    ocr = request.getfixturevalue(
        ocr_system
    )  # Get the fixture by name (has to be done this way because the fixture is parametrized)
    with Image.open(os.path.join(DATA_DIR, "ocr_samples.png")) as img:
        rot_img = img.rotate(rotation_angle, expand=True, fillcolor="white")
        _, extra = ocr.ocr(rot_img, return_extra=True)
        assert extra["tilt_angle"] == pytest.approx(rotation_angle, abs=0.01)


@pytest.mark.parametrize("filename, rotation", rotation_test_documents)
def test_azure_ocr_rotation(ocr, filename, rotation):
    """Check that the rotation angle is correctly detected"""
    img = Image.open(os.path.join(DATA_DIR, filename))
    _, extras = ocr.ocr(img, return_extra=True)
    assert extras["document_rotation"] == rotation


def test_azure_ocr_auto_rotation(unrotated_bboxes, ocr_with_auto_rotate):
    """Check that the auto rotation works correctly"""
    rotated_images_list = ["ocr_test_90deg.png", "ocr_test_180deg.png", "ocr_test_270deg.png"]

    for img_filename in rotated_images_list:
        img = Image.open(os.path.join(DATA_DIR, img_filename))
        rotated_bboxes, extras = ocr_with_auto_rotate.ocr(img, return_extra=True)
        for unrot_bbox, rot_bbox in zip(unrotated_bboxes, rotated_bboxes):
            assert unrot_bbox.get_float_list() == pytest.approx(rot_bbox.get_float_list(), abs=0.1)

        assert isinstance(extras["rotated_image"], Image.Image)


# We often have bugs when processing completely empty documents, so let's test that
@pytest.mark.parametrize("filename", ["no_ocr.png", "no_ocr.tif"])
def test_document_without_text(ocr_with_auto_rotate, filename):
    img = Image.open(os.path.join(DATA_DIR, filename))
    res = ocr_with_auto_rotate.ocr(img)
    assert len(res) == 0


# Check discretization to the nearest 90 degrees
@pytest.mark.parametrize(
    "angle, expected",
    [
        (0, 0),
        (45, 90),
        (44, 0),
        (135, 180),
        (136, 180),
        (225, 270),
        (226, 270),
        (315, 0),
        (314, 270),
        (405, 90),  # Testing beyond 360 degrees
        (-45, 0),  # Testing negative angles
        (-90, 270),  # Testing negative angles
    ],
)
def test_discretize_angle_to_90_deg(angle, expected):
    assert _discretize_angle_to_90_deg(angle) == expected


class TestDetermineEndpointAndKey:
    def test_with_provided_endpoint_and_key(self):
        endpoint, key = _determine_endpoint_and_key("https://example.com", "test_key")
        assert endpoint == "https://example.com"
        assert key == "test_key"

    def test_with_env_vars_for_endpoint_and_key(self, monkeypatch):
        monkeypatch.setenv("AZURE_OCR_ENDPOINT", "https://env.example.com")
        monkeypatch.setenv("AZURE_OCR_KEY", "env_key")
        endpoint, key = _determine_endpoint_and_key(None, None)
        assert endpoint == "https://env.example.com"
        assert key == "env_key"

    def test_with_partial_env_vars(self, monkeypatch):
        monkeypatch.setenv("AZURE_OCR_ENDPOINT", "https://partial.example.com")
        endpoint, key = _determine_endpoint_and_key(None, "partial_key")
        assert endpoint == "https://partial.example.com"
        assert key == "partial_key"

    def test_with_file_credentials(self, mocker, monkeypatch):
        # Simulate absence of environmental variables
        monkeypatch.delenv("AZURE_OCR_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OCR_KEY", raising=False)
        # Simulate json data loaded from a file
        credentials_data = '{"endpoint": "https://file.example.com", "key": "file_key"}'
        mocker.patch("os.path.exists", return_value=True)
        mocker.patch("builtins.open", mock_open(read_data=credentials_data))
        mocker.patch("os.path.expanduser", return_value="fake/path/ocr_credentials.json")
        mocker.patch("json.load", return_value=json.loads(credentials_data))

        endpoint, key = _determine_endpoint_and_key(None, None)
        assert endpoint == "https://file.example.com"
        assert key == "file_key"

    def test_raises_exception_when_file_missing(self, mocker, monkeypatch):
        monkeypatch.delenv("AZURE_OCR_ENDPOINT", raising=False)
        monkeypatch.delenv("AZURE_OCR_KEY", raising=False)
        mocker.patch("os.path.expanduser", return_value="fake/path/ocr_credentials.json")
        mocker.patch("builtins.open", side_effect=FileNotFoundError)

        # Check that assert is raised when file is missing
        with pytest.raises(Exception):
            _determine_endpoint_and_key(None, None)
