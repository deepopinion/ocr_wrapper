import os
from PIL import Image
from ocr_wrapper import GoogleOCR
import pytest

filedir = os.path.dirname(__file__)
DATA_DIR = os.path.join(filedir, "data")


@pytest.fixture
def ocr():
    return GoogleOCR()


def test_google_ocr(ocr):
    img = Image.open(os.path.join(DATA_DIR, "ocr_test_big.png"))

    res, extra = ocr.ocr(img, return_extra=True)
    text = " ".join([str(r.text) for r in res])
    assert text == "This is a test ."
    assert len(extra["confidences"][0]) == len(res)


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


@pytest.mark.parametrize("filename, rotation", rotation_test_documents)
def test_google_ocr_rotation(ocr, filename, rotation):
    img = Image.open(os.path.join(DATA_DIR, filename))
    _, extras = ocr.ocr(img, return_extra=True)
    assert extras["document_rotation"] == rotation


@pytest.fixture
def ocr_with_auto_rotate():
    return GoogleOCR(auto_rotate=True)


# Fixture for unrotated bboxes
@pytest.fixture
def unrotated_bboxes(ocr):
    img = Image.open(os.path.join(DATA_DIR, "ocr_test.png"))
    return ocr.ocr(img)


def test_google_ocr_auto_rotation(unrotated_bboxes, ocr_with_auto_rotate):
    rotated_images_list = ["ocr_test_90deg.png", "ocr_test_180deg.png", "ocr_test_270deg.png"]

    for img_filename in rotated_images_list:
        img = Image.open(os.path.join(DATA_DIR, img_filename))
        rotated_bboxes = ocr_with_auto_rotate.ocr(img)
        for unrot_bbox, rot_bbox in zip(unrotated_bboxes, rotated_bboxes):
            assert unrot_bbox.get_float_list() == pytest.approx(rot_bbox.get_float_list(), abs=0.1)


def test_document_without_text(ocr_with_auto_rotate):
    filename = "no_ocr.png"
    img = Image.open(os.path.join(DATA_DIR, filename))
    res = ocr_with_auto_rotate.ocr(img)
    assert len(res) == 0
