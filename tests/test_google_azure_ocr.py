import os

import pytest
from ocr_wrapper import GoogleAzureOCR
from PIL import Image
from ocr_wrapper.google_azure_ocr import merge_idx_lists

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
    return GoogleAzureOCR(ocr_samples=1, correct_tilt=True, auto_rotate=False)


@pytest.fixture
def ocr_with_auto_rotate():
    return GoogleAzureOCR(auto_rotate=True, ocr_samples=1, correct_tilt=True)


# Fixture for unrotated bboxes
@pytest.fixture
def unrotated_bboxes(ocr):
    img = Image.open(os.path.join(DATA_DIR, "ocr_test.png"))
    return ocr.ocr(img)


def test_google_azure_ocr(ocr):
    img = Image.open(os.path.join(DATA_DIR, "ocr_test_big.png"))

    res, extra = ocr.ocr(img, return_extra=True)
    text = " ".join([str(r.text) for r in res])
    assert text == "This is a test ."
    assert len(extra["confidences"][0]) == len(res)


def test_google_azure_orc_single_sample():
    img = Image.open(os.path.join(DATA_DIR, "ocr_test_big.png"))
    ocr = GoogleAzureOCR(auto_rotate=True, ocr_samples=1)

    res, extra = ocr.ocr(img, return_extra=True)
    text = " ".join([str(r.text) for r in res])
    assert text == "This is a test ."
    assert len(extra["confidences"][0]) == len(res)


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
def test_google_azure_ocr_rotation(ocr, filename, rotation):
    img = Image.open(os.path.join(DATA_DIR, filename))
    _, extras = ocr.ocr(img, return_extra=True)
    assert extras["document_rotation"] == rotation


@pytest.mark.parametrize(
    "rotated_image_filename", ["ocr_test_90deg.png", "ocr_test_180deg.png", "ocr_test_270deg.png"]
)
def test_google_azure_ocr_auto_rotation(unrotated_bboxes, ocr_with_auto_rotate, rotated_image_filename):
    img = Image.open(os.path.join(DATA_DIR, rotated_image_filename))
    rotated_bboxes = ocr_with_auto_rotate.ocr(img)
    for unrot_bbox, rot_bbox in zip(unrotated_bboxes, rotated_bboxes):
        assert unrot_bbox.get_float_list() == pytest.approx(rot_bbox.get_float_list(), abs=0.1)


@pytest.mark.parametrize("filename", ["no_ocr.png", "no_ocr.tif"])
def test_document_without_text(ocr_with_auto_rotate, filename):
    img = Image.open(os.path.join(DATA_DIR, filename))
    res = ocr_with_auto_rotate.ocr(img)
    assert len(res) == 0


def test_date_range_correction(ocr):
    test_filename = "date_range_correction.png"
    img = Image.open(os.path.join(DATA_DIR, test_filename))
    res = ocr.ocr(img)
    texts = [r.text for r in res]
    expected_dates = ["4/01/2021", "4/01/2022"]
    for expected_date in expected_dates:
        assert expected_date in texts
    assert "-" in texts


def test_azure_date_range_split(ocr):
    test_filename = "date_range_split.png"
    img = Image.open(os.path.join(DATA_DIR, test_filename))
    res = ocr.ocr(img)
    assert len(res) == 15
    texts = [r.text for r in res]
    expected_dates = ["03/01/2016", "03/01/2017", "03/01/2018", "03/01/2019", "03/01/2020", "03/01/2021"]
    for expected_date in expected_dates:
        assert expected_date in texts
    assert "-" in texts


@pytest.mark.parametrize(
    "raw_a, raw_b, sorted_ab, expected",
    [
        ([1, 2, 3, 4, 5], [6, 7, 8], [4, 2, 5, 6, 7, 3, 1, 8], [1, 8, 2, 3, 4, 5, 6, 7]),
        ([], [], [], []),
        ([], [2, 3, 4], [4, 2, 3], [4, 2, 3]),
        ([4, 5, 6], [], [5, 6, 4], [4, 5, 6]),
        ([1], [2, 3, 4, 5], [3, 4, 1, 5, 2], [3, 4, 1, 5, 2]),
    ],
)
def test_merge_idx_lists(raw_a, raw_b, sorted_ab, expected):
    res = merge_idx_lists(raw_a, raw_b, sorted_ab)
    assert res == expected
