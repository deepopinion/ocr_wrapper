import os
from PIL import Image
from ocr_wrapper import GoogleOCR
import pytest

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


@pytest.fixture(scope="session")
def ocr():
    return GoogleOCR(ocr_samples=2)


@pytest.fixture(scope="session")
def ocr_with_auto_rotate():
    return GoogleOCR(auto_rotate=True, ocr_samples=2)


@pytest.fixture()
def ocr_forced_single_response(ocr):
    ocr.supports_multi_samples = False
    yield ocr
    ocr.supports_multi_samples = True


# Fixture for unrotated bboxes
@pytest.fixture
def unrotated_bboxes(ocr):
    img = Image.open(os.path.join(DATA_DIR, "ocr_test.png"))
    return ocr.ocr(img)


def test_google_ocr(ocr):
    img = Image.open(os.path.join(DATA_DIR, "ocr_test_big.png"))
    res, extra = ocr.ocr(img, return_extra=True)
    text = " ".join([r["text"] for r in res])
    assert text == "This is a test ."
    assert all([r["bbox"].original_size == extra["rotated_image"].size for r in res])


def test_google_ocr_forced_single_response(ocr_forced_single_response, mocker):
    single_response_spy = mocker.spy(ocr_forced_single_response, "_get_ocr_response")
    multi_response_spy = mocker.spy(ocr_forced_single_response, "_get_multi_response")

    img_path = os.path.join(DATA_DIR, "ocr_test_big.png")
    with Image.open(img_path) as img:
        res, extra = ocr_forced_single_response.ocr(img, return_extra=True)
        text = " ".join([r["text"] for r in res])
        assert text == "This is a test ."
        assert all([r["bbox"].original_size == extra["rotated_image"].size for r in res])

        multi_response_spy.assert_not_called()
        single_response_spy.assert_called_once()


def test_google_orc_single_sample():
    img = Image.open(os.path.join(DATA_DIR, "ocr_test_big.png"))
    ocr = GoogleOCR(auto_rotate=True, ocr_samples=1)
    res, extra = ocr.ocr(img, return_extra=True)
    text = " ".join([r["text"] for r in res])
    assert text == "This is a test ."
    assert all([r["bbox"].original_size == extra["rotated_image"].size for r in res])


@pytest.mark.parametrize("rotation_angle", [0.5, -2.2, 3.5])
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
def test_google_ocr_rotation(ocr, filename, rotation):
    img = Image.open(os.path.join(DATA_DIR, filename))
    _, extras = ocr.ocr(img, return_extra=True)
    assert extras["document_rotation"] == rotation


def test_google_ocr_auto_rotation(unrotated_bboxes, ocr_with_auto_rotate):
    rotated_images_list = ["ocr_test_90deg.png", "ocr_test_180deg.png", "ocr_test_270deg.png"]

    for img_filename in rotated_images_list:
        img = Image.open(os.path.join(DATA_DIR, img_filename))
        rotated_bboxes = ocr_with_auto_rotate.ocr(img)
        for unrot, rot in zip(unrotated_bboxes, rotated_bboxes):
            assert unrot["bbox"].to_normalized() == pytest.approx(rot["bbox"].to_normalized(), abs=0.1)


@pytest.mark.parametrize("filename", ["no_ocr.png", "no_ocr.tif"])
def test_document_without_text(ocr_with_auto_rotate, filename):
    img = Image.open(os.path.join(DATA_DIR, filename))
    res = ocr_with_auto_rotate.ocr(img)
    assert len(res) == 0
