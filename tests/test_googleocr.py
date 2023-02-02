import os
from PIL import Image
from ocr_wrapper import GoogleOCR

filedir = os.path.dirname(__file__)
DATA_DIR = os.path.join(filedir, "data")


def test_google_ocr():
    img = Image.open(os.path.join(DATA_DIR, "ocr_test_big.png"))
    ocr = GoogleOCR()

    res = ocr.ocr(img)
    text = " ".join([str(r.text) for r in res])
    assert text == "This is a test ."


def test_google_ocr_rotation():
    ocr = GoogleOCR()

    def get_rotation(filename):
        img = Image.open(os.path.join(DATA_DIR, filename))
        _, extras = ocr.ocr(img, return_extra=True)
        return extras["document_rotation"]

    # Check english text
    assert get_rotation("ocr_test.png") == 0
    assert get_rotation("ocr_test_90deg.png") == 90
    assert get_rotation("ocr_test_180deg.png") == 180
    assert get_rotation("ocr_test_270deg.png") == 270

    # Check purely arabic text
    assert get_rotation("pure_arabic.jpg") == 0
    assert get_rotation("pure_arabic_90deg.jpg") == 90
    assert get_rotation("pure_arabic_180deg.jpg") == 180
    assert get_rotation("pure_arabic_270deg.jpg") == 270

    # Check mixed english/arabic text
    assert get_rotation("mixed_arabic.jpg") == 0
    assert get_rotation("mixed_arabic_90deg.jpg") == 90
    assert get_rotation("mixed_arabic_180deg.jpg") == 180
    assert get_rotation("mixed_arabic_270deg.jpg") == 270
