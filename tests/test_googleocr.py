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
