import os
from PIL import Image
from ocr_wrapper import EasyOCR

filedir = os.path.dirname(__file__)
DATA_DIR = os.path.join(filedir, "data")


def test_easy_ocr():
    img = Image.open(os.path.join(DATA_DIR, "ocr_test.png"))
    ocr = EasyOCR(languages=["en"], width_thr=0.1)

    res = ocr.ocr(img)
    text = " ".join([r["text"] for r in res])
    assert text == "This is a test."
    assert all([r["bbox"].original_size == img.size for r in res])
