import os
from PIL import Image
from ocr_wrapper import EasyOCR

filedir = os.path.dirname(__file__)
DATA_DIR = os.path.join(filedir, "data")


def test_easy_ocr():
    img = Image.open(os.path.join(DATA_DIR, "ocr_test_big.png"))
    ocr = EasyOCR(languages=["en"], width_thr=0.1)

    res, extra = ocr.ocr(img, return_extra=True)
    text = " ".join([str(r.text) for r in res])
    assert "This is a test" in text
    assert "confidences" in extra
    assert len(extra["confidences"][0]) == len(res)
