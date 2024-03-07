import os

import pytest
from ocr_wrapper.google_document_ocr_checkbox_detector import (
    GoogleDocumentOcrCheckboxDetector,
)
from PIL import Image

PROJECT_ID = "1059850693164"
PROCESSOR_ID = "60d8544ada1705c3"


@pytest.fixture
def checkbox_detector(monkeypatch):
    monkeypatch.setenv("GOOGLE_DOC_OCR_PROJECT_ID", PROJECT_ID)
    monkeypatch.setenv("GOOGLE_DOC_OCR_PROCESSOR_ID", PROCESSOR_ID)
    return GoogleDocumentOcrCheckboxDetector()


filedir = os.path.dirname(__file__)
DATA_DIR = os.path.join(filedir, "data")


def test_checkbox_detection(checkbox_detector):
    filename = os.path.join(DATA_DIR, "checkbox.png")
    img = Image.open(filename)
    res, confidences = checkbox_detector.detect_checkboxes(page=img)
    assert len(res) == len(confidences)

    checked = []
    unchecked = []
    for r in res:
        if r.text == "☑":
            checked.append(r)
        elif r.text == "☐":
            unchecked.append(r)
    assert len(checked) == 8
    assert len(unchecked) == 24
