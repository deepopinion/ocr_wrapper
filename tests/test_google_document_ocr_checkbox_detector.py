import os

import pytest
from ocr_wrapper.google_document_ocr_checkbox_detector import (
    GoogleDocumentOcrCheckboxDetector,
)
from PIL import Image

project_id = "1059850693164"
processor_id = "60d8544ada1705c3"


@pytest.fixture
def checkbox_detector():
    return GoogleDocumentOcrCheckboxDetector(
        project_id=project_id, processor_id=processor_id
    )


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
