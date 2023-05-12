import os
import pickle
import pytest
from PIL import Image

import ocr_wrapper.aggregate_multiple_responses as amr

filedir = os.path.dirname(__file__)
DATA_DIR = os.path.join(filedir, "data")


def test_aggregate_ocr_samples():
    with open(os.path.join(DATA_DIR, "ocr_samples.pkl"), "rb") as f:
        ocr_samples = pickle.load(f)
    ocr_samples = amr.aggregate_ocr_samples(ocr_samples, original_width=1110, original_height=875)
    assert len(ocr_samples) == 400


@pytest.mark.parametrize("image_filename", ["ocr_samples.png"])
def test_generate_img_sample(image_filename):
    image = Image.open(os.path.join(DATA_DIR, image_filename))
    img_sample = amr.generate_img_sample(image, n=0)
    assert img_sample.size == image.size
    img_sample = amr.generate_img_sample(image, n=1, k=1)
    assert img_sample.size == (image.size[0] // 2, image.size[1] // 2)
