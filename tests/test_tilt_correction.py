import os

import numpy as np
import pytest
from ocr_wrapper.tilt_correction import _closest_90_degree_distance, correct_tilt
from PIL import Image
from pytest import mark

filedir = os.path.dirname(__file__)
DATA_DIR = os.path.join(filedir, "data")


@mark.parametrize(
    "angle, expected",
    [
        (91, 1),
        (89, -1),
        (170, -10),
        (0, 0),
        (90, 0),
        (180, 0),
        (270, 0),
        (360, 0),
        (-1, -1),
        (-89, 1),
        (-91, -1),
        (-359, 1),
        (361, 1),
        (450, 0),
        (719, -1),
        (721, 1),
    ],
)
def test_closest_90_degree_distance(angle, expected):
    assert _closest_90_degree_distance(angle) == expected


@pytest.mark.parametrize("angle", np.linspace(0, 9, 15))
def test_correct_tilt(angle):
    file = os.path.join(DATA_DIR, "mixed_arabic.jpg")
    img = Image.open(file)
    rotated_img = img.rotate(angle)
    _, detected_angle = correct_tilt(rotated_img)
    assert pytest.approx(abs(detected_angle), abs=0.1) == angle
