import os

import ocr_wrapper
import pytest
from hypothesis import assume, given, settings
from hypothesis.strategies import (
    composite,
    floats,
    from_regex,
    integers,
    lists,
    text,
)
from ocr_wrapper import BBox
from ocr_wrapper.bbox import draw_bboxes, get_label2color_dict
from PIL import Image, ImageColor

filedir = os.path.dirname(__file__)
DATA_DIR = os.path.join(filedir, "data")


@composite
def bbox(draw):
    """Defines a strategy for generating valid BBox objects."""
    x = draw(floats(min_value=0, max_value=1))
    y = draw(floats(min_value=0, max_value=1))
    width = draw(floats(min_value=1e-10, max_value=1))
    height = draw(floats(min_value=1e-10, max_value=1))

    # Make sure the generated bbox is valid
    assume(x + width <= 1)
    assume(y + height <= 1)

    return (x, y, width, height)


@settings(max_examples=1000)
@given(xywh=bbox())
def test_360deg_rotate_equality(xywh):
    """Test that rotating a bbox by 360 degrees is the same as not rotating it."""
    bbox = BBox.from_xywh(*xywh)
    original = bbox.get_float_list()
    for _ in range(4):
        bbox = bbox.rotate(90)

    assert pytest.approx(bbox.get_float_list(), abs=1e-2) == original


color_code_regex = r"^#[0-9a-fA-F]{6}$"


@given(
    xywh1=bbox(),
    xywh2=bbox(),
    xywh3=bbox(),
    colors=lists(elements=from_regex(color_code_regex), min_size=3, max_size=3),
    fill_colors=lists(elements=from_regex(color_code_regex), min_size=3, max_size=3),
    texts=lists(elements=text(), min_size=3, max_size=3),
    fill_opacities=lists(elements=floats(min_value=0, max_value=1), min_size=3, max_size=3),
    fontsize=floats(min_value=1, max_value=100),
    maxaugment=floats(min_value=0, max_value=1),
    strokewidths=integers(min_value=1, max_value=10),
)
def test_draw_bbox(
    xywh1, xywh2, xywh3, colors, fill_colors, texts, fill_opacities, fontsize, maxaugment, strokewidths
):
    """Test that drawing bboxes works."""
    img = Image.open(os.path.join(DATA_DIR, "ocr_test.png"))
    bbox1 = BBox.from_xywh(*xywh1)
    bbox2 = BBox.from_xywh(*xywh2)
    bbox3 = BBox.from_xywh(*xywh3)
    bboxes = [bbox1, bbox2, bbox3]
    draw_bboxes(
        img,
        bboxes,
        texts=texts,
        colors=colors,
        fill_colors=fill_colors,
        fill_opacities=fill_opacities,
        fontsize=fontsize,
        max_augment=maxaugment,
        strokewidths=strokewidths,
    )


@given(lists(elements=text()))
def test_get_label2color_dict(labels):
    d = get_label2color_dict(labels)
    assert len(d) == len(set(labels))
    # Check that all colors are unique if we have <= 64 labels and we have repetitions otherwise
    if len(labels) <= 64:
        assert len(d.values()) == len(set(d.values()))
    else:
        assert len(d.values()) == 64


@given(
    color=from_regex(color_code_regex),
    goal_brightness=floats(min_value=0, max_value=1),
)
def test_get_color_with_defined_brightness(color, goal_brightness):
    """Test that we can get a color with a defined brightness."""
    color = ocr_wrapper.bbox.get_color_with_defined_brightness(color, goal_brightness)
    # Check we got a valid color
    assert ImageColor.getcolor(color, "RGB") is not None
