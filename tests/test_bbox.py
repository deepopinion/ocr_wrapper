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
    x1 = draw(floats(min_value=0, max_value=1, exclude_max=True))
    y1 = draw(floats(min_value=0, max_value=1, exclude_max=True))
    x2 = draw(floats(min_value=x1, max_value=1, exclude_min=True))
    y2 = draw(floats(min_value=y1, max_value=1, exclude_min=True))

    # Make sure the generated bbox is valid
    assume(x1 < x2)
    assume(y1 < y2)

    return (x1, y1, x2, y2)


@settings(max_examples=1000)
@given(bounds=bbox())
def test_360deg_rotate_equality(bounds):
    """Test that rotating a bbox by 360 degrees is the same as not rotating it."""
    bbox = BBox.from_normalized_bounds(bounds, original_size=(1000, 1000))
    original = bbox.to_normalized()
    for _ in range(4):
        bbox = bbox.rotate(90)

    assert pytest.approx(bbox.to_normalized(), abs=1e-2) == original


color_code_regex = r"^#[0-9a-fA-F]{6}$"


@given(
    bounds1=bbox(),
    bounds2=bbox(),
    bounds3=bbox(),
    colors=lists(elements=from_regex(color_code_regex), min_size=3, max_size=3),
    fill_colors=lists(elements=from_regex(color_code_regex), min_size=3, max_size=3),
    texts=lists(elements=text(), min_size=3, max_size=3),
    fill_opacities=lists(elements=floats(min_value=0, max_value=1), min_size=3, max_size=3),
    fontsize=floats(min_value=1, max_value=100),
    maxaugment=floats(min_value=0, max_value=1),
    strokewidths=integers(min_value=1, max_value=10),
)
def test_draw_bbox(
    bounds1, bounds2, bounds3, colors, fill_colors, texts, fill_opacities, fontsize, maxaugment, strokewidths
):
    """Test that drawing bboxes works."""
    img = Image.open(os.path.join(DATA_DIR, "ocr_test.png"))
    bbox1 = BBox.from_normalized_bounds(bounds1, original_size=img.size)
    bbox2 = BBox.from_normalized_bounds(bounds2, original_size=img.size)
    bbox3 = BBox.from_normalized_bounds(bounds3, original_size=img.size)
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
