import pytest
from hypothesis import given, assume, settings
from hypothesis.strategies import floats, composite
from ocr_wrapper import BBox


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
