from pytest import mark
from ocr_wrapper.tilt_correction import _closest_90_degree_distance


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
