import pytest

from ocr_wrapper.data_clean_utils import split_date_boxes
from ocr_wrapper import BBox


@pytest.mark.parametrize(
    "inpt, expected_texts",
    [
        (
            BBox(0, 0, 1, 0, 1, 1, 0, 1, text="01/01/2021 - 01/01/2022"),
            ["01/01/2021", "-", "01/01/2022"],
        ),
        (
            BBox(0, 0, 1, 0, 1, 1, 0, 1, text="01.01.2021 - 01.01.2022"),
            ["01.01.2021", "-", "01.01.2022"],
        ),
        (
            BBox(0, 0, 1, 0, 1, 1, 0, 1, text="01/01/2021-01/01/2022"),
            ["01/01/2021", "-", "01/01/2022"],
        ),
        (
            BBox(0, 0, 1, 0, 1, 1, 0, 1, text="01.01.2021-01.01.2022"),
            ["01.01.2021", "-", "01.01.2022"],
        ),
        (
            BBox(0, 0, 1, 0, 1, 1, 0, 1, text="01/01/2021 -01/01/2022"),
            ["01/01/2021", "-", "01/01/2022"],
        ),
        (
            BBox(0, 0, 1, 0, 1, 1, 0, 1, text="01.01.2021 -01.01.2022"),
            ["01.01.2021", "-", "01.01.2022"],
        ),
        (
            BBox(0, 0, 1, 0, 1, 1, 0, 1, text="01/01/2021- 01/01/2022"),
            ["01/01/2021", "-", "01/01/2022"],
        ),
        (
            BBox(0, 0, 1, 0, 1, 1, 0, 1, text="01.01.2021- 01.01.2022"),
            ["01.01.2021", "-", "01.01.2022"],
        ),
    ],
)
def test_split_date_boxes(inpt, expected_texts):
    results = split_date_boxes([inpt])
    for res, expected_text in zip(results, expected_texts):
        assert res.text == expected_text
