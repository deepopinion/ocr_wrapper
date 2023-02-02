from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from random import random
from typing import Optional, Union
from uuid import uuid4

from PIL import Image, ImageDraw, ImageFont
from shapely.geometry import Polygon


@dataclass
class BBox:
    """A class modeling the different forms of bounding boxes"""

    TLx: Union[float, int]  # Top Left
    TLy: Union[float, int]
    TRx: Union[float, int]  # Top Right
    TRy: Union[float, int]
    BRx: Union[float, int]  # Bottom Left
    BRy: Union[float, int]
    BLx: Union[float, int]  # Bottom Right
    BLy: Union[float, int]
    in_pixels: bool = (
        False  # If True, the coordinates are absolute pixel values, otherwise they are relative in the range [0, 1]
    )
    text: Optional[str] = None
    label: Optional[str] = None

    def __post_init__(
        self,
    ):
        # Clip coordinates < 0
        (
            self.TLx,
            self.TLy,
            self.TRx,
            self.TRy,
            self.BRx,
            self.BRy,
            self.BLx,
            self.BLy,
        ) = [
            max(coord, 0.0)
            for coord in [
                self.TLx,
                self.TLy,
                self.TRx,
                self.TRy,
                self.BRx,
                self.BRy,
                self.BLx,
                self.BLy,
            ]
        ]

        # Ensure that coordinates are in the right range if the user claimed they are not pixel values
        for coord in [
            self.TLx,
            self.TLy,
            self.TRx,
            self.TRy,
            self.BRx,
            self.BRy,
            self.BLx,
            self.BLy,
        ]:
            if not self.in_pixels:
                if not 0 <= coord <= 1.01:  # Give a little bit of leeway
                    raise ValueError(
                        f"BBox claimed to not be in pixel values, but contained value {coord}, which is out of the range [0,1]"
                    )

        # Ensure that point of the bounding box are in the correct order
        tl, tr, br, rl = (
            (self.TLx, self.TLy),
            (self.TRx, self.TRy),
            (self.BRx, self.BRy),
            (self.BLx, self.BLy),
        )
        pointlist = [tl, tr, br, rl]
        # Sort list by x coord and split into left and right half
        pointlist.sort(key=lambda x: x[0])
        leftlist = pointlist[:2]
        rightlist = pointlist[2:]
        # Sort sublists by y coord
        leftlist.sort(key=lambda x: x[1])
        rightlist.sort(key=lambda x: x[1])
        # Extract points
        self.TLx, self.TLy = leftlist[0]
        self.BLx, self.BLy = leftlist[1]
        self.TRx, self.TRy = rightlist[0]
        self.BRx, self.BRy = rightlist[1]

        # Increase size of bounding box if it is collapsed
        def expand_if_necessary(a, b):
            if a != b:
                return a, b
            if self.in_pixels:
                b = a + 1
            else:
                b = a + 0.001
            return a, b

        self.TLx, self.TRx = expand_if_necessary(self.TLx, self.TRx)
        self.BLx, self.BRx = expand_if_necessary(self.BLx, self.BRx)
        self.TLy, self.BLy = expand_if_necessary(self.TLy, self.BLy)
        self.TRy, self.BRy = expand_if_necessary(self.TRy, self.BRy)

        # Sanity checks for the saved points
        assert self.TLx != self.TRx
        assert self.BLx != self.BRx
        assert self.TLy != self.BLy
        assert self.TRy != self.BRy
        assert self.TLx < self.TRx
        assert self.BLx < self.BRx
        assert self.TLy < self.BLy
        assert self.TRy < self.BRy

    @staticmethod
    def from_float_list(
        float_list: list[float],
        in_pixels: bool = False,
        text: Optional[str] = None,
        label: Optional[str] = None,
    ) -> "BBox":
        return BBox(*float_list, in_pixels=in_pixels, text=text, label=label)

    @staticmethod
    def from_easy_ocr_output(bbox_list: list[list[int]]) -> "BBox":
        tl, tr, bl, br = bbox_list
        out = [float(xy) for coord in [tl, tr, bl, br] for xy in coord]
        return BBox.from_float_list(out, in_pixels=True)

    @staticmethod
    def from_layoutlm(bbox_list: list[int], text: Optional[str] = None, label: Optional[str] = None) -> "BBox":
        x1, y1, x2, y2 = [b / 1000 for b in bbox_list]
        return BBox(x1, y1, x2, y1, x2, y2, x1, y2, in_pixels=False, text=text, label=label)

    @staticmethod
    def from_dict(dictionary):
        """Creates a BBox from a dictionary"""
        return BBox(**dictionary)

    @staticmethod
    def from_labelstudio_coords(coord_dict):
        """Given a coordinates dictionary, in the format supplied by LabelStudio, returns a BBox instance"""
        x, y, width, height = [
            value / 100
            for value in [
                coord_dict["x"],
                coord_dict["y"],
                coord_dict["width"],
                coord_dict["height"],
            ]
        ]

        # Values are sometimes minimally above 1.0 or below 0.0, so we have to clip
        def clip(val):
            return max(min(val, 1.0), 0.0)

        return BBox(
            TLx=clip(x),
            TLy=clip(y),
            TRx=clip(x + width),
            TRy=clip(y),
            BRx=clip(x + width),
            BRy=clip(y + height),
            BLx=clip(x),
            BLy=clip(y + height),
            in_pixels=False,
        )

    @staticmethod
    def from_labelstudio_format(js):
        """Given a list of dictionaries, encoding one bounding box from the LabelStudio json format

        (js[...]["annotations"][0][...]["result"]) with all the same ["id"] (specifying it is data from one bounding box),
        creates a BBox from it containing all the data which could be determined (i.e. if a "labels" type is present, the label
        will be set, if "textarea" is present, the text will be set)

        """
        # Check if the given data only captures one bounding box
        ids = set(j["id"] for j in js)
        if len(ids) > 1:
            raise Exception(
                f"The given list of dictionaries contained information for multiple bounding boxes, which is not valid. ids: {ids}"
            )

        # Make the different types of entries (rectangle, textarea, labels) easily accesible
        typedicts = {}
        for j in js:
            typedicts[j["type"]] = j

        # Check if rectangle entry is there (this should always be given) and extract bbox data from it
        if "rectangle" not in typedicts:
            raise Exception("No rectangle entry found in list of dicts. This entry is mandatory")
        bbox = BBox.from_labelstudio_coords(typedicts["rectangle"]["value"])

        # If a textarea entry is given, get the OCR text from there
        if "textarea" in typedicts:
            text = typedicts["textarea"]["value"]["text"]
            if len(text) != 1:
                raise Exception(f"Error. The text field should have one entry, but consists of {text} for id {ids}")
            bbox.text = text[0]

        # If a labels entry is given, get the label from there
        if "labels" in typedicts:
            try:
                labels = typedicts["labels"]["value"]["labels"]
            except Exception:  # It rarely happens that the labels entry is missing. Catch that and set labels to empty
                labels = []
            if len(labels) == 0:  # Sometimes there is a label entry given, but no actual label is listed
                bbox.label = None
            elif len(labels) == 1:
                bbox.label = labels[0]
            else:
                raise Exception(
                    f"Error. The labels field should have one or no entry, but consists of {labels} for id {ids}"
                )

        return bbox

    def to_dict(self):
        """Returns content of this class as a dictionary"""
        return dataclasses.asdict(self)

    def get_float_list(self) -> list[Union[float, int]]:
        """Returns coordinates from the bbox as a list"""
        return [
            self.TLx,
            self.TLy,
            self.TRx,
            self.TRy,
            self.BRx,
            self.BRy,
            self.BLx,
            self.BLy,
        ]

    def get_xpos_list(self) -> list[Union[float, int]]:
        """Returns all x positions as a list"""
        return [
            self.TLx,
            self.TRx,
            self.BRx,
            self.BLx,
        ]

    def get_ypos_list(self) -> list[Union[float, int]]:
        """Returns all y positions as a list"""
        return [
            self.TLy,
            self.TRy,
            self.BRy,
            self.BLy,
        ]

    def get_x_extrema(self) -> tuple[Union[float, int], Union[float, int]]:
        """Returns the min and max x positions"""
        xpos_list = self.get_xpos_list()
        return min(xpos_list), max(xpos_list)

    def get_y_extrema(self) -> tuple[Union[float, int], Union[float, int]]:
        """Returns the min and max y positions"""
        ypos_list = self.get_ypos_list()
        return min(ypos_list), max(ypos_list)

    def to_pixels(self, img_width: int, img_height: int) -> "BBox":
        """Changes the coordinates to pixel coordinates"""
        if self.in_pixels:
            return self
        return BBox(
            self.TLx * img_width,
            self.TLy * img_height,
            self.TRx * img_width,
            self.TRy * img_height,
            self.BRx * img_width,
            self.BRy * img_height,
            self.BLx * img_width,
            self.BLy * img_height,
            text=self.text,
            label=self.label,
            in_pixels=True,
        )

    def to_normalized(self, img_width, img_height):
        if not self.in_pixels:
            return self

        TLx, TLy, TRx, TRy, BRx, BRy, BLx, BLy = [
            min(max(val, 0.0), 1.0)
            for val in [
                self.TLx / img_width,
                self.TLy / img_height,
                self.TRx / img_width,
                self.TRy / img_height,
                self.BRx / img_width,
                self.BRy / img_height,
                self.BLx / img_width,
                self.BLy / img_height,
            ]
        ]

        return BBox(
            TLx,
            TLy,
            TRx,
            TRy,
            BRx,
            BRy,
            BLx,
            BLy,
            text=self.text,
            label=self.label,
            in_pixels=False,
        )

    def get_layoutlm_format(self) -> list[int]:
        """Returns the bbox in the format required by layoutlm (normalized to range [0,1000])
        of the upper left, and lower right corner. Possible rotation of the bbox gets lost in
        this conversion!"""
        if self.in_pixels:
            raise Exception(
                "Getting bbox in layoutlm format is only supported for normalized bboxes (i.e. not in pixels)"
            )
        # Recalculate coordinates using determined width and height
        xlist = [self.TLx, self.TRx, self.BLx, self.BRx]
        ylist = [self.TLy, self.TRy, self.BLy, self.BRy]
        x, y = min(xlist), min(ylist)
        width = max(xlist) - x
        height = max(ylist) - y
        assert x >= 0
        assert y >= 0
        assert width > 0
        assert height > 0
        return [int(coord * 1000) for coord in [x, y, x + width, y + height]]

    def get_label_studio_format(self) -> list[dict]:
        """Returns data in the format needed by LabelStudio. Depending on whether text and label are part of the BBox, more or less dicts are returned"""
        if self.in_pixels:
            raise Exception(
                "Getting bbox in LabelStudio format is only supported for normalized bboxes (i.e. not in pixels)"
            )
        # Bring bounding box in the correct format
        xlist = [x * 100 for x in [self.TLx, self.TRx, self.BLx, self.BRx]]
        ylist = [y * 100 for y in [self.TLy, self.TRy, self.BLy, self.BRy]]
        x, y = min(xlist), min(ylist)
        width = max(xlist) - x
        height = max(ylist) - y
        assert x >= 0
        assert y >= 0
        assert width > 0
        assert height > 0

        bbox = {"x": x, "y": y, "width": width, "height": height, "rotation": 0}
        region_id = str(uuid4())[:10]

        # Build dict for rectangle
        rectangle = {
            "id": region_id,
            "type": "rectangle",
            "from_name": "bbox",
            "to_name": "image",
            "value": bbox,
        }

        # Build labels
        if self.label is not None:
            labels = {
                "id": region_id,
                "type": "labels",
                "value": dict(labels=[self.label], **bbox),
                "from_name": "label",
                "to_name": "image",
            }
        else:
            labels = None

        # Build dict for textarea (if text was given)
        if self.text is not None:
            textarea = {
                "id": region_id,
                "type": "textarea",
                "from_name": "transcription",
                "to_name": "image",
                "value": dict(text=[self.text], **bbox),
                "score": 0.0,
            }
        else:
            textarea = None

        res = []

        for part in [rectangle, textarea, labels]:
            if part is not None:
                res.append(part)

        return res

    def get_shapely_polygon(self) -> Polygon:
        """Returns the bounding box as a shapely polygon"""
        tl = (self.TLx, self.TLy)
        tr = (self.TRx, self.TRy)
        br = (self.BRx, self.BRy)
        bl = (self.BLx, self.BLy)
        return Polygon([tl, tr, br, bl])

    def area(self) -> float:
        """Returns the area of the bounding box"""
        return self.get_shapely_polygon().area

    def intersection_area_percent(self, that: "BBox") -> float:
        """Returns the area of the intersection of this bbox with that bbox as a proportion of the area of this bbox.

        i.e. max is 1.0, min is 0.0
        """
        self_poly = self.get_shapely_polygon()
        that_poly = that.get_shapely_polygon()
        inter_poly = self_poly.intersection(that_poly)
        return inter_poly.area / self_poly.area

    def get_augmented(self, max_augment: float) -> "BBox":
        """Returns a randomly disturbed version of the bbox with a max disturbance of max_augment"""
        lst = self.get_float_list()
        aug_l = [e + random() * max_augment for e in lst]
        return BBox.from_float_list(
            aug_l,
            self.in_pixels,
            text=self.text,
            label=self.label,
        )

    def combine(self, that: "BBox", infix: str = " ") -> "BBox":
        """Given a second bounding box `that`, returns a new bounding box containing both boxes.

        The algorithm will always return a horizontally/vertically aligned bbox"""

        # Check if comparing the two boxes is even valid
        if self.in_pixels != that.in_pixels:
            raise Exception(
                "Trying to merge two BBox that do not have the same normalization-status. "
                f"self.in_pixels = {self.in_pixels}, that.in_pixels = {that.in_pixels}"
            )

        # Create a list of all x and y positions for all points in the bounding boxes
        xs = self.get_xpos_list() + that.get_xpos_list()
        ys = self.get_ypos_list() + that.get_ypos_list()
        # Get the extrema
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)

        # Concatenate the text contained in both boxes
        if self.text is None and that.text is None:
            text = None
        else:
            a_text = "" if self.text is None else self.text
            b_text = "" if that.text is None else that.text
            text = a_text + infix + b_text

        return BBox(
            min_x,
            min_y,
            max_x,
            min_y,
            max_x,
            max_y,
            min_x,
            max_y,
            text=text,
            in_pixels=self.in_pixels,
        )

    def get_horizontal_distance(self, that: "BBox") -> Union[float, int]:
        """Given a second bbox `that`, returns the horizontal distance between the two boxes"""
        _, self_max_x = self.get_x_extrema()
        that_min_x, _ = that.get_x_extrema()
        return that_min_x - self_max_x


def draw_bboxes(
    img: Image.Image,
    bboxes: list[BBox],
    texts: Union[list[str], str] = "",
    colors: Union[list[str], str] = "blue",
    strokewidths: Union[list[int], int] = 3,
    fontsize: int = 10,
    max_augment: float = 0.0,  # Amount of to randomly change position of the bbox (for easier interpretation wiht overlapping bboxes)
):
    """Given a PIL image and a list of bounding boxes, visualizes them in a copy of the image and returns the image

    Entries which have ``None`` as a color will not show up as a bounding box (text will still be shown though)
    """
    if not isinstance(colors, list):
        colors = [colors] * len(bboxes)
    if not isinstance(texts, list):
        texts = [texts] * len(bboxes)
    if not isinstance(strokewidths, list):
        strokewidths = [strokewidths] * len(bboxes)
    assert len(bboxes) == len(texts) == len(colors) == len(strokewidths)
    img = img.copy()
    width, height = img.size

    fontsize = int((fontsize / 1000) * width)
    font = ImageFont.truetype("NotoSans-Regular.ttf", fontsize)

    draw = ImageDraw.Draw(img)

    # Draw boxes
    for bbox, color, text, strokewidth in zip(bboxes, colors, texts, strokewidths):
        if color is not None:
            bbox = bbox.to_pixels(width, height)
            bbox = bbox.get_augmented(max_augment=max_augment)
            draw.polygon(bbox.get_float_list(), outline=color, width=strokewidth)
        # Draw text above box
        if text != "" and text is not None:
            draw.text(
                (bbox.TLx + 10, bbox.TLy - fontsize - 3),
                text=text,
                fill=color,
                font=font,
            )
    return img
