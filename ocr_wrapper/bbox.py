from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass
from functools import lru_cache
from random import random
from typing import Optional, TypeVar, Union
from uuid import uuid4

from opentelemetry import trace
from PIL import Image, ImageColor, ImageDraw, ImageFont
from shapely import affinity
from shapely.errors import GEOSException
from shapely.geometry import Polygon

tracer = trace.get_tracer(__name__)

T = TypeVar("T")


def get_color_with_defined_brightness(color, goal_brightness=0.5):
    """
    Darkens a color represented in hex code.

    Arguments:
    color -- a string representing the color in hex code (e.g. "#FF0000" for red) or a color name from Pillow
    goal_brightness -- a float representing the desired brightness level of the color.

    Returns:
    A string representing the darkened color in hex code.
    """
    try:
        red, green, blue = ImageColor.getcolor(color, "RGB")
    except ValueError:
        try:
            red, green, blue = ImageColor.getrgb(color)
        except ValueError:
            return color

    # Calculate the original brightness level of the color
    brightness = (red + green + blue) / 3 / 255

    # Calculate the new brightness level
    new_brightness = goal_brightness
    scale_factor = new_brightness / (brightness + 0.0001)  # Epsilon to prevent dib by zero

    # Darken the RGB values by the given factor and scale the values to achieve the desired brightness level
    red = min(255, int(red * scale_factor))
    green = min(255, int(green * scale_factor))
    blue = min(255, int(blue * scale_factor))

    # Convert the darkened RGB values back to hex code
    dark_hex = "#{:02x}{:02x}{:02x}".format(red, green, blue)

    return dark_hex


def get_label2color_dict(labels: list[str]) -> dict[str, str]:
    """
    Given a list of labels, returns a dictionary mapping labels to colors in hex format (e.g. #a3f2c3)

    Maximally divergent colors are assigned if possible and red hues are avoided since they should be reserved for
    errors etc.

    There are multiple color palletes that will be automatically chosen depending on the number of labels and can be found in
    pallet.json. The smallest pallet that can fit all labels will be chosen. If the number of labels is larger than 64,
    the largest pallet will repeat after 64 colors.

    https://medialab.github.io/iwanthue/ was used to generate the palletes.
    """
    labels = sorted(list(set(labels)))  # Remove duplicates
    # Load the color pallets from pallets.json
    with open(os.path.join(os.path.dirname(__file__), "pallets.json"), "r") as f:
        pallets = json.load(f)

    assert len(pallets) > 0, "No pallets found in pallets.json"

    # Find the smallest color pallet that has enough colors for the number of labels we have
    pallet = []  # Make static checkers happy, otherwise they can think that pallet might be unbound
    for pallet in pallets:
        if len(pallet) >= len(labels):
            break

    # Create a dictionary mapping labels to colors
    label2color = {}
    for i, label in enumerate(labels):
        label2color[label] = pallet[i % len(pallet)]

    return label2color


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

    def __hash__(self):
        return hash(
            (
                self.TLx,
                self.TLy,
                self.TRx,
                self.TRy,
                self.BRx,
                self.BRy,
                self.BLx,
                self.BLy,
                self.in_pixels,
                self.text,
                self.label,
            )
        )

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
    def from_xywh(
        upperleft_x: Union[float, int],
        upperleft_y: Union[float, int],
        width: Union[float, int],
        height: Union[float, int],
        in_pixels: bool = False,
    ) -> BBox:
        """Creates a BBox from the upper left corner, width and height"""
        return BBox(
            upperleft_x,
            upperleft_y,
            upperleft_x + width,
            upperleft_y,
            upperleft_x + width,
            upperleft_y + height,
            upperleft_x,
            upperleft_y + height,
            in_pixels=in_pixels,
        )

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

    @lru_cache(maxsize=16000)
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

        i.e. max is 1.0, min is 0.0. If one of the boxes is self-intersecting, 0.0 is returned.
        """
        self_poly = self.get_shapely_polygon()
        that_poly = that.get_shapely_polygon()
        try:
            inter_poly = self_poly.intersection(that_poly)
        except GEOSException:
            return 0.0
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

    def rotate_90deg_ccw(self) -> "BBox":
        """Rotates the bbox by 90 degrees counter clockwise around (0,0)"""
        # Only works for normalized bboxes
        if self.in_pixels:
            raise Exception("Rotation only supported for normalized bboxes")
        # Get the polygon
        poly = self.get_shapely_polygon()
        # Rotate it
        rotated_poly = affinity.rotate(poly, -90, origin=(0, 0))
        # Get the new coordinates
        x, y = rotated_poly.exterior.coords.xy
        # Translate the coordinates so the upper left corner is at (0,0) agains
        y = [e + 1 for e in y]
        # Create the new bbox
        return BBox.from_float_list(
            [x[0], y[0], x[1], y[1], x[2], y[2], x[3], y[3]],
            self.in_pixels,
            text=self.text,
            label=self.label,
        )

    def rotate(self, angle: int) -> "BBox":
        """
        Rotates the bbox by angle degrees counter clockwise around (0,0)

        Args:
            angle: angle in degrees. Only 0, 90, 180, and 270 are valid
        """
        # Applies rotate_90deg_ccw multiple times. Not very efficient, but it works, is easy to understand,
        # and is less prone to errors than a more complex implementation. Also not very time critical.
        if angle == 0:
            return self
        elif angle == 90:
            return self.rotate_90deg_ccw()
        elif angle == 180:
            return self.rotate_90deg_ccw().rotate_90deg_ccw()
        elif angle == 270:
            return self.rotate_90deg_ccw().rotate_90deg_ccw().rotate_90deg_ccw()
        else:
            raise Exception(f"Only 90, 180, and 270 are valid angles, but {angle} was given")


@tracer.start_as_current_span("draw_bboxes")
def draw_bboxes(
    img: Image.Image,
    bboxes: list[BBox],
    *,
    texts: Union[list[str], str] = "",
    colors: Union[list[str], str] = "blue",
    strokewidths: Union[list[int], int] = 3,
    fill_colors: Union[list[str], str] = "blue",
    fill_opacities: Union[list[float], float] = 0.0,
    fontsize: int = 10,
    max_augment: float = 0.0,  # Amount of to randomly change position of the bbox (for easier interpretation wiht overlapping bboxes)
    text_goal_brightness: Optional[float] = None,
) -> Image.Image:
    """Draws bounding boxes with texts, colors, etc. on a PIL image

    Args:
        img (PIL.Image.Image): The image to draw the bounding boxes on.
        bboxes (list[BBox]): The list of bounding boxes to draw on the image.
        texts (Union[list[str], str]): A list of texts to draw on top of each bounding box,
            in the same order as `bboxes`. Alternatively, a single string can be passed for
            the same text to be drawn on all bounding boxes. Defaults to an empty string.
        colors (Union[list[str], str]): A list of colors to draw the bounding boxes and texts in.
            Alternatively, a single string can be passed for the same color to be used for all bounding boxes.
            Defaults to "blue".
        strokewidths (Union[list[int], int]): A list of widths for the bounding box borders, in pixels.
            Alternatively, a single integer can be passed for the same width to be used for all bounding boxes.
            Defaults to 3.
        fill_colors (Union[list[str], str]): A list of colors to fill the bounding boxes with.
            Alternatively, a single string can be passed for the same color to be used for all bounding boxes.
            Defaults to 'blue'.
        fill_opacities (Union[list[float], float]): A list of opacities for the fill colors.
            Alternatively, a single float can be passed for the same opacity to be used for all bounding boxes.
            Value of 1.0 means fully opaque, 0.0 means fully transparent. Defaults to 0.0
        fontsize (int): The size of the font for the text drawn on top of the bounding boxes.
            Defaults to 10.
        max_augment (float): The maximum amount of random position change for the bounding boxes.
            A float between 0 and 1. This is useful when there are overlapping bounding boxes and
            one needs to be shifted a bit for better visibility. Defaults to 0.
        text_goal_brightness (float): The goal brightness for the text. If None, the text will be drawn in the original color.

    Returns:
        (PIL.Image.Image): A copy of the image with the bounding boxes and text drawn on it.
    """

    # We support both single values and lists, so we convert everything to lists if needed
    def singe2list(single: Union[T, list[T]]) -> list[T]:
        return [single] * len(bboxes) if not isinstance(single, list) else single

    texts = singe2list(texts)
    colors = singe2list(colors)
    strokewidths = singe2list(strokewidths)
    fill_colors = singe2list(fill_colors)
    fill_opacities = singe2list(fill_opacities)
    if not (len(bboxes) == len(texts) == len(colors) == len(strokewidths) == len(fill_colors) == len(fill_opacities)):
        raise Exception(
            f"Length of bboxes ({len(bboxes)}) and texts ({len(texts)}) and colors ({len(colors)}) and "
            f"strokewidths ({len(strokewidths)}) and fill_colors ({len(fill_colors)}) and "
            f"fill_opacities ({len(fill_opacities)}) must be the same"
        )
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    width, height = img.size

    # Get font
    fontsize = int((fontsize / 1000) * width)
    fontsize = max(1, fontsize)  # Prevent fontsize from being 0 which would cause an exception
    absolute_font_path = os.path.join(os.path.dirname(__file__), "NotoSans-Regular.ttf")  # Pillow needs absolute paths
    font = ImageFont.truetype(absolute_font_path, fontsize)

    def color2rgba(color, opacity):
        if color is None or opacity < 0.01:
            return None
        return ImageColor.getrgb(color) + (int(opacity * 255),)

    # Draw boxes
    for bbox, color, text, strokewidth, fill_color, fill_opacity in zip(
        bboxes, colors, texts, strokewidths, fill_colors, fill_opacities
    ):
        fill_color = color2rgba(fill_color, fill_opacity)

        if color is not None or fill_color is not None:
            bbox = bbox.to_pixels(width, height)
            bbox = bbox.get_augmented(max_augment=max_augment)
            draw.polygon(bbox.get_float_list(), outline=color, fill=fill_color, width=strokewidth)

        # Draw text above box
        if text_goal_brightness is not None:
            color = get_color_with_defined_brightness(color, text_goal_brightness)
        if text != "" and text is not None:
            draw.text(
                (bbox.TLx + 10, bbox.TLy - fontsize - 3),
                text=text,
                fill=color,
                font=font,
            )
    return img
