from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import shapely
import shapely.affinity
from dataclasses_json import dataclass_json
from PIL import Image, ImageColor, ImageDraw, ImageFont
from shapely.geometry import Polygon


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


@dataclass_json
@dataclass
class BBox:
    """Representation of a quadrangle bounding box on an image, and methods to access different
    formats of its coordinates.

    Coordinates are stored as x- and y-values of the 4 corners, *relative to the size of the image* (0-1).
    """

    TLx: float  # Top Left
    TLy: float
    TRx: float  # Top Right
    TRy: float
    BRx: float  # Bottom Left
    BRy: float
    BLx: float  # Bottom Right
    BLy: float
    original_width: int  # Width of the image the bounding box is on, in pixels
    original_height: int  # Height of the image the bounding box is on, in pixels

    def __hash__(self):
        return hash((self.coords, self.original_width, self.original_height))

    @property
    def coords(self):
        return (self.TLx, self.TLy, self.TRx, self.TRy, self.BRx, self.BRy, self.BLx, self.BLy)

    @coords.setter
    def coords(self, coords):
        self.TLx, self.TLy, self.TRx, self.TRy, self.BRx, self.BRy, self.BLx, self.BLy = coords

    @property
    def original_size(self):
        return (self.original_width, self.original_height)

    @original_size.setter
    def original_size(self, original_size):
        self.original_width, self.original_height = original_size

    def __post_init__(self):
        # Clip coordinates between 0 and 1
        self.coords = [min(max(c, 0), 1) for c in self.coords]

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
            if abs(a - b) < 0.001:
                return a, a + 0.001
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

    @classmethod
    def from_normalized(cls, coords: Union[list, tuple], original_size: Union[list, tuple]):
        """Creates a BBox from normalized values (0-1)"""
        if not all(-0.001 <= c <= 1.001 for c in coords):
            raise ValueError(
                f"Normalized coords must be in the range [0,1], but got {coords}. Use BBox.from_pixels() if your coords are in pixels."
            )
        if not len(coords) == 8:
            raise ValueError(f"Bounding box coordinates must be a sequence of length 8, but was {coords}")
        if not len(original_size) == 2:
            raise ValueError(f"Original size must be a sequence of length 2, but was {original_size}")
        return cls(*coords, *original_size)

    @classmethod
    def from_pixels(cls, coords: Union[list, tuple], original_size: Union[list, tuple]):
        """Creates a BBox from pixel values"""
        if not len(coords) == 8:
            raise ValueError(f"Bounding box coordinates must be a sequence of length 8, but was {coords}")
        if not len(original_size) == 2:
            raise ValueError(f"Original size must be a sequence of length 2, but was {original_size}")

        normalized_coords = [c / s for c, s in zip(coords, original_size * 4)]
        if not all(-0.02 <= c <= 1.02 for c in normalized_coords):
            raise ValueError(
                f"Pixel coords must not be larger than the original image size, but coords {coords} for original size {original_size}."
            )
        return cls(*normalized_coords, *original_size)

    @classmethod
    def from_normalized_bounds(cls, bounds: tuple[float, float, float, float], original_size: Union[list, tuple]):
        """Creates a BBox from the normalized upper left and bottom right corner"""
        x1, y1, x2, y2 = bounds
        coords = (x1, y1, x2, y1, x2, y2, x1, y2)
        return cls.from_normalized(coords, original_size)

    @classmethod
    def from_pixel_bounds(cls, bounds: tuple[int, int, int, int], original_size: Union[list, tuple]):
        """Creates a BBox from the upper left and bottom right corner in pixels"""
        x1, y1, x2, y2 = bounds
        coords = (x1, y1, x2, y1, x2, y2, x1, y2)
        return cls.from_pixels(coords, original_size)

    @lru_cache
    def get_shapely_polygon(self) -> Polygon:
        """Returns the bounding box as a normalized shapely polygon"""
        tl = (self.TLx, self.TLy)
        tr = (self.TRx, self.TRy)
        br = (self.BRx, self.BRy)
        bl = (self.BLx, self.BLy)
        return Polygon([tl, tr, br, bl])

    def to_normalized(self):
        """Returns the bounding box as a 8-tuple of normalized values (0-1)"""
        return self.coords

    def to_pixels(self):
        """Returns the bounding box as a 8-tuple of pixel values"""
        return tuple(int(c * s) for c, s in zip(self.coords, self.original_size * 4))

    def to_normalized_bounds(self) -> tuple[float, float, float, float]:
        """Returns the bounding box as a 4-tuple (x1, y1, x2, y2) of its normalized maximum extent.
        Note that this can be larger than the original bounding box if it is not axis-aligned."""
        return self.get_shapely_polygon().bounds

    def to_pixel_bounds(self):
        """Returns the bounding box as a 4-tuple (x1, y1, x2, y2) of its maximum extent in pixels.
        Note that this can be larger than the original bounding box if it is not axis-aligned."""
        normalized_bounds = self.to_normalized_bounds()
        return tuple(int(c * s) for c, s in zip(normalized_bounds, self.original_size * 2))

    def rotate(self, angle) -> "BBox":
        """Returns a new BBox that is rotated around the center of the image.
        Warning: If you do this together with rotating the image, you have to manually calculate the new original size
        Args:
            angle: The angle in degrees to rotate the bounding box. Positive values are clockwise.
        """
        poly = self.get_shapely_polygon()
        rotated_poly = shapely.affinity.rotate(poly, -angle, origin=(0.5, 0.5))
        coords = np.array(rotated_poly.exterior.coords[:-1])  # Last point is the same as the first
        return BBox.from_normalized(coords.flatten().tolist(), self.original_size)


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
):
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
    def singe2list(single):
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
    img = img.copy()  # Make a copy of the image to not modify the original
    # Set up image and draw so transparent boxes work
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")

    width, _ = img.size

    # Get font
    fontsize = int((fontsize / 1000) * width)
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

        coords = bbox.to_pixels()
        if color is not None or fill_color is not None:
            coords = [coord + random.uniform(-max_augment, max_augment) * s for coord, s in zip(coords, img.size * 4)]
            draw.polygon(coords, outline=color, fill=fill_color, width=strokewidth)

        # Draw text above box
        if text_goal_brightness is not None:
            color = get_color_with_defined_brightness(color, text_goal_brightness)
        if text != "" and text is not None:
            draw.text(
                (coords[0] + 10, coords[1] - fontsize - 3),
                text=text,
                fill=color,
                font=font,
            )
    return img
