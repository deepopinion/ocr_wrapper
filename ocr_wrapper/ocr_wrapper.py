from __future__ import annotations
from abc import ABC, abstractmethod
from hashlib import sha256

import shelve
from io import BytesIO
from typing import Any, Optional, Union

from PIL import Image, ImageDraw, ImageOps
from .bbox import BBox


def rotate_image(image: Image.Image, angle: int) -> Image.Image:
    """
    Rotate an image by a given angle.
    Only 0, 90, 180, and 270 degrees are supported. Other angles will raise an Exception.
    """
    # The rotation angles in Pillow are counter-clockwise and ours are clockwise.
    if angle == 0:
        return image
    if angle == 90:
        return image.transpose(Image.Transpose.ROTATE_90)
    elif angle == 180:
        return image.transpose(Image.Transpose.ROTATE_180)
    elif angle == 270:
        return image.transpose(Image.Transpose.ROTATE_270)
    else:
        raise Exception(f"Unsupported angle: {angle}")


class OcrWrapper(ABC):
    """Base class for OCR engines. Subclasses must implement ``_get_ocr_response``
    and ``_conver_ocr_response``."""

    def __init__(
        self,
        *,
        cache_file: Optional[str] = None,
        max_size: Optional[int] = 1024,
        auto_rotate: bool = False,
        verbose: bool = False,
    ):
        self.cache_file = cache_file
        self.max_size = max_size
        self.auto_rotate = auto_rotate
        self.verbose = verbose
        self.extra = {}  # Extra information to be returned by ocr()

    def ocr(
        self, img: Image.Image, return_extra: bool = False
    ) -> Union[list[dict[str, Union[BBox, str]]], tuple[list[dict[str, Union[BBox, str]]], dict[str, Any]]]:
        """Returns OCR result as a list of dicts, one per bounding box detected.

        Args:
            img: Image to be processed
            return_extra: If True, additionally returns a dict containing extra information given by the OCR engine.
        Returns:
            A list of dicts, one per bounding box detected. Each dict contains at least the keys "bbox" and "text", specifying the
            location of the bounding box and the text contained respectively. Specific OCR engines may add other keys to
            these dicts to return additional information about a bounding box.
            If ``return_extra`` is True, returns a tuple with the usual return list, and a dict containing extra
            information about the whole document (e.g. rotaion angle) given by the OCR engine.
        """
        # Keep copy of original image
        original_img = img.copy()
        # Resize image if needed. If the image is smaller than max_size, it will be returned as is
        if self.max_size is not None:
            img = self._resize_image(img, self.max_size)
        # Get response from an OCR engine
        response = self._get_ocr_response(img)
        # Convert the response to our internal format
        result = self._convert_ocr_response(img, response)

        if self.auto_rotate and "document_rotation" in self.extra:
            angle = self.extra["document_rotation"]
            # Rotate image
            self.extra["rotated_image"] = rotate_image(original_img, angle)
            # Rotate boxes. The given rotation will be done counter-clockwise
            for r in result:
                r["bbox"] = r["bbox"].rotate(angle)

        if return_extra:
            return result, self.extra
        return result

    @staticmethod
    def _resize_image(img: Image.Image, max_size: int) -> Image.Image:
        """Resize the image to a maximum size, keeping the aspect ratio."""
        width, height = img.size
        if width > max_size or height > max_size:
            if width > height:
                new_width = max_size
                new_height = int(max_size * height / width)
            else:
                new_height = max_size
                new_width = int(max_size * width / height)
            img = img.resize((new_width, new_height), resample=Image.Resampling.LANCZOS)
        return img

    @abstractmethod
    def _get_ocr_response(self, img: Image.Image) -> Any:
        pass

    @abstractmethod
    def _convert_ocr_response(self, img: Image.Image, response) -> list[dict[str, Union[BBox, str]]]:
        pass

    @staticmethod
    def draw(image: Image.Image, boxes: list[BBox], texts: list[str]):
        """draw the bounding boxes over the original image to visualize result"""
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        all_text = []
        for box, text in zip(boxes, texts):
            corners = box.to_pixels()
            draw.polygon(corners, fill=None, outline="red")
            all_text.append(text)
        return image, " ".join(all_text)

    @staticmethod
    def _pil_img_to_png(image: Image.Image) -> bytes:
        """Converts a pil image to png in memory"""
        with BytesIO() as output:
            image.save(output, "PNG")
            output.seek(0)
            return output.read()

    @staticmethod
    def _get_bytes_hash(_bytes):
        """Returns the sha256 hash in hex form of a bytes object"""
        h = sha256()
        h.update(_bytes)
        img_hash = h.hexdigest()
        return img_hash

    def _get_from_shelf(self, img: Image.Image):
        """Get a OCR response from the cache, if it exists."""
        if self.cache_file is not None:
            with shelve.open(self.cache_file) as db:
                img_bytes = self._pil_img_to_png(img)
                img_hash = self._get_bytes_hash(img_bytes)
                if img_hash in db.keys():  # We have a cached version
                    if self.verbose:
                        print(f"Using cached results for hash {img_hash}")
                    return db[img_hash]
        return None

    def _put_on_shelf(self, img: Image.Image, response):
        if self.cache_file is not None:
            with shelve.open(self.cache_file) as db:
                img_bytes = self._pil_img_to_png(img)
                img_hash = self._get_bytes_hash(img_bytes)
                db[img_hash] = response
