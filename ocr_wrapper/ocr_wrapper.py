from __future__ import annotations
from abc import ABC, abstractmethod
from hashlib import sha256

import shelve
from io import BytesIO
from typing import Optional, List

from PIL import Image, ImageDraw, ImageOps
from .bbox import BBox


class OcrWrapper(ABC):
    """Base class for OCR engines. Subclasses must implement ``_get_ocr_response``
    and ``_conver_ocr_response``."""

    def __init__(
        self,
        *,
        cache_file: Optional[str] = None,
        max_size: Optional[int] = 1024,
        verbose: bool = False,
    ):
        self.cache_file = cache_file
        self.max_size = max_size
        self.verbose = verbose

    def ocr(self, img: Image.Image) -> List[BBox]:
        """Returns OCR result as a list of normalized BBox"""
        # Resize image if needed
        if self.max_size is not None:
            img = self._resize_image(
                img, self.max_size
            )  # If the image is smaller than max_size, it will be returned as is
        # Get response from an OCR engine
        response = self._get_ocr_response(img)
        # Convert the response to our internal format
        bboxes = self._convert_ocr_response(response)
        # Normalize all boxes
        width, height = img.size
        bboxes = [
            bbox.to_normalized(img_width=width, img_height=height) for bbox in bboxes
        ]
        return bboxes

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
    def _get_ocr_response(self, img: Image.Image):
        pass

    @abstractmethod
    def _convert_ocr_response(self, response) -> List[BBox]:
        pass

    @staticmethod
    def draw(image: Image.Image, boxes: List[BBox]):
        """draw the bounding boxes over the original image to visualize result"""
        image = ImageOps.exif_transpose(image)
        if image.mode != "RGB":
            image = image.convert("RGB")
        draw = ImageDraw.Draw(image)
        all_text = []
        for box in boxes:
            unnormalized_box = box.to_pixels(*image.size)
            corners = unnormalized_box.get_float_list()
            text = box.text
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
