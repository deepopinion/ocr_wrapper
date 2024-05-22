from __future__ import annotations

import dbm
import os
import shelve
import warnings
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha256
from io import BytesIO
from threading import Lock
from typing import Any, Optional, Union, cast, overload, Literal

from PIL import Image, ImageDraw, ImageOps

from .aggregate_multiple_responses import aggregate_ocr_samples, generate_img_sample
from .bbox import BBox
from .compat import bboxs2dicts, dicts2bboxs
from .tilt_correction import correct_tilt
from .data_clean_utils import split_date_boxes
from .bbox_utils import merge_bbox_lists_with_confidences
from .qr_barcodes import detect_qr_barcodes


class OcrCacheDisabled:
    pass


OCR_CACHE_DISABLED = OcrCacheDisabled()


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
    and ``_convert_ocr_response``."""

    def __init__(
        self,
        *,
        cache_file: Union[str, None, OcrCacheDisabled] = None,
        max_size: Optional[int] = 1024,
        auto_rotate: bool = False,  # Compensate for multiples of 90deg rotation (after OCR using OCR info)
        correct_tilt: bool = True,  # Compensate for small rotations (purely based on image content)
        ocr_samples: int = 2,
        supports_multi_samples: bool = False,
        add_checkboxes: bool = False,
        add_qr_barcodes: bool = False,
        verbose: bool = False,
    ):
        if cache_file == OCR_CACHE_DISABLED:
            cache_file = None
        else:
            cache_file = cache_file or os.getenv("OCR_WRAPPER_CACHE_FILE", None)

        self.cache_file = cast(Optional[str], cache_file)
        self.max_size = max_size
        self.auto_rotate = auto_rotate
        self.correct_tilt = correct_tilt
        self.ocr_samples = ocr_samples
        self.supports_multi_samples = supports_multi_samples
        self.verbose = verbose
        self.add_checkboxes = add_checkboxes
        # Currently only GoogleAzureOCR (which does not inherit from this class) supports checkboxes, so we print a warning if it's enabled
        if self.add_checkboxes:
            warnings.warn("Checkbox detection is only supported by GoogleAzureOCR")
        self.add_qr_barcodes = add_qr_barcodes
        self.shelve_mutex = Lock()  # Mutex to ensure that only one thread is writing to the cache file at a time

    @overload
    def ocr(self, img: Image.Image, return_extra: Literal[False]) -> list[BBox]: ...
    @overload
    def ocr(self, img: Image.Image, return_extra: Literal[True]) -> tuple[list[BBox], dict]: ...
    @overload
    def ocr(self, img: Image.Image, return_extra: bool) -> Union[list[BBox], tuple[list[BBox], dict]]: ...
    def ocr(self, img: Image.Image, return_extra: bool = False):
        """Returns OCR result as a list of normalized BBox

        Args:
            img: Image to be processed
            return_extra: If True, returns a tuple of (bboxes, extra) where extra is a dict containing extra information
        """
        extra = {}

        # Correct tilt (i.e. small rotation)
        if self.correct_tilt:
            img, tilt_angle = correct_tilt(img)
            extra["rotated_image"] = img
            extra["tilt_angle"] = tilt_angle

        # Keep copy of the image in its full size
        full_size_img = img.copy()
        # Resize image if needed. If the image is smaller than max_size, it will be returned as is
        if self.max_size is not None:
            img = self._resize_image(img, self.max_size)

        # Get response from an OCR engine
        if self.ocr_samples == 1 or not self.supports_multi_samples:
            if self.ocr_samples > 1 and self.verbose:
                print("Warning: This OCR engine does not support multiple samples. Using only one sample.")
            ocr = self._get_ocr_response(img)
            bboxes, sample_extra = self._convert_ocr_response(ocr)
            # Normalize all boxes
            width, height = img.size
            bboxes = [bbox.to_normalized(img_width=width, img_height=height) for bbox in bboxes]
        else:
            bboxes, sample_extra = self._get_multi_response(img)
        extra.update(sample_extra)

        # Convert confidences to a list of lists
        if "confidences" in extra:
            extra["confidences"] = [extra["confidences"]]

        if self.auto_rotate and "document_rotation" in extra:
            angle = extra["document_rotation"]
            # Rotate image
            extra["rotated_image"] = rotate_image(full_size_img, angle)
            # Rotate boxes. The given rotation will be done counter-clockwise
            bboxes = [bbox.rotate(angle) for bbox in bboxes]

        # Split date-range boxes
        confidences = cast(list[float], extra["confidences"][0])
        bboxes, confidences = split_date_boxes(bboxes, confidences)

        # Detect and add QR and barcodes if needed
        if self.add_qr_barcodes:
            qr_barcodes = detect_qr_barcodes(full_size_img)
            qr_dummy_confidences = [1.0] * len(qr_barcodes)
            bboxes, confidences = merge_bbox_lists_with_confidences(
                bboxes,
                confidences,
                qr_barcodes,
                qr_dummy_confidences,
                document_width=full_size_img.width,
                document_height=full_size_img.height,
            )

        extra["confidences"] = [confidences]

        if return_extra:
            return bboxes, extra
        return bboxes

    @overload
    def multi_img_ocr(
        self, imgs: list[Image.Image], return_extra: Literal[False], max_workers: int = ...
    ) -> list[list[BBox]]: ...
    @overload
    def multi_img_ocr(
        self, imgs: list[Image.Image], return_extra: Literal[True], max_workers: int = ...
    ) -> tuple[list[list[BBox]], list[dict]]: ...
    @overload
    def multi_img_ocr(
        self, imgs: list[Image.Image], return_extra: bool, max_workers: int = ...
    ) -> Union[list[list[BBox]], tuple[list[list[BBox]], list[dict]]]: ...
    def multi_img_ocr(self, imgs: list[Image.Image], return_extra: bool = False, max_workers: int = 32):
        """Returns OCR result for a list of images instead of a single image.

        Depending on the specific wrapper, might execute faster than calling ocr() multiple times.

        Args:
            imgs: Images to be processed
            return_extra: If True, returns a tuple of (bboxes, extra) where extra is a list of dicts containing extra information
            max_workers: Maximum number of threads to use for parallel processing
        """
        results = []
        for img in imgs:
            results.append(self.ocr(img, return_extra=return_extra))

        if return_extra:
            bboxes, extras = zip(*results)
            return list(bboxes), list(extras)
        else:
            return results

    def _get_multi_response(self, img: Image.Image) -> tuple[list[BBox], dict[str, Any]]:
        """Get OCR response from multiple samples of the same image.

        The processing of the individual samples is done in parallel (using threads).
        This does not run on multiple cores, but it is mitigating the latency of calling
        the external OCR engine multiple times.
        """
        responses: list[tuple[list[BBox], dict[str, Any]] | None] = [None] * self.ocr_samples

        # Get individual OCR responses in parallel
        def process_sample(i: int):
            img_sample = generate_img_sample(img, i)
            extra = {"img_samples": img_sample}
            response = self._get_ocr_response(img_sample)
            result, sample_extra = self._convert_ocr_response(response)
            extra.update(sample_extra)

            # Normalize boxes
            width, height = img_sample.size
            result = [bbox.to_normalized(img_width=width, img_height=height) for bbox in result]
            return result, extra, i

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(process_sample, i) for i in range(self.ocr_samples)}

            for future in as_completed(futures):
                response, extra, i = future.result()
                responses[i] = (response, extra)

        # Convert to new format as dicts
        new_format_responses = []
        extra = {"img_samples": []}
        for response_tuple in responses:
            assert response_tuple is not None
            response, sample_extra = response_tuple
            new_format_response = bboxs2dicts(response, sample_extra.pop("confidences"))
            new_format_responses.append(new_format_response)
            extra["img_samples"].append(sample_extra.pop("img_samples"))
            extra.update(sample_extra)  # Overwrite extra with the last sample's remaining extra keys

        response = aggregate_ocr_samples(new_format_responses, img.size)

        # Convert back to old format
        response_old_format, new_confidences = dicts2bboxs(response)
        extra["confidences"] = new_confidences
        return response_old_format, extra

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
    def _convert_ocr_response(self, response) -> tuple[list[BBox], dict[str, Any]]:
        pass

    @staticmethod
    def draw(image: Image.Image, boxes: list[BBox]):
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
    def _pil_img_to_compressed(image: Image.Image, compression: str = "png") -> bytes:
        """Converts a pil image to "compressed" image (e.g. png, webp) in memory"""
        with BytesIO() as output:
            if compression.lower() == "png":
                image.save(output, "PNG", compress_level=5)
            elif compression.lower() == "webp":
                image.save(output, "WebP", lossless=True, quality=0)
            else:
                raise Exception(f"Unsupported compression: {compression}")

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
            with self.shelve_mutex:
                try:
                    with shelve.open(self.cache_file, "r") as db:
                        img_bytes = self._pil_img_to_compressed(img)
                        img_hash = self._get_bytes_hash(img_bytes)
                        if img_hash in db:  # We have a cached version
                            if self.verbose:
                                print(f"Using cached results for hash {img_hash}")
                            return db[img_hash]
                except dbm.error:
                    pass  # db could not be opened

    def _put_on_shelf(self, img: Image.Image, response):
        if self.cache_file is not None:
            with self.shelve_mutex:
                with shelve.open(self.cache_file, "c") as db:
                    img_bytes = self._pil_img_to_compressed(img)
                    img_hash = self._get_bytes_hash(img_bytes)
                    db[img_hash] = response
