"""
This file contains the GoogleAzureWrapper class, which is a wrapper that combines the GoogleOCR and AzureOCR classes.

An image is analyzed by GoogleOCR and AzureOCR in parallel, a heuristic is used to remove probably incorrect bboxes from the GoogleOCR result, and bboxes that are found by Azure and don't have a high overlap with the bboxes found by Google
are added to the final list of bboxes.
"""

from __future__ import annotations

import dbm
import os
import re
import shelve
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Literal, Optional, Union, cast, overload

import rtree
from opentelemetry import trace
from PIL import Image

from ocr_wrapper import AzureOCR, BBox, GoogleOCR
from ocr_wrapper.google_document_ocr_checkbox_detector import GoogleDocumentOcrCheckboxDetector
from ocr_wrapper.ocr_wrapper import OCR_CACHE_DISABLED, OcrCacheDisabled, rotate_image
from ocr_wrapper.tilt_correction import correct_tilt

from .bbox_utils import merge_bbox_lists
from .data_clean_utils import split_date_boxes
from .qr_barcodes import detect_qr_barcodes
from .utils import get_img_hash, set_image_attributes

tracer = trace.get_tracer(__name__)


class GoogleAzureOCR:
    """An OCR wrapper that combines the GoogleOCR and AzureOCR classes."""

    def __init__(
        self,
        cache_file: Union[None, str, OcrCacheDisabled] = None,
        ocr_samples: Optional[int] = None,
        supports_multi_samples: bool = False,
        max_size: Optional[int] = 4096,
        auto_rotate: Optional[bool] = None,
        correct_tilt: Optional[bool] = None,
        add_checkboxes: bool = False,  # If True, Document OCR by Google is used to detect checkboxes
        add_qr_barcodes: bool = False,  # If True, QR barcodes are detected and added as BBoxes
        verbose: bool = False,
    ):
        if ocr_samples is not None and ocr_samples != 1:
            print(
                "Warning: The argument ocr_samples is ignored by GoogleAzureOCR and can't be set to a value other than 1 or None"
            )
        if supports_multi_samples:
            print("Warning: The argument supports_multi_samples is ignored by GoogleAzureOCR and can't be set to True")
        if auto_rotate == False:
            print("Warning: The auto_rotate argument is ignored by GoogleAzureOCR and can't be set to False")
        if correct_tilt == False:
            print("Warning: The correct_tilt argument is ignored by GoogleAzureOCR and can't be set to False")

        if cache_file == OCR_CACHE_DISABLED:
            cache_file = None
        else:
            cache_file = cache_file or os.getenv("OCR_WRAPPER_CACHE_FILE", None)

        self.cache_file = cast(Optional[str], cache_file)
        self.max_size = max_size
        self.verbose = verbose
        self.add_checkboxes = add_checkboxes
        self.add_qr_barcodes = add_qr_barcodes
        self.shelve_mutex = Lock()  # Mutex to ensure that only one thread is writing to the cache file at a time

    @overload
    def ocr(self, img: Image.Image, return_extra: Literal[False]) -> list[BBox]: ...
    @overload
    def ocr(self, img: Image.Image, return_extra: Literal[True]) -> tuple[list[BBox], dict]: ...
    @overload
    def ocr(self, img: Image.Image, return_extra: bool = False) -> Union[list[BBox], tuple[list[BBox], dict]]: ...

    @tracer.start_as_current_span(name="GoogleAzureOCR.ocr")
    def ocr(self, img: Image.Image, return_extra: bool = False):
        """Runs OCR on an image using both Google and Azure OCR, and combines the results.

        Args:
            img (Image.Image): The image to run OCR on.
            return_extra (bool, optional): Whether to return extra information. Defaults to False.
        """
        span = trace.get_current_span()
        set_image_attributes(span, img)

        # Return cached result if it exists
        if self.cache_file is not None:
            img_hash = get_img_hash(img)
            cached = self._get_from_shelf(img_hash, return_extra)
            if cached is not None:
                span.add_event("Using cached results")
                return cached

        google_ocr = GoogleOCR(
            cache_file=OCR_CACHE_DISABLED,
            auto_rotate=True,
            correct_tilt=False,
            ocr_samples=1,
            max_size=self.max_size,
            verbose=self.verbose,
        )
        azure_ocr = AzureOCR(
            cache_file=OCR_CACHE_DISABLED,
            auto_rotate=False,
            correct_tilt=False,
            ocr_samples=1,
            max_size=self.max_size,
            verbose=self.verbose,
        )
        if self.add_checkboxes:
            checkbox_ocr = GoogleDocumentOcrCheckboxDetector()

        # Do the tilt angle correction ourselves externally to have consistend input to Google and Azure
        img, tilt_angle = correct_tilt(img)
        span.set_attribute("tilt_angle", tilt_angle)

        # Run Google OCR and Azure OCR in parallel via theadpool
        with tracer.start_as_current_span(name="GoogleAzureOCR.ocr.threadpool") as span:
            with ThreadPoolExecutor(max_workers=2) as executor:
                future_google = executor.submit(google_ocr.ocr, img, return_extra=True)
                future_azure = executor.submit(azure_ocr.ocr, img, return_extra=True)
                if self.add_checkboxes:
                    future_checkbox = executor.submit(checkbox_ocr.detect_checkboxes, img)
                if self.add_qr_barcodes:
                    future_qr_barcodes = executor.submit(detect_qr_barcodes, img)

                google_bboxes, google_extra = future_google.result()
                span.set_attribute("nb_google_bboxes", len(google_bboxes))
                azure_bboxes, _ = future_azure.result()
                span.set_attribute("nb_azure_bboxes", len(azure_bboxes))
                if self.add_checkboxes:
                    checkbox_bboxes, _ = future_checkbox.result()
                    span.set_attribute("nb_checkbox_bboxes", len(checkbox_bboxes))
                if self.add_qr_barcodes:
                    qr_bboxes = future_qr_barcodes.result()
                    span.set_attribute("nb_qr_bboxes", len(qr_bboxes))

        # Use the rotation information from google to correctly rotate the image and the bboxes
        google_rotation_angle = google_extra["document_rotation"]
        span.set_attribute("google_rotation_angle", google_rotation_angle)

        azure_bboxes = [bbox.rotate(google_rotation_angle) for bbox in azure_bboxes]
        if self.add_checkboxes:
            checkbox_bboxes = [bbox.rotate(google_rotation_angle) for bbox in checkbox_bboxes]
        azure_bboxes = split_date_boxes(azure_bboxes)
        img = rotate_image(img, google_rotation_angle)

        # Remove unwanted bboxes from Google OCR result
        google_bboxes = _filter_unwanted_google_bboxes(google_bboxes, width_height_ratio=img.width / img.height)
        span.set_attribute("nb_filtered_google_bboxes", len(google_bboxes))

        # Combine the bboxes from Google and Azure
        bbox_overlap_checker = BBoxOverlapChecker(google_bboxes)
        azure_bboxes_to_add = []

        for bbox in azure_bboxes:
            if len(bbox_overlap_checker.get_overlapping_bboxes(bbox)) == 0:
                azure_bboxes_to_add.append(bbox)

        document_width, document_height = img.size
        combined_bboxes = merge_bbox_lists(
            google_bboxes,
            azure_bboxes_to_add,
            document_width=document_width,
            document_height=document_height,
        )
        span.set_attribute("nb_combined_bboxes", len(combined_bboxes))

        if self.add_checkboxes:
            # Remove bboxes that have a high overlap with detected checkboxes since sometimes
            # an x etc. is detected for a checkbox
            checkbox_overlap_checker = BBoxOverlapChecker(checkbox_bboxes)
            combined_bboxes = [
                bbox
                for bbox in combined_bboxes
                if len(checkbox_overlap_checker.get_overlapping_bboxes(bbox, threshold=0.5)) == 0
            ]

            # Merge in the checkbox bboxes
            combined_bboxes = merge_bbox_lists(
                combined_bboxes,
                checkbox_bboxes,
                document_width=document_width,
                document_height=document_height,
            )
            span.set_attribute("nb_combined_bboxes_after_checkboxes", len(combined_bboxes))

        if self.add_qr_barcodes:
            combined_bboxes = merge_bbox_lists(
                combined_bboxes,
                qr_bboxes,
                document_width=document_width,
                document_height=document_height,
            )
            span.set_attribute("nb_combined_bboxes_after_qr_barcodes", len(combined_bboxes))

        # Build extra information dict
        extra = {
            "document_rotation": google_rotation_angle,
            "tilt_angle": tilt_angle,
            "confidences": [[0.9] * len(combined_bboxes)],
            "rotated_image": img,
        }

        if return_extra:
            result = combined_bboxes, extra
        else:
            result = combined_bboxes

        if self.cache_file is not None:
            self._put_on_shelf(img_hash, return_extra, result)  # Cache result # type: ignore
        return result

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
    @tracer.start_as_current_span(name="GoogleAzureOCR.multi_img_ocr")
    def multi_img_ocr(self, imgs: list[Image.Image], return_extra: bool = False, max_workers: int = 32):
        """Runs OCR in parallel on multiple images using both Google and Azure OCR, and combines the results.

        Args:
            img (list[Image.Image]): The pages to run OCR on.
            return_extra (bool, optional): Whether to return extra information. Defaults to False.
            max_workers (int, optional): The maximum number of threads to use. Defaults to 32.
        """
        span = trace.get_current_span()
        span.set_attribute("nb_images", len(imgs))
        span.set_attribute("max_workers", max_workers)
        # Execute self.ocr in parallel on all images
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self.ocr, img, return_extra) for img in imgs]
            results = [future.result() for future in futures]

        if return_extra:
            bboxes: list[list[BBox]]
            extras: list[dict]
            bboxes, extras = zip(*results)
            return bboxes, extras

        else:
            results = cast(list[list[BBox]], results)
            return results

    @tracer.start_as_current_span(name="GoogleAzureOCR._get_from_shelf")
    def _get_from_shelf(self, img_hash: str, return_extra: bool):
        """Get a OCR response from the cache, if it exists."""
        if self.cache_file is not None:
            hash_str = repr(("googleazure", img_hash, return_extra))
            with self.shelve_mutex:
                try:
                    with shelve.open(self.cache_file, "r") as db:
                        if hash_str in db:  # We have a cached version
                            if self.verbose:
                                print(f"Using cached results for hash {hash_str}")
                            return db[hash_str]
                except dbm.error:
                    pass  # db could not be opened

    @tracer.start_as_current_span(name="GoogleAzureOCR._put_on_shelf")
    def _put_on_shelf(self, img_hash: str, return_extra: bool, response):
        if self.cache_file is not None:
            hash_str = repr(("googleazure", img_hash, return_extra))
            with self.shelve_mutex:
                with shelve.open(self.cache_file, "c") as db:
                    db[hash_str] = response


class BBoxOverlapChecker:
    """
    Class to check whether a bbox overlaps with any of a list of bboxes.

    Uses an RTree to quickly find overlapping bboxes.

    Args:
        bboxes (list[BBox]): The bboxes that will be checked for overlap against
    """

    def __init__(self, bboxes: list[BBox]):
        self.bboxes = bboxes
        self.rtree = rtree.index.Index()
        for i, bbox in enumerate(bboxes):
            self.rtree.insert(i, bbox.get_shapely_polygon().bounds)

    def get_overlapping_bboxes(self, bbox: BBox, threshold: float = 0.01) -> list[BBox]:
        """Returns the bboxes that overlap with the given bbox.

        Args:
            bbox (BBox): The bbox to check for overlap.
            threshold (float, optional): The minimum overlap that is required for a bbox to be returned (0.0 to 1.0).
                Defaults to 0.01. Overlap is checked in both directions.

        Returns:
            list[BBox]: The bboxes that overlap with the given bbox.
        """
        span = trace.get_current_span()
        span.set_attribute("threshold", threshold)

        overlapping_bboxes = []
        for i in self.rtree.intersection(bbox.get_shapely_polygon().bounds):
            if (
                bbox.intersection_area_percent(self.bboxes[i]) > threshold
                or self.bboxes[i].intersection_area_percent(bbox) > threshold
            ):
                overlapping_bboxes.append(self.bboxes[i])

        return overlapping_bboxes


def _get_box_height(bbox: BBox) -> float:
    """Returns the height of the bbox

    Args:
        bbox (BBox): The bbox to get the height of.

    Returns:
        float: The height of the bbox.
    """
    return abs(bbox.BLy - bbox.TLy)


def _get_median_box_height(bboxes: list[BBox]) -> float:
    """Returns the median height of the bboxes in the list

    Args:
        bboxes (list[BBox]): The bboxes to get the median height of.

    Returns:
        float: The median height of the bboxes.
    """
    if len(bboxes) == 0:
        return 0.0
    heights = [_get_box_height(bbox) for bbox in bboxes]
    heights.sort()
    n = len(heights)
    if n % 2 == 0:
        return (heights[n // 2 - 1] + heights[n // 2]) / 2
    return heights[n // 2]


def _bbox_is_vertically_aligned(bb: BBox, width_height_ratio: float) -> bool:
    """Returns whether a bbox is vertically aligned.

    Args:
        bb (BBox): The bbox to check.

    Returns:
        bool: Whether the bbox is vertically aligned.
    """
    width = abs(bb.BRx - bb.TLx) * width_height_ratio
    height = abs(bb.BLy - bb.TLy)

    return width < height


def _filter_date_boxes(bboxes: list[BBox], max_boxes_range: int = 10) -> list[BBox]:
    """
    Filters out bounding boxes that, concatenated, match patterns like "dd/mm/yyyy - dd/mm/yyyy".

    Args:
        bboxes (list[BBox]): The bboxes to filter.
        max_boxes_range (int, optional): The maximum number of bboxes to consider for a match. Defaults to 10.
    """
    max_boxes_range = min(max_boxes_range, len(bboxes))
    date_range_pattern = r"^\s*\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4}\s*-\s*\d{1,2}\s*/\s*\d{1,2}\s*/\s*\d{4}\s*$"

    # Function to get all overlapping consecutive elements of a list of a given length
    # e.g. consecutive_elements([1, 2, 3, 4], 2) -> [(1, 2), (2, 3), (3, 4)]
    def consecutive_elements(lst, n):
        return zip(*(lst[i:] for i in range(n)))

    # Function to check if the concatenation of a given combination of strings matches the pattern
    def is_match(combination):
        concatenated = "".join(c.text for c in combination).replace(" ", "")
        _match = re.match(date_range_pattern, concatenated)
        return _match

    # Generate all combinations of strings in the list of different lengths
    for r in range(max_boxes_range, 1, -1):
        for comb in consecutive_elements(bboxes, r):
            if is_match(comb):
                # Remove the strings in the combination from the original list
                for item in comb:
                    if item in bboxes:
                        bboxes.remove(item)
                return _filter_date_boxes(bboxes)  # Recursively call to find more matches

    return bboxes


@tracer.start_as_current_span(name="_filter_unwanted_google_bboxes")
def _filter_unwanted_google_bboxes(bboxes: list[BBox], width_height_ratio: float) -> list[BBox]:
    """Filters out probably incorrect bboxes from the GoogleOCR result.

    Currently does the following filtering:
    - Removes bboxes with an area that is bigger than the median area of all bboxes in the list and that are vertically aligned
    - Filters out bounding boxes that, concatenated, match patterns like "dd/mm/yyyy - dd/mm/yyyy".

    Args:
        bboxes (list[BBox]): The bboxes to filter.
        width_height_ratio (float): The width to height ratio of the image.

    Returns:
        list[BBox]: The bboxes with the filtered out bboxes removed.
    """
    span = trace.get_current_span()
    span.set_attribute("width_height_ratio", width_height_ratio)

    median_height = _get_median_box_height(bboxes)
    span.set_attribute("median_height", median_height)
    filtered_bboxes: list[BBox] = []
    for bbox in bboxes:
        # Don't include bboxes that are higher than the median height (+5%) and are vertically aligned
        # The +5% is to account for cases where all bounding boxes are basically the same height and we don't want to filter
        # out arbitrary bboxes that just happen to be a bit higher and vertical
        # We try to filter out columns of digits that are detected as a single bbox (which happens in GoogleOCR sometimes)
        if _get_box_height(bbox) > median_height * 1.05 and _bbox_is_vertically_aligned(bbox, width_height_ratio):
            # Never filter out single character bboxes
            if bbox.text is not None and len(bbox.text.strip()) != 1:
                continue
        filtered_bboxes.append(bbox)

    filtered_bboxes = _filter_date_boxes(filtered_bboxes)
    span.set_attribute("nb_filtered_bboxes", len(filtered_bboxes))
    return filtered_bboxes
