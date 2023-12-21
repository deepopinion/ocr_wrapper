"""
This file contains the GoogleAzureWrapper class, which is a wrapper that combines the GoogleOCR and AzureOCR classes.

An image is analyzed by GoogleOCR and AzureOCR in parallel, a heuristic is used to remove probably incorrect bboxes from the GoogleOCR result, and bboxes that are found by Azure and don't have a high overlap with the bboxes found by Google
are added to the final list of bboxes.
"""
from __future__ import annotations
from typing import Union, cast, Optional

from PIL import Image
from ocr_wrapper import GoogleOCR, AzureOCR, draw_bboxes
from concurrent.futures import ThreadPoolExecutor, as_completed
from ocr_wrapper import BBox
from ocr_wrapper.tilt_correction import correct_tilt
from ocr_wrapper.ocr_wrapper import rotate_image
import rtree


class GoogleAzureOCR:
    """An OCR wrapper that combines the GoogleOCR and AzureOCR classes."""

    def __init__(
        self,
        cache_file: Optional[str] = None,
        ocr_samples: int = 2,
        supports_multi_samples: bool = False,
        max_size: Optional[int] = 1024,
        auto_rotate: bool = False,  # Compensate for multiples of 90deg rotation (after OCR using OCR info)
        correct_tilt: bool = True,  # Compensate for small rotations (purely based on image content)
        verbose: bool = False,
    ):
        self.max_size = max_size
        self.google_ocr = GoogleOCR(auto_rotate=True, correct_tilt=False, ocr_samples=1, max_size=max_size)
        self.azure_ocr = AzureOCR(auto_rotate=False, correct_tilt=False, ocr_samples=1, max_size=max_size)

    def ocr(self, img: Image.Image, return_extra: bool = False) -> Union[list[BBox], tuple[list[BBox], dict]]:
        """Runs OCR on an image using both Google and Azure OCR, and combines the results.

        Args:
            img (Image.Image): The image to run OCR on.
            return_extra (bool, optional): Whether to return extra information. Defaults to False.
        """
        # Do the tilt angle correction ourselves externally to have consistend input to Google and Azure
        img, tilt_angle = correct_tilt(img)

        # Run Google OCR and Azure OCR in parallel via theadpool
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_google = executor.submit(self.google_ocr.ocr, img, return_extra=True)
            future_azure = executor.submit(self.azure_ocr.ocr, img, return_extra=True)
            google_bboxes, google_extra = cast(tuple[list[BBox], dict], future_google.result())
            azure_bboxes, azure_extra = cast(tuple[list[BBox], dict], future_azure.result())

        # Use the rotation information from google to correctly rotate the image and the bboxes
        google_rotation_angle = google_extra["document_rotation"]
        google_bboxes = [bbox.rotate(google_rotation_angle) for bbox in google_bboxes]
        azure_bboxes = [bbox.rotate(google_rotation_angle) for bbox in azure_bboxes]
        img = rotate_image(img, google_rotation_angle)

        # Remove unwanted bboxes from Google OCR result
        google_bboxes = _filter_unwanted_google_bboxes(google_bboxes)

        # Combine the bboxes from Google and Azure
        bbox_overlap_checker = BBoxOverlapChecker(google_bboxes)
        combined_bboxes = google_bboxes.copy()
        for bbox in azure_bboxes:
            if len(bbox_overlap_checker.get_overlapping_bboxes(bbox)) == 0:
                combined_bboxes.append(bbox)

        # Build extra information dict
        extra = {
            "document_rotation": google_rotation_angle,
            "tilt_angle": tilt_angle,
            "confidences": [[0.9] * len(combined_bboxes)],
            "rotated_image": img,
        }

        if return_extra:
            return combined_bboxes, extra
        else:
            return combined_bboxes


class BBoxOverlapChecker:
    def __init__(self, bboxes: list[BBox]):
        self.bboxes = bboxes
        self.rtree = rtree.index.Index()
        for i, bbox in enumerate(bboxes):
            self.rtree.insert(i, bbox.get_shapely_polygon().bounds)

    def get_overlapping_bboxes(self, bbox: BBox, threshold: float = 0.1) -> list[BBox]:
        """Returns the bboxes that overlap with the given bbox.

        Args:
            bbox (BBox): The bbox to check for overlapping bboxes.
            threshold (float, optional): The minimum overlap that is required for a bbox to be returned (0.0 to 1.0).
                Defaults to 0.1. Overlap is checked in both directions.

        Returns:
            list[BBox]: The bboxes that overlap with the given bbox.
        """
        overlapping_bboxes = []
        for i in self.rtree.intersection(bbox.get_shapely_polygon().bounds):
            if (
                bbox.intersection_area_percent(self.bboxes[i]) > threshold
                or self.bboxes[i].intersection_area_percent(bbox) > threshold
            ):
                overlapping_bboxes.append(self.bboxes[i])
        return overlapping_bboxes


def _get_mean_bbox_area(bboxes: list[BBox]) -> float:
    """Returns the mean area of the bboxes in the list

    Args:
        bboxes (list[BBox]): The bboxes to get the mean area of.

    Returns:
        float: The mean area of the bboxes.
    """
    return sum(bbox.area() for bbox in bboxes) / len(bboxes)


def _bbox_is_vertically_aligned(bb: BBox) -> bool:
    """Returns whether a bbox is vertically aligned.

    Args:
        bb (BBox): The bbox to check.

    Returns:
        bool: Whether the bbox is vertically aligned.
    """
    width = abs(bb.BRx - bb.TLx)
    height = abs(bb.BLy - bb.TLy)
    return width < height


def _filter_unwanted_google_bboxes(bboxes: list[BBox]) -> list[BBox]:
    """Filters out probably incorrect bboxes from the GoogleOCR result.

    Currently does the following filtering:
    - Removes bboxes with an area that is bigger than the mean area of all bboxes in the list and that are vertically aligned

    Args:
        bboxes (list[BBox]): The bboxes to filter.

    Returns:
        list[BBox]: The filtered bboxes.
    """
    mean_area = _get_mean_bbox_area(bboxes)
    filtered_bboxes = []
    for bbox in bboxes:
        if bbox.area() < mean_area or not _bbox_is_vertically_aligned(bbox):
            filtered_bboxes.append(bbox)
    return filtered_bboxes
