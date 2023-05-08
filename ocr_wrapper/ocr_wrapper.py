from __future__ import annotations

import shelve
from abc import ABC, abstractmethod
from hashlib import sha256
from io import BytesIO
from typing import Any, Optional, Union

from PIL import Image, ImageDraw, ImageOps

from .bbox import BBox
from .bbox_utils import bbox_intersection_area_percent


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


def _find_overlapping_bboxes(bbox: dict, bboxes: list[dict], threshold: float):
    """
    Find the bounding boxes that overlap with the given bounding box.

    Args:
    bbox: The reference bounding box dict.
    bboxes (list): A list of bounding boxes dicts to check for overlap.
    threshold (float): The threshold for the percentage of overlap.

    Returns:
    list: A list of overlapping bounding boxes, including the reference bounding box.
    """
    overlapping_bboxes = [bbox]
    for other_bbox in bboxes:
        overlap1 = bbox_intersection_area_percent(bbox["bbox"], other_bbox["bbox"])
        overlap2 = bbox_intersection_area_percent(other_bbox["bbox"], bbox["bbox"])
        if overlap1 > threshold and overlap2 > threshold:
            overlapping_bboxes.append(other_bbox)
            bbox["overlap"] = overlap1
            other_bbox["overlap"] = overlap2
    return overlapping_bboxes


def _group_overlapping_bboxes(bboxes, threshold: float):
    """
    Group the bounding boxes that have an overlap greater than the given threshold.

    Args:
    bboxes: The dictionaries containing the bounding boxes and other information.
    threshold (float): The threshold for the percentage of overlap.

    Returns:
    list: A list of lists containing the groups of overlapping bounding boxes.
    """
    groups = []
    bboxes = bboxes.copy()

    while len(bboxes) > 0:
        bbox = bboxes.pop(0)
        overlapping_bboxes = _find_overlapping_bboxes(bbox, bboxes, threshold)
        groups.append(overlapping_bboxes)
        for overlapping_bbox in overlapping_bboxes:
            if overlapping_bbox != bbox:
                bboxes.remove(overlapping_bbox)

    return groups


def _generate_img_sample(img: Image.Image, n: int) -> Image.Image:
    """Takes an image and a sample number and returns a new image that has been changed in some way.
    Currently we are only resizing the image.

    If  n=0, the original image is returned.
    """
    if n == 0:
        return img
    # Make the image smaller by 1/(n/2)
    new_size = tuple(int(x * (1 - n * 0.2)) for x in img.size)
    return img.resize(new_size, resample=Image.Resampling.LANCZOS)


def _get_overall_confidence(responses: list[Any]) -> float:
    """Returns the overall confidence of an OCR response as the mean confidence of all bounding boxes.
    The confidence is calculated as the average of the individual confidences.
    """
    try:
        overall_confidence = sum(response["confidence"] for response in responses) / len(responses)
    except KeyError:
        overall_confidence = 0

    return overall_confidence


def _get_highest_confidence_response(responses: list[Any]):
    """Returns the response with the highest confidence and its id"""
    best_response = responses[0]
    best_response_id = 0
    best_confidence = _get_overall_confidence(best_response)
    print(f"Confidence of response 0 = {best_confidence}")
    for i, response in enumerate(responses[1:]):
        confidence = _get_overall_confidence(response)
        print(f"Confidence of response {i+1} = {confidence}")
        if confidence > best_confidence:
            best_response = response
            best_response_id = i + 1
            best_confidence = confidence
    print(f"Best response: {best_response_id} with confidence {best_confidence}")

    return best_response, best_response_id


def _aggregate_ocr_samples(responses: list[Any]) -> Any:
    if len(responses) == 1:
        return responses[0]
    else:
        bboxes = []
        for i, response in enumerate(responses):
            for res_dict in response:
                res_dict["response_id"] = i
                bboxes.append(res_dict)

        bbox_groups = _group_overlapping_bboxes(bboxes, 0.1)

        # Determine responses with the overall highest confidence
        best_response, best_response_id = _get_highest_confidence_response(responses)

        # Add all bboxes which are not overlapping with any other bbox and are not already part of the best response
        for bbox_group in bbox_groups:
            bbox = bbox_group[0]
            if len(bbox_group) == 1 and bbox["response_id"] != best_response:
                # Check if bbox overlaps with any other bbox that is already in the best response
                highest_overlap = 0
                for best_bbox in best_response:
                    if bbox_intersection_area_percent(bbox["bbox"], best_bbox["bbox"]) > highest_overlap:
                        highest_overlap = bbox_intersection_area_percent(bbox["bbox"], best_bbox["bbox"])

                if highest_overlap < 0.5:
                    best_response.append(bbox_group[0])

        # Assign the original image size to all bboxes
        original_width = responses[0][0]["bbox"].original_width
        original_height = responses[0][0]["bbox"].original_height
        for bbox_dict in best_response:
            bbox_dict["bbox"].original_width = original_width
            bbox_dict["bbox"].original_height = original_height

        return best_response


class OcrWrapper(ABC):
    """Base class for OCR engines. Subclasses must implement ``_get_ocr_response``
    and ``_convert_ocr_response``."""

    def __init__(
        self,
        *,
        cache_file: Optional[str] = None,
        max_size: Optional[int] = 1024,
        auto_rotate: bool = False,
        ocr_samples: int = 1,
        verbose: bool = False,
    ):
        self.cache_file = cache_file
        self.max_size = max_size
        self.auto_rotate = auto_rotate
        self.ocr_samples = ocr_samples
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
        result = self._get_multi_response(img)

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

    def _get_multi_response(self, img: Image.Image) -> list[Any]:
        """Get OCR response from multiple samples of the same image."""
        responses = []
        for i in range(self.ocr_samples):
            img_sample = _generate_img_sample(img, i)
            response = self._get_ocr_response(img_sample)
            result = self._convert_ocr_response(img_sample, response)
            responses.append(result)

        response = _aggregate_ocr_samples(responses)

        return response

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
