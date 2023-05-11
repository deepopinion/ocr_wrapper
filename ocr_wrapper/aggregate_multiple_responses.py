"""
Functions to generate multiple OCR responses and integrate them into a single response, improving the overall
quality of the response.
"""
import rtree
from PIL import Image

from ocr_wrapper.bbox_utils import bbox_intersection_area_percent


def check_overlap(args):
    bbox, other_bboxes, threshold = args
    results = []
    for other_bbox in other_bboxes:
        overlap1 = bbox_intersection_area_percent(bbox["bbox"], other_bbox["bbox"])
        overlap2 = bbox_intersection_area_percent(other_bbox["bbox"], bbox["bbox"])
        if overlap1 > threshold and overlap2 > threshold:
            results.append(other_bbox)
    return results


def _get_poly_intersection_area(p1, p2):
    inter_poly = p1.intersection(p2)
    return inter_poly.area / p1.area


def _find_overlapping_bboxes(bbox: dict, idx, orig_polygons, poly2bbox, threshold: float) -> list[dict]:
    overlapping_bboxes = [bbox]

    polygon = bbox["bbox"].get_shapely_polygon()
    potential_matches = [orig_polygons[pos] for pos in idx.intersection(polygon.bounds)]
    for match in potential_matches:
        if polygon == match:
            continue

        overlap1 = _get_poly_intersection_area(polygon, match)
        overlap2 = _get_poly_intersection_area(match, polygon)
        if overlap1 > threshold and overlap2 > threshold:
            overlapping_bboxes.append(poly2bbox[match])

    return overlapping_bboxes


def _group_overlapping_bboxes(bboxes: list[dict], threshold: float) -> list[list[dict]]:
    """
    Group the bounding boxes that have an overlap greater than the given threshold.

    Args:
    bboxes: The dictionaries containing the bounding boxes and other information.
    threshold (float): The threshold for the percentage of overlap.

    Returns:
    list: A list of lists containing the groups of overlapping bounding boxes.
    """
    # Create a list of polygons and a dictionary mapping back polygons to bounding boxes
    polygons = []
    poly2bbox = {}

    for bbox in bboxes:
        poly = bbox["bbox"].get_shapely_polygon()
        polygons.append(poly)
        poly2bbox[poly] = bbox

    # Create an rtree index for the polygons so we can quickly find intersecting polygons
    idx = rtree.index.Index()
    poly2treeid = {}
    for i, polygon in enumerate(polygons):
        idx.insert(i, polygon.bounds)
        poly2treeid[polygon] = i

    orig_polygons = polygons.copy()

    groups = []
    while len(polygons) > 0:
        polygon = polygons.pop(0)
        bbox = poly2bbox[polygon]
        overlapping_bboxes = _find_overlapping_bboxes(bbox, idx, orig_polygons, poly2bbox, threshold)
        groups.append(overlapping_bboxes)
        for overlapping_bbox in overlapping_bboxes:
            if overlapping_bbox != bbox:
                polygon = overlapping_bbox["bbox"].get_shapely_polygon()
                polygons.remove(polygon)
                idx.delete(poly2treeid[polygon], polygon.bounds)

    return groups
    # Create the grouped bounding boxes, depending on the overlap threshold

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


def generate_img_sample(img: Image.Image, n: int, *, k: float = 0.2) -> Image.Image:
    """Takes an image and a sample number and returns a new image that has been changed in some way.
    Currently we are only resizing the image.

    If n=0, the original image is returned.

    Args:
        img: The image to be changed
        n: The sample number
        k: The factor by which the image is resized (bigger means for each increase of n, the image is resized more)
    """
    if n == 0:
        return img

    factor = 1 / (1 + n * k)
    new_size = tuple(int(x * factor) for x in img.size)
    return img.resize(new_size, resample=Image.Resampling.LANCZOS)


def _get_overall_confidence(responses: list[dict]) -> float:
    """Returns the overall confidence of an OCR response as the mean confidence of all bounding boxes.
    The confidence is calculated as the average of the individual bbox confidences.

    If no confidence is available, 0 is returned
    """
    try:
        overall_confidence = sum(response["confidence"] for response in responses) / len(responses)
    except KeyError:
        overall_confidence = 0

    return overall_confidence


def _get_highest_confidence_response(responses: list[list[dict]]) -> list[dict]:
    """Returns the response with the highest confidence and its id"""
    best_response = max(responses, key=lambda x: _get_overall_confidence(x))

    return best_response


def _add_single_bboxes(
    best_response: list[dict],
    bbox_groups: list[list[dict]],
    overlap_threshold: float = 0.5,
) -> list[dict]:
    """
    Adds single bounding boxes from bbox_group to the best_response if their overlap with any
    bounding box in best_response is less than the overlap_threshold.

    This can be used to enrich one response with bounding boxes that have been missed

    Args:
        best_response (list[dict]): A list of dictionaries representing the best response bounding boxes.
        bbox_groups (list[list[dict]]): A list of lists, each containing dictionaries representing bounding boxes.
        overlap_threshold (float, optional): The threshold for overlapping bounding boxes. Defaults to 0.1.
            If the overlap is bigger than this, we don't consider it to be a new bounding box.

    Returns:
        list[dict]: The updated best_response list with single bounding boxes added.
    """
    for bbox_group in bbox_groups:
        if len(bbox_group) == 1:  # Check if we have a single bbox
            bbox_candidate = bbox_group[0]
            # Check if the bbox candidate overlaps with any other bbox that is already in the best response
            # This will also be the case if the bbox candidate is already in the best response
            overlaps = [
                bbox_intersection_area_percent(bbox_candidate["bbox"], best_bbox["bbox"])
                for best_bbox in best_response
            ]
            highest_overlap = max(overlaps, default=0)

            if highest_overlap < overlap_threshold:
                best_response.append(bbox_candidate)

    return best_response


def aggregate_ocr_samples(responses: list[list[dict]], original_width: int, original_height: int) -> list[dict]:
    """Given multiple responses of the same document page, aggregates them into a more reliable response."""
    if len(responses) == 1:  # If there is only one response, return it unchanged
        return responses[0]
    else:
        # Extract all bounding boxes from the responses and add the response id they came from
        # The bounding boxes are actually dictionaries containing the bbox and additional other info like the text
        #    and possibly the uncertainty etc.
        bboxes = []
        for i, response in enumerate(responses):
            for res_dict in response:
                res_dict["response_id"] = i
                bboxes.append(res_dict)

        # Group the bounding boxes that overlap with each other given a threshold
        # We are grouping here to find bounding boxes that are completely novel to one of the responses
        bbox_groups = _group_overlapping_bboxes(bboxes, 0.1)

        # Determine response with the overall highest confidence. We will use that one as the basis response we try to improve
        best_response = _get_highest_confidence_response(responses)

        # Add all bboxes which are not overlapping with any other bbox and are not already part of the current best response
        # This is done to enrich the best selected response with bounding boxes that might have been missed (which is a
        # common fault of many OCR solutions)
        best_response = _add_single_bboxes(best_response, bbox_groups)

        # Assign the original image size to all bboxes
        for bbox_dict in best_response:
            bbox_dict["bbox"].original_width = original_width
            bbox_dict["bbox"].original_height = original_height

        return best_response
