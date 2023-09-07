"""
Additional functions for working with BBox objects.
Mainly here for compatibilty with the new ocr_wrapper package.
"""
from __future__ import annotations
from functools import lru_cache

from .bbox import BBox


@lru_cache(maxsize=16000)
def bbox_intersection_area_ratio(bb1: BBox, bb2: BBox) -> float:
    """Returns the area of the intersection of BBox `bb1` with BBox `bb2` as a ratio of the area of `bb1`.

    i.e. max is 1.0, min is 0.0
    """
    self_poly = bb1.get_shapely_polygon()
    that_poly = bb2.get_shapely_polygon()
    # Sometimes the polygons are invalid (usually because of self-intersection), in which case we return 0.0. We should have a closer look why this happens, but for now it seems to be a rare occurence and this is a quick fix that doesn't seem to affect the results much.
    if not self_poly.is_valid or not that_poly.is_valid:
        return 0.0
    if self_poly.intersects(that_poly):
        inter_poly = self_poly.intersection(that_poly)
        return inter_poly.area / self_poly.area
    else:
        return 0.0
