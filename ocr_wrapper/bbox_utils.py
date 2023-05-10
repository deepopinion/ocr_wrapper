"""
Additional functions for working with BBox objects.
"""
from .bbox import BBox


def bbox_intersection_area_percent(bb1: BBox, bb2: BBox) -> float:
    """Returns the area of the intersection of BBox `bb1` with BBox `bb2` as a proportion of the area of `bb1`.

    i.e. max is 1.0, min is 0.0
    """
    self_poly = bb1.get_shapely_polygon()
    that_poly = bb2.get_shapely_polygon()
    inter_poly = self_poly.intersection(that_poly)
    return inter_poly.area / self_poly.area
