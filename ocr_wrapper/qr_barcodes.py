from pyzbar.pyzbar import decode, Decoded
from ocr_wrapper.bbox import BBox
from PIL import Image, ImageFilter

import time


def _decoded_to_coordinate_list(decoded: Decoded) -> list[float]:
    """Takes a Decoded object and returns a list of coordinates in the format [TLx, TLy, TRx, TRy, BRx, BRy, BLx, BLy]"""
    left, top, width, height = decoded.rect.left, decoded.rect.top, decoded.rect.width, decoded.rect.height
    return [left, top, left + width, top, left + width, top + height, left, top + height]


def _decoded_to_ocr_text(decoded: Decoded) -> str:
    """Takes a Decoded object and returns a text that can be used instead of the usual OCR text

    The format looks as follows:
    ```TYPE[[DATA]]```

    The regex (\w+)\[\[([^\]]+)\]\] can be used to extract this information again, the two capturing groups
    will contain the type and the data respectively

    Valid types can be found in the pyzbar.pyzbar.ZBarSymbol enum
    """
    return f"{decoded.type}[[{decoded.data.decode('utf-8')}]]"


def _decoded_to_bbox(decoded: Decoded) -> BBox:
    """Takes a Decoded object and returns a corresponding BBox object

    - Coordinates will be in pixel values
    - Text will be in the format TYPE[[DATA]] where TYPE is the type of the barcode, selected from the pyzbar.pyzbar.ZBarSymbol enum
    - Label will be None
    """
    coordinates = _decoded_to_coordinate_list(decoded)
    text = _decoded_to_ocr_text(decoded)
    bbox = BBox.from_float_list(coordinates, text=text, in_pixels=True)
    return bbox


def _detect_raw_qr_barcodes(image: Image.Image) -> list[Decoded]:
    """Detects barcodes in an image and returns a list of Decoded objects"""
    # Make the image binary black and white to improve detection
    image = image.convert("L")
    image = image.point(lambda x: 0 if x < 128 else 255, "1")

    decoded_objects = decode(image)
    return decoded_objects


def detect_qr_barcodes(image: Image.Image) -> list[BBox]:
    """Detects barcodes in an image and returns a list of BBox objects"""
    decoded_objects = _detect_raw_qr_barcodes(image)
    bboxes = [_decoded_to_bbox(decoded) for decoded in decoded_objects]
    # Normalize the BBoxes so they are not in pixel coordinates anymore
    width, height = image.size
    bboxes = [bbox.to_normalized(width, height) for bbox in bboxes]
    print(bboxes)
    return bboxes
