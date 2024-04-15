import re
from hashlib import sha256

from PIL import Image


def _get_bytes_hash(_bytes):
    """Returns the sha256 hash in hex form of a bytes object"""
    h = sha256()
    h.update(_bytes)
    img_hash = h.hexdigest()
    return img_hash


def get_img_hash(img: Image.Image) -> str:
    """Returns a hash of the image."""
    hash_str = str(_get_bytes_hash(img.tobytes()))
    return hash_str


def has_arabic_text(s: str) -> bool:
    """Detects if a string contains Arabic text based on character ranges."""

    arabic_ranges = (
        (0x0600, 0x06FF),  # Arabic
        (0x0750, 0x077F),  # Arabic Supplement
        (0x08A0, 0x08FF),  # Arabic Extended-A
        (0xFB50, 0xFDFF),  # Arabic Presentation Forms-A
        (0xFE70, 0xFEFF),  # Arabic Presentation Forms-B
    )

    for char in s:
        if any(start <= ord(char) <= end for start, end in arabic_ranges):
            return True  # Arabic character found
    return False  # No Arabic characters detected


def flip_number_blocks(input_string):
    """Reverses the order of digit blocks in a string while keeping non-digit blocks in place."""
    # Split into parts of digits and non-digits
    parts = re.split(r"(\D+)", input_string)
    # Reverse the order of digit parts
    digit_parts = [part for part in parts if part.isdigit()]
    reversed_digits = digit_parts[::-1]

    # Reconstruction with the digit parts reversed
    result = [reversed_digits.pop() if part.isdigit() else part for part in parts]

    return "".join(result)
