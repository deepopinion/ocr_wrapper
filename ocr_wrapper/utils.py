import re
from hashlib import sha256

from PIL import Image
from opentelemetry.trace import Span


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
    # Split the string into segments of digits and non-digits
    parts = re.split(r"(\D+)", input_string)

    # Filter and reverse the order of the digit-only parts
    digit_parts = [part for part in parts if part.isdigit()]
    reversed_digits = digit_parts[::-1]

    # Reassemble the string with reversed digit blocks and original non-digit separators
    result = []
    digit_index = 0

    for part in parts:
        if part.isdigit():
            result.append(reversed_digits[digit_index])
            digit_index += 1
        else:
            result.append(part)

    return "".join(result)


def resize_image(img: Image.Image, max_size: int) -> Image.Image:
    """Resize the image so the bigger side has max_size pixels, keeping the aspect ratio."""
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


def set_image_attributes(span: Span, image: Image.Image) -> None:
    """
    Assigns specific attributes of an image to a span for tracing purposes.

    Parameters:
    span (Span): The span object to which image attributes will be assigned.
    image (Image.Image): The image from which attributes like size, mode, channel information,
                         and format will be extracted and set on the span.
    """
    span.set_attribute("image_size", image.size)
    span.set_attribute("image_mode", image.mode)
    span.set_attribute("channel_info", image.getbands())
    span.set_attribute("image_format", image.format if image.format else "Unknown")
