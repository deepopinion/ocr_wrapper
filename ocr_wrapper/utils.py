from PIL import Image
from hashlib import sha256


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
