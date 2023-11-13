from __future__ import annotations


from ocr_wrapper import GoogleOCR, AzureOCR, draw_bboxes
from ocr_wrapper.compat import bboxs2dicts
from pdf2image import convert_from_bytes
from io import BytesIO

import streamlit as st
from streamlit_profiler import Profiler

import tempfile
from PIL import Image

import time


def whiten_image(img: Image.Image, amount: float) -> Image.Image:
    """Makes an image brighter"""
    return Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), amount)


def resize_min(img: Image.Image, min_size: int) -> Image.Image:
    """Resizes an image so that the smallest dimension is at least min_size"""
    w, h = img.size
    if w < h:
        return img.resize((min_size, int(h * min_size / w)))
    else:
        return img.resize((int(w * min_size / h), min_size))


# Allow uploading of PDFs
uploaded_file = st.file_uploader("Choose a file")

# Select OCR engine
ocr_engine = st.selectbox("Select OCR engine", ["Google", "Azure"])
ocr_engine_class = {"Google": GoogleOCR, "Azure": AzureOCR}[ocr_engine]

# Allow selection of whether to output ocr box order etc.
do_profiling = st.checkbox("Profile", value=False)
use_ocr_cache = st.checkbox("Use OCR Cache", value=False)
auto_rotate = st.checkbox("Auto rotate image")
output_order = st.checkbox("Output OCR box order")
output_text = st.checkbox("Output OCR text")
show_confidence = st.checkbox("Show confidence (low confidence is a darker blue)")
ocr_samples = st.number_input("Number of OCR samples", min_value=1, max_value=10, value=2)

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    filelike = BytesIO(bytes_data)
    if uploaded_file.name.lower().endswith("pdf"):
        with tempfile.TemporaryDirectory() as tmppath:
            pages = convert_from_bytes(filelike.read())
    else:
        pages = [Image.open(filelike)]

    ocr = ocr_engine_class(
        ocr_samples=ocr_samples,
        cache_file="googlecache.gcache" if use_ocr_cache else None,
        auto_rotate=auto_rotate,
        verbose=True,
    )

    for page in pages:
        # Start time measurement
        start = time.time()

        if do_profiling:
            with Profiler():
                bboxes, extras = ocr.ocr(page, return_extra=True)
        else:
            bboxes, extras = ocr.ocr(page, return_extra=True)

        # End time measurement and print time (we only measure the OCR time, not the image loading time etc.)
        end = time.time()
        st.write("Time taken for OCR: ", end - start, "seconds")

        bboxes = bboxs2dicts(bboxes, extras["confidences"][0])

        if "rotated_image" in extras:
            page = extras["rotated_image"]

        if output_order:
            texts = [str(i) for i in list(range(len(bboxes)))]
        else:
            texts = ["" for _ in bboxes]

        if output_text:
            texts = [t + " " + b["text"] for t, b in zip(texts, bboxes)]

        if show_confidence:
            # Normalize confidence to be between 0 and 1
            cmin, cmax = min(bbox["confidence"] for bbox in bboxes), max(bbox["confidence"] for bbox in bboxes)
            # fill_opacities = [1 - ((bbox["confidence"] - cmin) / (cmax - cmin)) for bbox in bboxes]
            fill_opacities = [1 - bbox["confidence"] for bbox in bboxes]
        else:
            fill_opacities = 0.2

        img = resize_min(page, 2048)

        img = draw_bboxes(
            img=img,
            bboxes=[bbox["bbox"] for bbox in bboxes],
            strokewidths=0,  # Could also be a list for each bbox
            colors="blue",
            fill_colors="blue",
            fill_opacities=fill_opacities,
            texts=texts,
            fontsize=10,
        )
        st.image(img)
        st.markdown(f"Number of OCR boxes: {len(bboxes)}")
        if "img_samples" in extras:
            st.markdown("### Image samples")
            for img_sample in extras["img_samples"]:
                st.image(img_sample)
