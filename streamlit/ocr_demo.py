from __future__ import annotations


from ocr_wrapper import GoogleOCR, draw_bboxes
from pdf2image import convert_from_bytes
from io import BytesIO

import streamlit as st

import tempfile
from PIL import Image

import time


def whiten_image(img: Image.Image, amount: float) -> Image.Image:
    """Makes an image brighter"""
    return Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), amount)


# Allow uploading of PDFs
uploaded_file = st.file_uploader("Choose a file")
# Allow selection of whether to output ocr box order
auto_rotate = st.checkbox("Auto rotate image")
output_order = st.checkbox("Output OCR box order")
output_text = st.checkbox("Output OCR text")
show_confidence = st.checkbox("Show confidence (low confidence is a darker blue)")
show_bbox_code = st.checkbox("Show bbox code")
ocr_samples: int = st.number_input("Number of OCR samples", min_value=1, max_value=10, value=2)

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    filelike = BytesIO(bytes_data)
    if uploaded_file.name.lower().endswith("pdf"):
        with tempfile.TemporaryDirectory() as tmppath:
            pages = convert_from_bytes(filelike.read())
    else:
        pages = [Image.open(filelike)]

    ocr = GoogleOCR(ocr_samples=ocr_samples, cache_file="googlecache.gcache", auto_rotate=auto_rotate)

    # Start time measurement
    start = time.time()

    for page in pages:
        bboxes, extras = ocr.ocr(page, return_extra=True)

        if auto_rotate:
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

        img = draw_bboxes(
            img=page,
            bboxes=[bbox["bbox"] for bbox in bboxes],
            strokewidths=0,  # Could also be a list for each bbox
            colors="blue",
            fill_colors="blue",
            fill_opacities=fill_opacities,
            texts=texts,
            fontsize=10,
        )
        st.image(img)

        if show_bbox_code:
            st.write(bboxes)

    # End time measurement and print time
    end = time.time()
    st.write("Time taken: ", end - start)
