from __future__ import annotations


from ocr_wrapper import GoogleOCR, draw_bboxes
from pdf2image import convert_from_bytes
from io import BytesIO

import streamlit as st
import tempfile
from PIL import Image


def whiten_image(img: Image.Image, amount: float) -> Image.Image:
    """Makes an image brighter"""
    return Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), amount)


# Allow uploading of PDFs
uploaded_file = st.file_uploader("Choose a file")
# Allow selection of whether to output ocr box order
output_order = st.checkbox("Output OCR box order")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    filelike = BytesIO(bytes_data)
    if uploaded_file.name.lower().endswith("pdf"):
        with tempfile.TemporaryDirectory() as tmppath:
            pages = convert_from_bytes(filelike.read())
    else:
        pages = [Image.open(filelike)]

    ocr = GoogleOCR(max_size=2048)

    for page in pages:
        bboxes = ocr.ocr(page)
        # st.image(page)

        # page = whiten_image(page, 0.3)

        if output_order:
            texts = [str(i) for i in list(range(len(bboxes)))]
        else:
            texts = ""

        img = draw_bboxes(
            img=page,
            bboxes=bboxes,
            strokewidths=0,  # Could also be a list for each bbox
            colors="blue",
            fill_colors="blue",
            fill_opacities=0.2,
            texts=texts,
            fontsize=10,
        )
        st.image(img)
