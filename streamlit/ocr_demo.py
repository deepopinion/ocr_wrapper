from __future__ import annotations


from ocr_wrapper import GoogleOCR, AzureOCR, GoogleAzureOCR, draw_bboxes
from ocr_wrapper.compat import bboxs2dicts
from pdf2image import convert_from_bytes
from io import BytesIO

import streamlit as st
from streamlit_profiler import Profiler

import tempfile
from PIL import Image

import time

# Set layout to wide
st.set_page_config(layout="wide")


def whiten_image(img: Image.Image, amount: float) -> Image.Image:
    """Makes an image brighter"""
    return Image.blend(img, Image.new("RGB", img.size, (255, 255, 255)), amount)


def resize_min(img: Image.Image, min_size: int) -> Image.Image:
    """Resizes an image so that the smallest dimension is at least min_size"""
    if min(img.size) > min_size:
        return img
    w, h = img.size
    if w < h:
        return img.resize((min_size, int(h * min_size / w)))
    else:
        return img.resize((int(w * min_size / h), min_size))


# Settings etc.
st.markdown("#### General Settings")
allow_big_images = st.checkbox("Allow big images (images will be displayed at original resolution)", value=False)
do_profiling = st.checkbox("Run Profiler", value=False)

st.divider()

st.markdown("#### OCR Settings")
ocr_engine = st.selectbox("Select OCR engine", ["Google", "Azure", "GoogleAzure"])
assert ocr_engine is not None
ocr_engine_class = {"Google": GoogleOCR, "Azure": AzureOCR, "GoogleAzure": GoogleAzureOCR}[ocr_engine]

use_ocr_cache = st.checkbox("Use OCR Cache", value=False)
auto_rotate = st.checkbox("Auto rotate image", value=True)
tilt_correction = st.checkbox("Tilt correction", value=True)
ocr_samples = st.number_input("Number of OCR samples", min_value=1, max_value=10, value=2)
max_size = st.slider("Max size of small side of image", min_value=256, max_value=4096, value=1024, step=64)
dpi = st.slider("DPI", min_value=50, max_value=1000, value=200, step=50)

st.divider()

st.markdown("#### Display Settings")
output_order = st.checkbox("Output OCR box order")
output_text_in_image = st.checkbox("Output OCR text in image")
output_ocr_text = st.checkbox("Output OCR text at the end")

show_confidence = st.checkbox("Show confidence (low confidence is a darker blue)")
confidence_text = st.checkbox("Show confidence as text")
confidence_threshold = st.slider(
    "Confidence threshold below which to filter boxes", min_value=0.0, max_value=1.0, value=0.0, step=0.01
)


font_size = st.slider("Font size", min_value=1, max_value=20, value=10, step=1)

st.divider()

uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    filelike = BytesIO(bytes_data)
    if uploaded_file.name.lower().endswith("pdf"):
        with tempfile.TemporaryDirectory() as tmppath:
            pages = convert_from_bytes(filelike.read(), dpi=dpi, size=max_size)
            print(pages[0].size)
    else:
        pages = [Image.open(filelike)]

    ocr = ocr_engine_class(
        ocr_samples=ocr_samples,
        cache_file="googlecache.gcache" if use_ocr_cache else None,
        auto_rotate=auto_rotate,
        correct_tilt=tilt_correction,
        max_size=max_size,
        verbose=True,
    )

    col1, col2 = st.columns(2)
    with col1:
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

            # Filter bboxes by confidence
            bboxes = [bbox for bbox in bboxes if bbox["confidence"] > confidence_threshold]

            if "rotated_image" in extras:
                page = extras["rotated_image"]
                if "tilt_angle" in extras:
                    st.write("Detected tilt angle:", extras["tilt_angle"])

            if output_order:
                texts = [str(i) for i in list(range(len(bboxes)))]
            else:
                texts = ["" for _ in bboxes]

            if output_text_in_image:
                texts = [t + " " + b["text"] for t, b in zip(texts, bboxes)]

            if confidence_text:
                texts = [t + " " + str(round(b["confidence"], 2)) for t, b in zip(texts, bboxes)]

            if show_confidence:
                # Normalize confidence to be between 0 and 1
                if len(bboxes) == 0:
                    cmin, cmax = 0.0, 1.0
                else:
                    cmin, cmax = min(bbox["confidence"] for bbox in bboxes), max(bbox["confidence"] for bbox in bboxes)
                fill_opacities = [1 - bbox["confidence"] for bbox in bboxes]
            else:
                fill_opacities = 0.2

            img = resize_min(page, 2048)

            img = draw_bboxes(
                img=img,
                bboxes=[bbox["bbox"] for bbox in bboxes],
                strokewidths=0,  # Could also be a list for each bbox
                colors="darkgreen",
                fill_colors="blue",
                fill_opacities=fill_opacities,
                texts=texts,
                fontsize=font_size,
            )
            if allow_big_images:
                st.image(img, width=img.size[0])
            else:
                st.image(img)

            if output_ocr_text:
                st.write("OCR Text:", " ".join([bbox["text"] for bbox in bboxes]))
            if "img_samples" in extras:
                st.markdown("### Image samples")
                for img_sample in extras["img_samples"]:
                    if allow_big_images:
                        st.image(img_sample, width=img_sample.size[0])
                    else:
                        st.image(img_sample)

    with col2:
        for page in pages:
            # Start time measurement
            start = time.time()

            second_ocr = GoogleOCR(
                ocr_samples=ocr_samples,
                cache_file="googlecache.gcache" if use_ocr_cache else None,
                auto_rotate=auto_rotate,
                correct_tilt=tilt_correction,
                max_size=max_size,
                verbose=True,
            )

            if do_profiling:
                with Profiler():
                    bboxes, extras = second_ocr.ocr(page, return_extra=True)
            else:
                bboxes, extras = second_ocr.ocr(page, return_extra=True)

            # End time measurement and print time (we only measure the OCR time, not the image loading time etc.)
            end = time.time()
            st.write("Time taken for OCR: ", end - start, "seconds")

            bboxes = bboxs2dicts(bboxes, extras["confidences"][0])

            # Filter bboxes by confidence
            bboxes = [bbox for bbox in bboxes if bbox["confidence"] > confidence_threshold]

            if "rotated_image" in extras:
                page = extras["rotated_image"]
                if "tilt_angle" in extras:
                    st.write("Detected tilt angle:", extras["tilt_angle"])

            if output_order:
                texts = [str(i) for i in list(range(len(bboxes)))]
            else:
                texts = ["" for _ in bboxes]

            if output_text_in_image:
                texts = [t + " " + b["text"] for t, b in zip(texts, bboxes)]

            if confidence_text:
                texts = [t + " " + str(round(b["confidence"], 2)) for t, b in zip(texts, bboxes)]

            if show_confidence:
                # Normalize confidence to be between 0 and 1
                if len(bboxes) == 0:
                    cmin, cmax = 0.0, 1.0
                else:
                    cmin, cmax = min(bbox["confidence"] for bbox in bboxes), max(bbox["confidence"] for bbox in bboxes)
                fill_opacities = [1 - bbox["confidence"] for bbox in bboxes]
            else:
                fill_opacities = 0.2

            img = resize_min(page, 2048)

            img = draw_bboxes(
                img=img,
                bboxes=[bbox["bbox"] for bbox in bboxes],
                strokewidths=0,  # Could also be a list for each bbox
                colors="darkgreen",
                fill_colors="blue",
                fill_opacities=fill_opacities,
                texts=texts,
                fontsize=font_size,
            )
            if allow_big_images:
                st.image(img, width=img.size[0])
            else:
                st.image(img)

            if output_ocr_text:
                st.write("OCR Text:", " ".join([bbox["text"] for bbox in bboxes]))
