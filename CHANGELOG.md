# ocr_wrapper Changelog
The version numbers are according to [Semantic Versioning](http://semver.org/).
(Versions prior to v1.0.0 might not strictly follow semantic versioning)

## Release v1.1.0  (2023-??-??)
### Added
- Added the output of confidence scores to GoogleOCR
- Added multiple OCR passes to improve OCR reliability
- Adds an environment variable `OCR_WRAPPER_CACHE_FILE` to specify an ocr cache file globally

### Changed
- Changed GoogleOCR to use WebP instead of PNG to transfer images to the cloud (reduces amount of transferred data by ~ 1/2)
### Fixed
- Adds forced conversion to RGB in pillow before sending data to OpenCV to fix a possible bug in Studio
- Fixes a rare bug where self-intersecting bounding boxes caused the OCR system to crash when using multi-pass OCR
- Fixed a problem whene trying to use multi-pass with OCR engines that don't support it yet. Now the system will return a warning message and use the single-pass option instead. (Currently only GoogleOCR is supported for multi-pass)
- Brings AzureOCR to a fully working state again. It has full functionality, with the exception for multi-pass OCR (which might not be needed for Azure though)
### Removed


## Release v1.0.0  (2023-05-03)
### Added
- Added option to define a text brightness in `draw_bboxes` to make the text more readable
- Streamlit demo to quickly test OCR solutions
### Changed
- Major rewrite of the `BBox` class to improve data consistency
- BBoxes now save the image size of the image they are on, to enable conversion from absolute to relative coordinates without additional data
- Instead of a list of `BBox`, `OcrWrapper.ocr()` now returns a list of dicts with keys  "bbox" and "text", which contain the `BBox` instance and its text respectively
### Removed
- Text and label are no longer part of the `BBox` object. BBoxes are intended to purely represent a quadrilateral on an image. This e.g. enables their usage in computer vision
- A lot of specialized getters and constructors have been removed. The principal formats offered by `BBox` are any combination of:
  - 8-tuple (TLx, TLy, TRx, ...) quadrilateral or 4-tuple (x1, y1, x2, y2) axis-aligned bounds
  - Relative (0-1) or absolute (px) coordinates

## Release v0.0.6  (2023-02-20)
### Added
- Adds option to fill bboxes with a transparent color using
  `draw_bboxes`

### Fixed
- Fixes bug when OCRing images with no text and using `auto_rotate`
  (leading to a division by zero)

## Release v0.0.5  (2023-02-16)
### Added
- Added an automatic resize option for documents. Turned off by
default, but can improve OCR results in some cases
- Added document rotation information to the GoogleOCR response.
- Added the option to automatically rotate the bounding boxes and the
image to the correct orientation if `auto_rotate == True` for
GoogleOCR
- Added the option to specify a specific endpoint for GoogleOCR and
fixed the default to europe
- Added method to generate a `BBox` from x,y coordinates of the upper
left and lower right corner and a width and height
- Added methods to rotate a `BBox` in 90 degree steps

## Release v0.0.4  (2022-10-06)
### Fixed
- Fixed problem with slight inconsistencies in the format coming from LabelStudio
- Fixed problem with values sometimes slightly above 1 when normalizing a BBox
