# UCN Changelog
The version numbers are according to [Semantic Versioning](http://semver.org/).
(Versions prior to v1.0.0 might not strictly follow semantic versioning)

## Next Release
### Added

### Changed

### Fixed

### Removed

## Release v0.0.10  (2023-11-02)
### Added
- Adds an environment variable `OCR_WRAPPER_CACHE_FILE` to specify an ocr cache file globally
### Changed
- Changed GoogleOCR to use WebP instead of PNG to transfer images to the cloud (reduces amount of transferred data by ~ 1/2)
### Fixed
- Fixed a bug that prevented Google OCR from working with one sample

## Release v0.0.9  (2023-09-07)
### Fixed
- Fixed a problem whene trying to use multi-pass with OCR engines that don't support it yet. Now the system will return a warning message and use the single-pass option instead. (Currently only GoogleOCR is supported for multi-pass)
- A rare bug where self-intersecting bounding boxes cause the OCR system to crash when using multi-pass OCR

## Release v0.0.8  (2023-05-23)
### Fixed
- Adds forced conversion to RGB in pillow before sending data to OpenCV to fix a possible bug in Studio

## Release v0.0.7  (2023-05-18)
### Added
- Added option to define a text brightness in `draw_bboxes` to make the text more readable
- Added confidence estimation to GoogleOCR
- Added multi-pass OCR and set the default to 2 to improve OCR reliability

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
