# ocr_wrapper Changelog
The version numbers are according to [Semantic Versioning](http://semver.org/).
(Versions prior to v1.0.0 might not strictly follow semantic versioning)

## Release v0.0.x ()
### Added

### Fixed

### Changed

### Removed


## Release v0.0.22 (2024-04-26)
### Added
- Fixing library version.

## Release v0.0.21 (2024-04-26)
### Added
- Started publishing the library to Google Artifact Registry.

## Release v0.0.20 (2024-03-26)
### Added
- Added a heuristic for splitting date ranges (dd.mm.yyyy-dd.mm.yyyy or dd/mm/yyyy-dd/mm/yyyy) that were merged into a single bounding box by GoogleOCR (and thus by GoogleAzureOCR as well), into three distinct bounding boxes.
### Changed
- Improved warning messages for set arguments of GoogleAzureOCR to be less confusing
### Removed
- No longer publishing to Sonatype Nexus

## Release v0.0.19 (2024-03-08)
### Added
- Publishing library to Sonatype Nexus.
### Fixed
- Fixed a problem where not all ocr wrappers accepted the `add_checkboxes` argument. (Checkboxes are still only supported for GoogleAzureOCR, but we prevent execution errors and only a warning will be issued)

## Release v0.0.18 (2024-03-07)
### Added
- Added checkbox detection with `GoogleDocumentOcrCheckboxDetector` and integrated into `GoogleAzureOCR`

## Release v0.0.17 (2024-03-01)
### Fixed
- Made OCR wrappers thread safe (previously it was only thread safe if every thread used its own instance of the wrapper class)
- Fixed a bug where cache files did not always work on all systems

## Release v0.0.16 (2024-02-06)
### Fixed
- Fixed a bug with concurrent OCR of multiple pages
- Fixed a problem when going above the rate limit of Azure

## Relase v0.0.15 (2024-02-01)
### Added
- Added new OCR wrapper that combines Google OCR and Azure OCR to compensate shortcomings of Google OCR
- Added new method `multi_img_ocr` to all OCR wrappers to be able to process multiple images at the same time
- Adds support for the environment variable `OCR_PROVIDER_MAPPING` that can give a list of OCR provider replacements. Eg. `"google=googleazure"` means that all places where `google` would be instantiated, `googleazure` will be used instead. This can be used to replace the default Google OCR with the new combined OCR wrapper.

## Release v0.0.14 (2024-01-12)
### Fixed
- Sets max size of longer side for GoogleOCR to 1024 (which it should have been already)

## Release v0.0.13 (2023-11-16)
### Changed
- Improved tilt correction to also support documents like passports

## Release v0.0.12 (2023-11-10)
### Added
- Added a first version of tilt correction. **Note:** `torch` and `torchvision` are new required packages because of this.
### Fixed
- Fixed a bug introduced by a missing required positional argument in `autoselect_ocr_engine`

## Release v0.0.11 (2023-11-09)
### Added
- Extended the possibilities on how to provide endpoint and key for Azure OCR in addition to the `credentials.json`. Have a look at the README.md for details.
### Fixed
- Brings AzureOCR to a fully working state again. It has full functionality, with the exception for multi-pass OCR (which might not be needed for Azure though)

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
