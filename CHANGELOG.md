# UCN Changelog
The version numbers are according to [Semantic Versioning](http://semver.org/).
(Versions prior to v1.0.0 might not strictly follow semantic versioning)

## Release v0.0.5  (2022-??-??)
### Added

### Changed

### Fixed

### Removed


## Release v0.0.5  (2022-02-16)
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
