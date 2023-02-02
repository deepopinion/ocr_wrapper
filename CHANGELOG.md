# UCN Changelog
The version numbers are according to [Semantic Versioning](http://semver.org/).
(Versions prior to v1.0.0 might not strictly follow semantic versioning)

## Release v0.0.5  (2022-??-??)
### Added
- Added an automatic resize option for documents. We found out, that
  using a max_size of 1024 for the documents improves the OCR quality.
- Adds document rotation information to the GoogleOCR response.
### Changed

### Fixed

### Removed


## Release v0.0.4  (2022-10-06)
### Fixed
- Fixed problem with slight inconsistencies in the format coming from LabelStudio
- Fixed problem with values sometimes slightly above 1 when normalizing a BBox
