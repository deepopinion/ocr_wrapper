# OCR Wrapper

This is a simple wrapper for word-segmented OCR. It provides a standardized interface to:

* PaddleOCR (local)
* EasyOCR (local)
* Azure Cognitive Skills
* Google Cloud
* AWS

For a usage example, see `tryme.ipynb`.

## Requirements
In addition to the automatically installed requirements, you also need to install the packages needed for the individual OCR solutions. These are not automatically installed, since we don't want to force dependencies which are not really needed if you only use one specific OCR solution.

- Google: `google-cloud-vision`
- AWS: `boto3`
- Azure: `azure-cognitiveservices-vision-computervision`
- PaddleOCR: `paddleocr`
- EasyOCR: `easyocr`

Depending on your operating system you might also have to install the Noto Sans font package (e.g. `noto-sans` in Ubuntu)

## Usage
Different OCR solutions are provided via the classes `GoogleOCR`, `AwsOCR`, `AzureOCR`, `EasyOCR`, and `PaddleOCR`.

All solutions have built in caching support, which can be activated by supplying a `cache_file` argument in the constructor. If a page has already been OCRed, the result from the cache will be used. Once an OCR class has been instantiated, an image can be OCRed with the `.ocr` Method which accepts an image in PIL format.

The result will be a list of `BBox` instances. Each `BBox` contains the coordinates of the four points of the bounding box and the recognized `text`. It can also contain an optional `label`, which is not used for OCR. `BBox` aslo supports many additional conversion functions, detecting overlap of bounding boxes, etc.

To easily visualize bounding boxes, the library also offers the method `draw_bboxes`.

### GoogleOCR
The credentials for Google OCR will be obtained from one of the following:
- The environment variable `GOOGLE_APPLICATION_CREDENTIALS`
- A credentials file `~/.config/gcloud/credentials.json`
- A credentials file `/credentials.json`
