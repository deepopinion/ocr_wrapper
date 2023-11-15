# OCR Wrapper

This is a simple wrapper for word-segmented OCR. It provides a standardized interface to:

* PaddleOCR (local)
* EasyOCR (local)
* Azure Cognitive Skills
* Google Cloud
* AWS

Further, it provides a "BBox" class which can be used to represent bounding boxes on documents, and convert
them from/to various formats used by different OCR and machine learning solutions.

For a usage example, see `tryme.ipynb`.

## Requirements
In addition to the automatically installed requirements, you also need to install the packages needed for the individual OCR solutions. These are not automatically installed, since we don't want to force dependencies which are not really needed if you only use one specific OCR solution.

- Google: `google-cloud-vision`
- AWS: `boto3`
- Azure: `azure-cognitiveservices-vision-computervision`
- PaddleOCR: `paddleocr`
- EasyOCR: `easyocr`

You can install all of them at once by selecting the `ocr` optional dependency: `pip install .[ocr]`.

Depending on your operating system you might also have to install the Noto Sans font package (e.g. `noto-sans` in Ubuntu)

## Usage
Different OCR solutions are provided via the classes `GoogleOCR`, `AwsOCR`, `AzureOCR`, `EasyOCR`, and `PaddleOCR`.

All solutions have built in caching support, which can be activated by supplying a `cache_file` argument in the constructor. If a page has already been OCRed, the result from the cache will be used. Once an OCR class has been instantiated, an image can be OCRed with the `.ocr` method, which accepts an image in `PIL.Image.Image` format.

The result will be a list of dicts, each representing one chunk of text (usually, one word). Each dict contains at least the following keys:
- `bbox`: A `ocr_wrapper.BBox` instance that represents the location of the bounding quadrilateral of the detection
- `text`: The text that was detected

Specific OCR engines may add other keys to return additional information about a bounding box.

To easily visualize bounding boxes, the library also offers the method `draw_bboxes`. See `tryme.ipynb` for a minimal code example.

### Autoselect
The function `autoselect_ocr_engine()` can be used to automatically return the class for the needed OCR engine, using the `OCR_PROVIDER` environment variable. `google`, `azure`, `aws`, `easy`, and `paddle` are valid settings. If no provider is explicitly set, Google OCR is chosen by default. 
In case an invalid OCR provider is specified, an `InvalidOcrProviderException` will be raised.

### GoogleOCR
The credentials for Google OCR will be obtained from one of the following sources, in this order:
- The environment variable `GOOGLE_APPLICATION_CREDENTIALS`
- A credentials file `~/.config/gcloud/credentials.json`
- A credentials file `/credentials.json`
