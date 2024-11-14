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

Use the following dependency groups to install the required packages for the OCR solutions you want to use:

- Azure: `ocr_wrapper[azure]`
- Google: `ocr_wrapper[google]`
- GoogleAzure: `ocr_wrapper[googleazure]` - this also supports checkbox detection
- AWS: `ocr_wrapper[aws]`
- EasyOCR: `ocr_wrapper[easy]`
- PaddleOCR: `ocr_wrapper[paddle]`

Additionally, you can decide whether to use torch or not (faster for large files). As a fallback, numpy is used. If you wanna disable torch 
install it via `ocr_wrapper[torch]`. Otherwise, ensure that you set the env variable `OCR_WRAPPER_NO_TORCH`.

Depending on your operating system you might also have to install the Noto Sans font package (e.g. `noto-sans` in Ubuntu)

## Usage
Different OCR solutions are provided via the classes `GoogleOCR`, `AwsOCR`, `AzureOCR`, `EasyOCR`, and `PaddleOCR`.

All solutions have built in caching support, which can be activated by supplying a `cache_file` argument in the constructor. If a page has already been OCRed, the result from the cache will be used. Once an OCR class has been instantiated, an image can be OCRed with the `.ocr` Method which accepts an image in PIL format.

The result will be a list of `BBox` instances. Each `BBox` contains the coordinates of the four points of the bounding box and the recognized `text`. It can also contain an optional `label`, which is not used for OCR. `BBox` aslo supports many additional conversion functions, detecting overlap of bounding boxes, etc.

To easily visualize bounding boxes, the library also offers the method `draw_bboxes`.

### Autoselect
The function `autoselect_ocr_engine()` can be used to automatically return the class for the needed OCR engine, using the `OCR_PROVIDER` environment variable. `google`, `azure`, `aws`, `easy`, and `paddle` are valid settings. If no provider is explicitly set, Google OCR is chosen by default. 
In case an invalid OCR provider is specified, an `InvalidOcrProviderException` will be raised.

### GoogleOCR and GoogleDocumentOcrCheckboxDetector
The credentials will be obtained from one of the following:
- The environment variable `GOOGLE_APPLICATION_CREDENTIALS`
- A credentials file `~/.config/gcloud/credentials.json`
- A credentials file `/credentials.json`

### AzureOCR
Credentials etc. for Azure OCR will be obtained from one of the following (in this order):
- From the `endpoint` and `key` arguments when creating AzureOCR
- From the environment variables `AZURE_OCR_ENDPOINT` and `AZURE_OCR_KEY`
- From the credentials file `~/.config/azure/ocr_credentials.json` that contains the keys `endpoint` and `key`

### GoogleAzureOCR
Credentials for GoogleOCR as well as AzureOCR have to be set

## Special Functionality
### QR and Barcode Detection and Decoding
All OCR wrappers support detection and decoding of QR codes as well as different forms of barcodes. Internally, the [zbar](https://zbar.sourceforge.net/) library is used.

The QR/bar-codes will be added as bounding boxes, and the contained information is returned as the OCR text. The text is formatted in the form `TYPE[[DATA]]` (e.g. `QRCODE[[Encoded Information]]`, `CODE39[[1234567890]]`, ...). All valid types can be found in the `pyzbar.pyzbar.ZBarSymbol` enum.

This has to be activated with the argument `add_qr_barcodes=True` when creating the wrapper (default is `False`). In these cases, the `pyzbar` Python library, as well as the zbar system library have to be installed.
