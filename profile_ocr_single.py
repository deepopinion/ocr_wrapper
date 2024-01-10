from datetime import datetime
import io
from PIL import Image

from ocr_wrapper import GoogleOCR
import yappi


def main():
    img_file = Image.open("imgs_comparison/passport-hellostudy-sample.jpg")

    ocr_client = GoogleOCR(ocr_samples=2, correct_tilt=True)
    ocr_client.ocr(img_file, denoise=True)


if __name__ == "__main__":
    yappi.set_clock_type("wall")
    yappi.start()
    main()
    threads = yappi.get_thread_stats()
    yappi.stop()
    stats = yappi.get_func_stats()
    stats.save(f"callgrind.out", "CALLGRIND")

    yappi.clear_stats()
