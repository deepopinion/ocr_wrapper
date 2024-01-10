import json
import os
import io
from timeit import default_timer as timer
from concurrent.futures import ThreadPoolExecutor, as_completed

from PIL import Image

from ocr_wrapper import GoogleOCR
import yappi


def main():
    img_names = os.listdir("imgs")[:1]
    img_files = [Image.open(f"imgs/{f}") for f in img_names]

    ocr_client = GoogleOCR(ocr_samples=2)
    times = []

    global_start = timer()
    results= []
    with ThreadPoolExecutor(max_workers=64) as executor:
        start = timer()
        futures = [executor.submit(ocr_client.ocr, img, denoise=False) for img in img_files]
        for future in as_completed(futures):
            res = future.result()
            times.append(timer() - start)
            results.append(res)
    global_end = timer()

    results = dict(zip(img_names, times))
    results["total_time"] = global_end - global_start
    with open("nocache-2sample-fix1.json", "w") as f:
        json.dump(results, f, indent=4)



if __name__ == "__main__":
    yappi.set_clock_type("wall")
    yappi.start()
    main()
    threads = yappi.get_thread_stats()
    for thread in threads:
        output = io.StringIO()
        yappi.get_func_stats(ctx_id=thread.id).sort("ttot").print_all(
            out=output, columns={0: ("name", 100), 1: ("ncall", 5), 2: ("tsub", 8), 3: ("ttot", 8), 4: ("tavg", 8)}
        )

        with open(f"yappi-{thread.name}-{thread.id}.txt", "w") as f:
            f.write(output.getvalue())
