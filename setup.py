from setuptools import find_packages, setup

setup(
    name="ocr_wrapper",
    version="0.0.5",
    packages=find_packages(),
    include_package_data=True,
    license="MIT",
    url="https://github.com/deepopinion/ocr_wrapper",
    author="DeepOpinion",
    author_email="hello@deepopinion.ai",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    install_requires=["Pillow", "Shapely"],
    zip_safe=False,
)
