from setuptools import find_packages, setup

setup(
    packages=find_packages(),
    package_data={"ocr_wrapper": ["pallets.json"]},
    zip_safe=False,
)