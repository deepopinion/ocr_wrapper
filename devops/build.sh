#!/bin/bash
echo "Setup build system"
pip install --upgrade setuptools wheel
echo "Build package"
python setup.py bdist_wheel
