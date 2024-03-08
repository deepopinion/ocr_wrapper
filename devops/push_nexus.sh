#!/bin/bash

echo "Push to Sonatype Nexus"

wheel=$(find dist -name "*.whl")
twine upload $wheel
