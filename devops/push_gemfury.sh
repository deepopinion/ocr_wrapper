#!/bin/bash

echo "Push to Gemfury"

wheel=$(find dist -name "*.whl")
curl -F "package=@$wheel" "https://$PIP_PUSH@push.fury.io/deepopinion"