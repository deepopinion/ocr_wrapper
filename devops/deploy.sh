#!/bin/bash

rm dist/*.whl
sh devops/build.sh
sh devops/push_gemfury.sh
