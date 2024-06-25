#!/bin/bash
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

# Get directory this script is in
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
OS=$(uname -s)

if [ "$OS" = "Darwin" ]; then
    DIR_OS="MacOS"
else
    DIR_OS="Linux"
fi

if [[ "$*" == *"--ios"* ]]; then
    DIR_OS="iOS"
elif [[ "$*" == *"--android"* ]]; then
    DIR_OS="Android"
fi

export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH

python3 $DIR/tools/ci_build/build.py --build_dir $DIR/build/$DIR_OS "$@"
