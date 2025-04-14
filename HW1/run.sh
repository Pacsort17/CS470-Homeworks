#!/bin/bash

if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_json> <output_json>"
    exit 1
fi

python3 code/run.py "$1" "$2" 