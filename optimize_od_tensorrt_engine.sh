#!/bin/bash

# Optimize for Object Detection.
# e.g. ./optimize_od_tensorrt_engine.sh yolox_x_body_head_hand_post_0102_0.5533_1x3x384x640.onnx

input_file="$1"

if [[ $input_file =~ 1x[0-9]+x([0-9]+)x([0-9]+) ]]; then
    H=${BASH_REMATCH[1]}
    W=${BASH_REMATCH[2]}
else
    echo "Error: Incorrect file name."
    exit 1
fi

start_time=$(date +%s)

sit4onnx -if "${input_file}" -fs 1 3 ${H} ${W}

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "elapsed_time: $elapsed sec"
