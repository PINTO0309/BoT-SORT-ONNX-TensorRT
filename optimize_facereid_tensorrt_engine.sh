#!/bin/bash

# Optimize for up to 100 ReIDs of 100 x 1.
# e.g. ./optimize_facereid_tensorrt_engine.sh face-reidentification-retail-0095_NMx3x128x128_post_feature_only.onnx

input_file="$1"

if [[ $input_file =~ NMx[0-9]+x([0-9]+)x([0-9]+) ]]; then
    H=${BASH_REMATCH[1]}
    W=${BASH_REMATCH[2]}
else
    echo "Error: Incorrect file name."
    exit 1
fi

start_time=$(date +%s)

for i in {1..20}; do
    sit4onnx -if "${input_file}" -fs ${i} 3 ${H} ${W} -fs 1 256
done

end_time=$(date +%s)
elapsed=$(( end_time - start_time ))
echo "elapsed_time: $elapsed sec"
