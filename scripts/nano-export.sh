#!/bin/bash

DIR=$(cd $(dirname $0); pwd)

PATH=/usr/src/tensorrt/bin:/usr/local/cuda/bin:${PATH}

cd ${DIR}/..

python3 scripts/export_jasper_onnx_to_trt.py \
	--max-seq-len 768 --seq-len 256 --max-batch-size 1 --batch-size 1 --workspace 512 encoder.onnx encoder.plan decoder.onnx decoder.plan

mkdir -p ../trtis-asr/pbr/model_repo/jasper-trt-d/1
cp decoder.plan ../trtis-asr/pbr/model_repo/jasper-trt-d/1/decoder251_53.plan
mkdir -p ../trtis-asr/pbr/model_repo/jasper-trt-e/1
cp encoder.plan ../trtis-asr/pbr/model_repo/jasper-trt-e/1/encoder251_53.plan
