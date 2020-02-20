#!/bin/bash

DIR=$(cd $(dirname $0); pwd)

PATH=/usr/src/tensorrt/bin:/usr/local/cuda/bin:${PATH}

cd ${DIR}/..

#python3 scripts/export_jasper_to_onnx.py --config quartznet15x5/quartznet15x5.yaml \
#       --nn_encoder quartznet15x5/JasperEncoder-STEP-247400.pt --nn_decoder quartznet15x5/JasperDecoderForCTC-STEP-2474 --onnx_encoder=encoder1.onnx --onnx_decoder=decoder1.onnx || exit 1

python3 scripts/export_jasper_onnx_to_trt.py \
	--max-seq-len 256 --seq-len 256 --max-batch-size 1 --batch-size 1 --workspace 256 encoder.onnx encoder.plan decoder.onnx decoder.plan

mkdir -p ../trtis-asr/pbr/model_repo/jasper-trt-d/1
cp decoder.plan ../trtis-asr/pbr/model_repo/jasper-trt-d/1/decoder_53.plan
mkdir -p ../trtis-asr/pbr/model_repo/jasper-trt-e/1
cp encoder.plan ../trtis-asr/pbr/model_repo/jasper-trt-e/1/encoder_53.plan
