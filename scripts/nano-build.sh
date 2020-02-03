#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."

cd ${DIR}

pip3 install --disable-pip-version-check -U -r requirements.txt && \
    mkdir -p /tmp/onnx-trt && cd /tmp/onnx-trt && rm -fr onnx-tensorrt &&\
    git clone -n https://github.com/onnx/onnx-tensorrt.git && cd onnx-tensorrt && git checkout 8716c9b && git submodule update --init --recursive && patch -f < ${DIR}/scripts/docker/onnx-trt.patch && \
    mkdir build && cd build && \
    cmake .. -DProtobuf_DEBUG=ON -DProtobuf_USE_STATIC_LIBS=ON \
	  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc \
	  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/tmp/onnx-trt/install -DGPU_ARCHS="53" && make -j && make install && \
    mv -f /tmp/onnx-trt/install/include/* /usr/include/aarch64-linux-gnu/ \
    && mv -f /tmp/onnx-trt/install/lib/libnvonnx* /usr/lib/aarch64-linux-gnu/ && ldconfig && \
    cd ${DIR}/nemo && pip3 install -e . && cd ../collections/nemo_asr && pip3 install -e .

# TODO cd ../nemo_nlp && pip install -e . && cd ../nemo_simple_gan && pip install -e . && cd ../nemo_tts && pip install -e .
# printf "#!/bin/bash\njupyter lab --no-browser --allow-root --ip=0.0.0.0" >> start-jupyter.sh && chmod +x start-jupyter.sh
