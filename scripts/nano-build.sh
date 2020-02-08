#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )/.."

cd ${DIR}

export INCLUDE=/usr/local/cuda-10.0/targets/aarch64-linux/include
# ;/usr/local/lib/python3.6/dist-packages/torch/include
# export LIB=/usr/local/cuda-10.0/targets/aarch64-linux/lib;/usr/local/lib/python3.6/dist-packages/torch/lib
export VERBOSE=ON
export LLVM_CONFIG=/usr/lib/llvm-7/bin/llvm-config

pip3 install --disable-pip-version-check -U -r requirements.txt

cd ${DIR}/nemo && pip3 install -e . && cd ../collections/nemo_asr && pip3 install -e .

# TODO cd ../nemo_nlp && pip install -e . && cd ../nemo_simple_gan && pip install -e . && cd ../nemo_tts && pip install -e .
# printf "#!/bin/bash\njupyter lab --no-browser --allow-root --ip=0.0.0.0" >> start-jupyter.sh && chmod +x start-jupyter.sh
