#!/bin/bash

D_FILE=Dockerfile
D_CONT=nemo:latest
USERID=$(id -u)
CONTAINER=nvcr.io/nvidia/nemo:v0.10

cat > $D_FILE <<EOF
FROM $CONTAINER
USER root
RUN source activate base \ 
    && conda install -y -c conda-forge flask pylint flake8 autopep8
#
# required set up
#
RUN source activate base \ 
    && mkdir /.local /.jupyter /.config /.cupy  \
    && chmod 777 /.local /.jupyter /.config /.cupy

RUN apt-get update
RUN apt-get install -y locales-all build-essential vim
ENV LC_ALL="en_US.utf8"
EXPOSE 8888
EXPOSE 8787
EXPOSE 8786

ARG USERNAME=nemo
ARG USER_UID=$USERID
ARG USER_GID=\$USER_UID

# Create the user
RUN groupadd --gid \$USER_GID \$USERNAME \
    && useradd --uid \$USER_UID --gid \$USER_GID -m \$USERNAME \
    && apt-get update \
    && apt-get install -y sudo \
    && echo \$USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/\$USERNAME \
    && chmod 0440 /etc/sudoers.d/\$USERNAME

USER \$USERNAME

WORKDIR /workspace
ENTRYPOINT /bin/bash -c 'source activate base; /bin/bash'
EOF

docker build -f $D_FILE -t $D_CONT .
