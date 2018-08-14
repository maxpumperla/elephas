FROM gw000/keras:2.1.3-py3-tf-gpu
MAINTAINER gw0 [http://gw.tnode.com/] <gw.2017@ena.one>

# install py3-tf-cpu/gpu (Python 3, TensorFlow, CPU/GPU)
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # install python 3
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-virtualenv \
    pkg-config \
    # requirements for numpy
    libopenblas-base \
    python3-numpy \
    python3-scipy \
    # requirements for keras
    python3-h5py \
    python3-yaml \
    python3-pydot \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

ARG TENSORFLOW_VERSION=1.10.0
ARG TENSORFLOW_DEVICE=gpu
ARG TENSORFLOW_APPEND=_gpu
RUN pip3 --no-cache-dir install https://storage.googleapis.com/tensorflow/linux/${TENSORFLOW_DEVICE}/tensorflow${TENSORFLOW_APPEND}-${TENSORFLOW_VERSION}-cp35-cp35m-linux_x86_64.whl

ARG KERAS_VERSION=2.1.3
ENV KERAS_BACKEND=tensorflow
RUN pip3 --no-cache-dir install git+https://github.com/fchollet/keras.git@${KERAS_VERSION}

# install additional debian packages
RUN apt-get update -qq \
 && apt-get install --no-install-recommends -y \
    # system tools
    less \
    procps \
    vim-tiny \
    # build dependencies
    build-essential \
    libffi-dev \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*


RUN mkdir -p app
WORKDIR /app
COPY ./requirements.txt /app

# Install requirements
RUN pip3 install -r ./requirements.txt
RUN pip3 install git+https://github.com/hyperopt/hyperopt.git