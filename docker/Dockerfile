FROM nvcr.io/nvidia/pytorch:21.10-py3

ENV DEBIAN_FRONTEND=noninteractive
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ENV FORCE_CUDA="1"


#RUN pip config set global.index-url https://mirrors.ustc.edu.cn/pypi/web/simple
RUN pip install cython
RUN pip install scikit-learn pandas
RUN pip install -U numpy
RUN MMCV_WITH_OPS=1 pip install mmcv-full==1.3.17
RUN pip install albumentations
RUN pip install PyTurboJPEG
RUN pip install mmdet==2.19.0
RUN pip install mmsegmentation==0.20.2
RUN pip install ujson
RUN pip install 'git+https://github.com/facebookresearch/detectron2.git'

ADD ./simota_cuda_ops /opt/simota_cuda_ops
RUN cd /opt/simota_cuda_ops \
    && python setup.py install

WORKDIR /workspace
