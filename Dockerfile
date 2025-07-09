FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /workspace/WinClip

RUN apt-get update && \
    apt-get install -y git python3 python3-pip ffmpeg libsm6 libxext6 && \
    rm -rf /var/lib/apt/lists/*

COPY . /workspace/WinClip

RUN pip3 install --upgrade pip && \
    pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html && \
    pip3 install setuptools==59.5.0 && \
    pip3 install --upgrade diffusers[torch] && \
    pip3 install opencv-python pycocotools matplotlib onnxruntime onnx ipykernel && \
    pip3 install transformers addict yapf timm loguru tqdm scikit-image scikit-learn pandas tensorboard seaborn open_clip_torch SciencePlots

CMD ["bash"]
