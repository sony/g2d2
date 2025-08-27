FROM nvidia/cuda:12.1.0-runtime-ubuntu20.04
LABEL maintainer="Hugging Face"
LABEL repository="diffusers"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && \
    apt install -y bash \
    build-essential \
    git \
    git-lfs \
    curl \
    ca-certificates \
    libsndfile1-dev \
    libgl1 \
    python3.8 \
    python3-pip \
    python3.8-venv && \
    rm -rf /var/lib/apt/lists

# make sure to use venv
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# pre-install the heavy dependencies (these can later be overridden by the deps from setup.py)
RUN python3 -m pip install --no-cache-dir --upgrade pip==23.3.1 && \
    python3 -m pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    invisible-watermark==0.2.0 && \
    python3 -m pip install --no-cache-dir \
    accelerate==0.24.1 \
    datasets==2.14.6 \
    hf-doc-builder==0.4.0 \
    huggingface-hub==0.17.3 \
    Jinja2==3.1.2 \
    librosa==0.10.1 \
    numpy==1.24.4 \
    scipy==1.10.1 \
    tensorboard==2.14.0 \
    transformers==4.35.0 \
    omegaconf==2.3.0 \
    pytorch-lightning==2.1.0 \
    diffusers[torch]==0.22.1 \
    matplotlib==3.7.3

CMD ["/bin/bash"]