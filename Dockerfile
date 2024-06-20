FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    xformers==0.0.25 git+https://github.com/camenduru/Lumina-T2X@dev && \
    git clone -b dev https://github.com/camenduru/Lumina-T2X /content/Lumina-T2X && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/ckpt/Lumina-Next-SFT/resolve/main/consolidated_ema.00-of-01.safetensors -d /content/Lumina-T2X/models -o consolidated_ema.00-of-01.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/ckpt/Lumina-Next-SFT/resolve/main/model_args.pth -d /content/Lumina-T2X/models -o model_args.pth

COPY ./worker_runpod.py /content/Lumina-T2X/lumina_next_compositional_generation/worker_runpod.py
WORKDIR /content/Lumina-T2X/lumina_next_compositional_generation
CMD python worker_runpod.py
