# syntax=docker/dockerfile:1

FROM rayproject/ray-ml:latest-gpu

EXPOSE 8265
#Tensorboard  run tensorboard --bind_all --logdir .
EXPOSE 6006 

COPY requirements_factorySim.txt .
USER root

RUN  apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libcairo2-dev \
        pkg-config \
        python3-dev \
        ffmpeg \
        wget \
    && wget -nv https://s3.amazonaws.com/ifcopenshell-builds/ifcopenshell-python-37-v0.6.0-517b819-linux64.zip \
    && unzip -q ifcopenshell-python-37-v0.6.0-517b819-linux64.zip -d $HOME/anaconda3/lib/python3.7/site-packages \
    && runuser -l  ray -c '$HOME/anaconda3/bin/pip --no-cache-dir install -U -r requirements_factorySim.txt' \
    && apt-get clean \
    && rm ifcopenshell-python-37-v0.6.0-517b819-linux64.zip \
    && rm -rf /tmp/* \
    && apt-get remove wget -y\
    && rm -rf /var/lib/apt/lists/*
 
WORKDIR $HOME/factorySim

COPY . .


WORKDIR $HOME/factorySim/env
RUN $HOME/anaconda3/bin/pip install -e .
USER ray
WORKDIR $HOME/factorySim
