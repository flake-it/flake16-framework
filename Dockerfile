FROM ubuntu:focal

ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y \
    build-essential \
    cmake \
    default-jdk \
    git \
    libcurl4-openssl-dev \
    libfreetype6-dev \
    libfribidi-dev \
    libharfbuzz-dev \
    libjpeg8-dev \
    libkrb5-dev \
    liblapack-dev \
    liblcms2-dev \
    libldap2-dev \
    libmysqlclient-dev \
    libopenblas-dev \
    libopenjp2-7-dev \
    libpq-dev \
    libsasl2-dev \
    libsecp256k1-dev \
    libsndfile1-dev \
    libssl-dev \
    libtiff5-dev \
    libwebp-dev \
    libxcb1-dev \
    nano \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-tk \
    sudo \
    tcl8.6-dev \
    tk8.6-dev \
    unixodbc-dev \
    virtualenv \
    zlib1g-dev

RUN useradd -ms /bin/bash user && \
    echo 'user:password' | chpasswd && \
    usermod -aG sudo user

USER user

WORKDIR /home/user

COPY --chown=user subjects.txt ./

COPY --chown=user experiment.py ./

COPY --chown=user requirements.txt ./

COPY --chown=user subjects ./subjects

COPY --chown=user showflakes ./showflakes

COPY --chown=user testinspect ./testinspect

RUN pip3 install -r requirements.txt && \
    python3 experiment.py setup
