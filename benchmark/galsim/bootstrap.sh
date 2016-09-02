#!/usr/bin/env bash

# Based on https://github.com/karenyyng/GalSim_dockerfile/blob/master/Dockerfile

set -evx

apt-get update
apt-get install -y \
        build-essential \
        gfortran \
        git \
        libatlas-base-dev \
        libboost-all-dev \
        libfftw3-dev \
        python \
        python-dev \
        python-pip \
        python-numpy \
        python-pandas \
        scons \
        software-properties-common \
        wget

pip install \
    astropy \
    future \
    starlink_pyast

mkdir -p /usr/src

cd /usr/src
echo "Installing TMV-cpp"
if [ ! -e tmv-0.73 ]; then
    wget https://github.com/rmjarvis/tmv/archive/v0.73.tar.gz
    tar xzvf v0.73.tar.gz
fi
cd tmv-0.73
scons
scons install

echo "Installing GalSim"
cd /usr/src
if [ ! -e GalSim ]; then git clone https://github.com/GalSim-developers/GalSim.git; fi
cd GalSim
git checkout releases/1.4
scons && scons install
