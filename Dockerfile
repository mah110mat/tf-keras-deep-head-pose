FROM nvcr.io/nvidia/tensorflow:21.03-tf2-py3

ENV DEBIAN_FRONTEND noninteractive
# Install linux packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential curl gdb git wget tmux byobu fdclone locales tzdata 
RUN apt-get update && apt-get install -y screen libgl1-mesa-glx libsm6 libxrender1 libxext-dev
RUN apt-get update && apt-get install -y python3-cairocffi \
    protobuf-compiler python3-pil python3-lxml python3-tk 
RUN pip3 install opencv-python
RUN pip3 install matplotlib
RUN pip3 install pandas


