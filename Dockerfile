FROM pinto0309/ubuntu22.04-cuda11.8-tensorrt8.5.3:latest

ENV DEBIAN_FRONTEND=noninteractive

RUN sudo apt-get update \
    && sudo apt-get install -y \
        nano python3-pip python3-mock libpython3-dev \
        libpython3-all-dev python-is-python3 wget curl cmake \
        software-properties-common sudo git \
    && sudo sed -i 's/# set linenumbers/set linenumbers/g' /etc/nanorc \
    && sudo apt clean \
    && sudo rm -rf /var/lib/apt/lists/*

RUN pip install pip -U \
    && pip install \
    onnx==1.15.0 \
    onnxsim==0.4.33 \
    # TensorRT 8.5.3
    https://github.com/PINTO0309/BoT-SORT-ONNX-TensorRT/releases/download/onnx/onnxruntime_gpu-1.16.1-cp310-cp310-linux_x86_64.whl \
    sit4onnx==1.0.7 \
    opencv-contrib-python==4.9.0.80 \
    numpy==1.24.3 \
    scipy==1.10.1

RUN pip install lap==0.4.0

RUN echo "sudo chmod 777 /dev/video*" >> ~/.bashrc