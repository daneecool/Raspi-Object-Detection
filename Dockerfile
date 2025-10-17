# Image Processing Environment with YOLO11n, PyTorch->ONNX, and NCNN
# Currently optimized for Intel GPU/CPU (NVIDIA support commented out for future use)
FROM ubuntu:22.04

# Uncomment below for NVIDIA GPU support in the future
# FROM nvidia/cuda:11.8-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Uncomment below for NVIDIA CUDA support
# ENV CUDA_HOME=/usr/local/cuda
# ENV PATH=${CUDA_HOME}/bin:${PATH}
# ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    pkg-config \
    libopencv-dev \
    libprotobuf-dev \
    protobuf-compiler \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Optional: Try to install additional GPU/graphics tools (may not be available on all systems)
RUN apt-get update && (apt-get install -y \
    mesa-utils \
    libvulkan1 \
    || true) && rm -rf /var/lib/apt/lists/*

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Build and install NCNN with proper tool installation
RUN git clone https://github.com/Tencent/ncnn.git && \
    cd ncnn && \
    git submodule update --init && \
    mkdir build && \
    cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DNCNN_BUILD_EXAMPLES=OFF \
          -DNCNN_BUILD_TOOLS=ON \
          -DNCNN_BUILD_TESTS=OFF \
          -DNCNN_PYTHON=ON \
          .. && \
    make -j$(nproc) && \
    make install && \
    ldconfig

# Install NCNN Python bindings properly  
RUN cd ncnn/python && \
    python3 setup.py build && \
    python3 setup.py install

# Create directories for models and data
RUN mkdir -p /workspace/models /workspace/data/input /workspace/data/output /workspace/scripts

# Copy project files
COPY . .

# Update PATH to include NCNN tools
ENV PATH=/usr/local/bin:$PATH
ENV NCNN_DIR=/usr/local

# Expose ports for services
EXPOSE 8888 5000

# Default command
CMD ["/bin/bash"]