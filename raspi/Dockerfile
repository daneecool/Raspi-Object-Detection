# Raspberry Pi 3B v1.2 Optimized Dockerfile
# ARM64 architecture with minimal dependencies for 1GB RAM
FROM arm64v8/ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Raspberry Pi specific optimizations
ENV OPENCV_LOG_LEVEL=ERROR
ENV YOLO_VERBOSE=False

# Install minimal system dependencies for Pi
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
    # OpenCV dependencies (minimal)
    libopencv-dev \
    python3-opencv \
    # Camera and video dependencies
    v4l-utils \
    # System monitoring
    htop \
    # Cleanup
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove -y \
    && apt-get autoclean

# Create symbolic link for python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory
WORKDIR /workspace

# Copy Pi-specific requirements first for better Docker layer caching
COPY requirements.raspi.txt requirements.txt

# Install Python dependencies with Pi optimizations
RUN pip3 install --no-cache-dir --break-system-packages \
    --index-url https://pypi.org/simple \
    --extra-index-url https://www.piwheels.org/simple \
    -r requirements.txt

# Create directories for models and data
RUN mkdir -p /workspace/models /workspace/data/input /workspace/data/output /workspace/scripts /workspace/tests

# Copy project files
COPY scripts/obj_detection.py scripts/
COPY data/ data/ 2>/dev/null || true

# Set up model directory permissions
RUN chmod -R 755 /workspace

# Add Pi-specific environment optimizations
ENV OMP_NUM_THREADS=4
ENV MKL_NUM_THREADS=4
ENV OPENBLAS_NUM_THREADS=4

# Expose ports for services (optional)
EXPOSE 5000

# Health check for Pi
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import cv2, ultralytics; print('OK')" || exit 1

# Default command
CMD ["/bin/bash"]