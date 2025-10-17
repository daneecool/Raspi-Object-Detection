#!/bin/bash
# Switch between Intel and NVIDIA GPU configurations

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="$SCRIPT_DIR/docker-compose.yml"
DOCKERFILE="$SCRIPT_DIR/Dockerfile"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

show_usage() {
    echo "Usage: $0 [intel|nvidia|status]"
    echo ""
    echo "Commands:"
    echo "  intel   - Switch to Intel GPU/CPU optimization"
    echo "  nvidia  - Switch to NVIDIA GPU optimization"
    echo "  status  - Show current configuration"
    echo ""
    echo "This script helps you switch between Intel and NVIDIA GPU configurations"
    echo "by commenting/uncommenting the appropriate lines in configuration files."
}

check_current_config() {
    echo "=== Current Configuration ==="
    
    # Check Dockerfile
    if grep -q "^FROM nvidia/cuda" "$DOCKERFILE"; then
        echo "Dockerfile: NVIDIA GPU (CUDA base image)"
    elif grep -q "^# FROM nvidia/cuda" "$DOCKERFILE"; then
        echo "Dockerfile: Intel optimized (NVIDIA commented out)"
    else
        echo "Dockerfile: Intel optimized (no NVIDIA support)"
    fi
    
    # Check docker-compose.yml
    if grep -q "^[[:space:]]*runtime: nvidia" "$COMPOSE_FILE"; then
        echo "Docker Compose: NVIDIA GPU enabled"
    elif grep -q "^[[:space:]]*# runtime: nvidia" "$COMPOSE_FILE"; then
        echo "Docker Compose: Intel GPU enabled (NVIDIA commented out)"
    else
        echo "Docker Compose: Intel GPU enabled"
    fi
    
    # Check requirements.txt
    if grep -q "^onnxruntime-gpu" "$REQUIREMENTS"; then
        echo "Requirements: NVIDIA GPU packages enabled"
    elif grep -q "^# onnxruntime-gpu" "$REQUIREMENTS"; then
        echo "Requirements: Intel packages enabled (NVIDIA commented out)"
    else
        echo "Requirements: Intel packages enabled"
    fi
    echo ""
}

switch_to_intel() {
    echo "Switching to Intel GPU/CPU configuration..."
    
    # Update Dockerfile
    sed -i 's/^FROM nvidia\/cuda/#&/' "$DOCKERFILE"
    sed -i 's/^# FROM ubuntu:22.04/FROM ubuntu:22.04/' "$DOCKERFILE"
    sed -i 's/^ENV CUDA_HOME/#&/' "$DOCKERFILE"
    sed -i 's/^ENV PATH=.*CUDA/#&/' "$DOCKERFILE"
    sed -i 's/^ENV LD_LIBRARY_PATH=.*CUDA/#&/' "$DOCKERFILE"
    
    # Update docker-compose.yml
    sed -i 's/^[[:space:]]*runtime: nvidia/    # &/' "$COMPOSE_FILE"
    sed -i 's/^[[:space:]]*- NVIDIA_VISIBLE_DEVICES/#      &/' "$COMPOSE_FILE"
    sed -i 's/^[[:space:]]*- NVIDIA_DRIVER_CAPABILITIES/#      &/' "$COMPOSE_FILE"
    sed -i 's/^[[:space:]]*# devices:/    devices:/' "$COMPOSE_FILE"
    sed -i 's/^[[:space:]]*#   - \/dev\/dri/      - \/dev\/dri/' "$COMPOSE_FILE"
    
    # Update requirements.txt
    sed -i 's/^onnxruntime-gpu/#&/' "$REQUIREMENTS"
    sed -i 's/^# intel-extension-for-pytorch/intel-extension-for-pytorch/' "$REQUIREMENTS"
    sed -i 's/^# openvino/openvino/' "$REQUIREMENTS"
    sed -i 's/^# onnxruntime-openvino/onnxruntime-openvino/' "$REQUIREMENTS"
    
    echo "✅ Switched to Intel configuration"
    echo "💡 Run 'docker-compose build' to rebuild with Intel optimizations"
}

switch_to_nvidia() {
    echo "Switching to NVIDIA GPU configuration..."
    
    # Update Dockerfile
    sed -i 's/^FROM ubuntu:22.04/#&/' "$DOCKERFILE"
    sed -i 's/^# FROM nvidia\/cuda/FROM nvidia\/cuda/' "$DOCKERFILE"
    sed -i 's/^# ENV CUDA_HOME/ENV CUDA_HOME/' "$DOCKERFILE"
    sed -i 's/^# ENV PATH=.*CUDA/ENV PATH=${CUDA_HOME}\/bin:${PATH}/' "$DOCKERFILE"
    sed -i 's/^# ENV LD_LIBRARY_PATH=.*CUDA/ENV LD_LIBRARY_PATH=${CUDA_HOME}\/lib64:${LD_LIBRARY_PATH}/' "$DOCKERFILE"
    
    # Update docker-compose.yml
    sed -i 's/^[[:space:]]*# runtime: nvidia/    runtime: nvidia/' "$COMPOSE_FILE"
    sed -i 's/^[[:space:]]*#   - NVIDIA_VISIBLE_DEVICES/      - NVIDIA_VISIBLE_DEVICES/' "$COMPOSE_FILE"
    sed -i 's/^[[:space:]]*#   - NVIDIA_DRIVER_CAPABILITIES/      - NVIDIA_DRIVER_CAPABILITIES/' "$COMPOSE_FILE"
    sed -i 's/^[[:space:]]*devices:/    # devices:/' "$COMPOSE_FILE"
    sed -i 's/^[[:space:]]*- \/dev\/dri/#      &/' "$COMPOSE_FILE"
    
    # Update requirements.txt
    sed -i 's/^# onnxruntime-gpu/onnxruntime-gpu/' "$REQUIREMENTS"
    
    echo "✅ Switched to NVIDIA configuration"
    echo "⚠️  Make sure you have NVIDIA Docker installed!"
    echo "💡 Run 'docker-compose build' to rebuild with NVIDIA support"
}

case "${1:-status}" in
    intel)
        switch_to_intel
        ;;
    nvidia)
        switch_to_nvidia
        ;;
    status)
        check_current_config
        ;;
    *)
        show_usage
        exit 1
        ;;
esac

echo ""
check_current_config