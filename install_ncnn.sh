#!/bin/bash
"""
NCNN Installation Script for Docker Container
Run this inside the Docker container to install NCNN tools properly.
"""

echo "🔧 Installing NCNN Tools in Docker Container"
echo "=============================================="

# Update package list
apt-get update

# Install additional dependencies
apt-get install -y \
    protobuf-compiler \
    libprotobuf-dev \
    libopencv-dev

# Navigate to a temporary directory
cd /tmp

# Remove any existing NCNN directory
rm -rf ncnn

# Clone NCNN repository
echo "📥 Cloning NCNN repository..."
git clone https://github.com/Tencent/ncnn.git
cd ncnn

# Update submodules
echo "🔄 Updating submodules..."
git submodule update --init

# Create build directory
mkdir -p build
cd build

# Configure build with all tools enabled
echo "⚙️  Configuring NCNN build..."
cmake -DCMAKE_BUILD_TYPE=Release \
      -DNCNN_BUILD_TOOLS=ON \
      -DNCNN_BUILD_EXAMPLES=OFF \
      -DNCNN_BUILD_TESTS=OFF \
      -DNCNN_PYTHON=ON \
      -DNCNN_BUILD_BENCHMARK=OFF \
      ..

# Build NCNN
echo "🔨 Building NCNN (this may take a while)..."
make -j$(nproc)

# Install NCNN tools to system
echo "📦 Installing NCNN tools..."
make install

# Add tools to PATH
echo "🔗 Adding NCNN tools to PATH..."
echo 'export PATH=/usr/local/bin:$PATH' >> ~/.bashrc
export PATH=/usr/local/bin:$PATH

# Install Python bindings
echo "🐍 Installing NCNN Python bindings..."
cd ../python
python3 setup.py build
python3 setup.py install

echo "✅ NCNN installation complete!"
echo ""
echo "🧪 Testing installation..."
which onnx2ncnn || echo "❌ onnx2ncnn not found in PATH"
which pnnx || echo "❌ pnnx not found in PATH"
python3 -c "import ncnn; print('✅ NCNN Python bindings working')" || echo "❌ NCNN Python bindings failed"

echo ""
echo "🚀 If successful, you can now use:"
echo "  - onnx2ncnn for ONNX to NCNN conversion"
echo "  - NCNN Python bindings for inference"