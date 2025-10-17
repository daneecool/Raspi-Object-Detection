#!/usr/bin/env python3
"""
NCNN Conversion Workaround
Since NCNN tools are not available, this script provides alternative solutions.
"""

import subprocess

def check_ncnn_availability():
    """Check if NCNN tools are available"""
    print("🔍 Checking NCNN Tool Availability...")
    
    # Check for onnx2ncnn tool
    onnx2ncnn_found = False
    try:
        result = subprocess.run(['which', 'onnx2ncnn'], capture_output=True, text=True)
        if result.returncode == 0:
            onnx2ncnn_found = True
            print(f"✅ onnx2ncnn found at: {result.stdout.strip()}")
        else:
            print("❌ onnx2ncnn not found in PATH")
    except Exception:
        print("❌ Cannot check for onnx2ncnn")
    
    # Check for PNNX (recommended alternative)
    pnnx_found = False
    try:
        result = subprocess.run(['which', 'pnnx'], capture_output=True, text=True)
        if result.returncode == 0:
            pnnx_found = True
            print(f"✅ pnnx found at: {result.stdout.strip()}")
        else:
            print("❌ pnnx not found in PATH")
    except Exception:
        print("❌ Cannot check for pnnx")
    
    # Check for NCNN Python bindings
    ncnn_python_found = False
    try:
        import importlib.util
        if importlib.util.find_spec("ncnn") is not None:
            ncnn_python_found = True
            print("✅ NCNN Python bindings available")
        else:
            print("❌ NCNN Python bindings not available")
    except ImportError:
        print("❌ NCNN Python bindings not available")
    
    return onnx2ncnn_found, pnnx_found, ncnn_python_found

def print_alternatives():
    """Print alternative solutions"""
    print("\n" + "="*60)
    print("🔧 NCNN CONVERSION ALTERNATIVES")
    print("="*60)
    
    print("\n1️⃣  Use ONNX Runtime with OpenVINO Backend (Recommended)")
    print("   - Optimized for Intel hardware")
    print("   - Good performance on CPU and Intel GPU")
    print("   - Already working in this container")
    print("   Example:")
    print("   ```python")
    print("   import onnxruntime as ort")
    print("   session = ort.InferenceSession('model.onnx',")
    print("                                   providers=['OpenVINOExecutionProvider'])")
    print("   ```")
    
    print("\n2️⃣  Use PyTorch with Intel Extension (IPEX)")
    print("   - Intel-optimized PyTorch")
    print("   - Good CPU performance")
    print("   - Available in this container")
    print("   Example:")
    print("   ```python")
    print("   import torch")
    print("   import intel_extension_for_pytorch as ipex")
    print("   model = torch.jit.trace(model, example_input)")
    print("   model = ipex.optimize(model)")
    print("   ```")
    
    print("\n3️⃣  Build NCNN Tools Manually")
    print("   - Clone NCNN repository")
    print("   - Build with CMake")
    print("   - Install tools to system PATH")
    print("   Commands:")
    print("   ```bash")
    print("   git clone https://github.com/Tencent/ncnn.git")
    print("   cd ncnn && mkdir build && cd build")
    print("   cmake -DCMAKE_BUILD_TYPE=Release ..")
    print("   make -j$(nproc)")
    print("   sudo make install")
    print("   ```")
    
    print("\n4️⃣  Use TensorRT (NVIDIA GPUs)")
    print("   - High performance on NVIDIA hardware")
    print("   - Requires NVIDIA GPU and TensorRT")
    print("   - Convert ONNX to TensorRT engine")
    
    print("\n5️⃣  Use TensorFlow Lite")
    print("   - Good for mobile/edge deployment")
    print("   - Convert PyTorch → ONNX → TensorFlow → TFLite")

def print_performance_comparison():
    """Print performance comparison of different backends"""
    print("\n" + "="*60)
    print("⚡ PERFORMANCE COMPARISON (Typical)")
    print("="*60)
    
    print("\nInference Speed (relative to PyTorch CPU):")
    print("   PyTorch CPU:           1.0x   (baseline)")
    print("   PyTorch + IPEX:        1.2-1.5x")
    print("   ONNX Runtime:          1.3-1.8x")
    print("   ONNX + OpenVINO:       1.5-2.5x")
    print("   NCNN:                  2.0-3.0x")
    print("   TensorRT (NVIDIA):     3.0-5.0x")
    
    print("\nModel Size (relative to PyTorch):")
    print("   PyTorch (.pt):         1.0x")
    print("   ONNX (.onnx):          1.8-2.0x")
    print("   NCNN (.bin + .param): 0.8-1.0x")
    print("   TensorRT (.engine):    0.6-0.8x")

def suggest_current_best_option():
    """Suggest the best option for current setup"""
    print("\n" + "="*60)
    print("💡 RECOMMENDED FOR YOUR SETUP")
    print("="*60)
    
    print("\nBased on your Intel hardware and current container:")
    print("\n🥇 BEST OPTION: ONNX Runtime with OpenVINO")
    print("   ✅ Already installed")
    print("   ✅ Intel hardware optimized")
    print("   ✅ Good performance")
    print("   ✅ Easy to use")
    
    print("\n🥈 SECOND OPTION: PyTorch with Intel Extension")
    print("   ✅ Already installed")
    print("   ✅ Intel optimized")
    print("   ⚠️  Requires model optimization")
    
    print("\n🥉 THIRD OPTION: Build NCNN manually")
    print("   ⚠️  Requires rebuilding container")
    print("   ⚠️  Additional complexity")
    print("   ✅ Best performance potential")

def main():
    print("🔧 NCNN Conversion Troubleshooting")
    print("="*50)
    
    # Check availability
    onnx2ncnn, pnnx, ncnn_python = check_ncnn_availability()
    
    if not any([onnx2ncnn, pnnx, ncnn_python]):
        print("\n❌ No NCNN tools available in current environment")
        print_alternatives()
        print_performance_comparison() 
        suggest_current_best_option()
    else:
        print("\n✅ Some NCNN tools are available!")
        if onnx2ncnn:
            print("   You can use onnx2ncnn for conversion")
        if pnnx:
            print("   You can use pnnx (recommended)")
        if ncnn_python:
            print("   You can use NCNN Python bindings")

if __name__ == "__main__":
    main()