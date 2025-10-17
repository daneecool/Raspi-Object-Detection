#!/usr/bin/env python3
"""
Intel GPU/CPU Optimized Inference Script
This script demonstrates how to use Intel optimizations for faster inference.
"""

import os
import cv2
import torch
import argparse
import time
from pathlib import Path

# Intel Extension for PyTorch
try:
    import intel_extension_for_pytorch as ipex
    IPEX_AVAILABLE = True
    print("Intel Extension for PyTorch is available")
except ImportError:
    IPEX_AVAILABLE = False
    print("Intel Extension for PyTorch not available")

# OpenVINO
try:
    from openvino.runtime import Core
    OPENVINO_AVAILABLE = True
    print("OpenVINO is available")
except ImportError:
    OPENVINO_AVAILABLE = False
    print("OpenVINO not available")

from ultralytics import YOLO

def optimize_yolo_for_intel(model_path, device='cpu'):
    """Optimize YOLO model for Intel hardware"""
    print(f"Loading YOLO model: {model_path}")
    model = YOLO(model_path)
    
    if IPEX_AVAILABLE and device == 'xpu':
        print("Optimizing model for Intel GPU (XPU)")
        # Intel GPU optimization
        model.model = ipex.optimize(model.model, dtype=torch.float16, level="O1")
    elif IPEX_AVAILABLE and device == 'cpu':
        print("Optimizing model for Intel CPU")
        # Intel CPU optimization with oneDNN
        model.model = ipex.optimize(model.model, dtype=torch.float32)
    
    return model

def convert_to_openvino(model_path, output_path):
    """Convert YOLO model to OpenVINO format for Intel optimization"""
    if not OPENVINO_AVAILABLE:
        print("OpenVINO not available for conversion")
        return None
    
    print(f"Converting {model_path} to OpenVINO format...")
    
    # First convert to ONNX
    model = YOLO(model_path)
    onnx_path = output_path.replace('.xml', '.onnx')
    
    try:
        # Export to ONNX first
        model.export(format='onnx', imgsz=640)
        onnx_file = model_path.replace('.pt', '.onnx')
        
        # Convert ONNX to OpenVINO
        import subprocess
        cmd = f"mo --input_model {onnx_file} --output_dir {Path(output_path).parent}"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"OpenVINO model saved to: {output_path}")
            return output_path
        else:
            print(f"OpenVINO conversion failed: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"Conversion error: {e}")
        return None

class OpenVINOInference:
    """OpenVINO inference for Intel optimization"""
    
    def __init__(self, model_path, device='CPU'):
        if not OPENVINO_AVAILABLE:
            raise ImportError("OpenVINO not available")
        
        self.core = Core()
        self.device = device
        
        # Available devices
        available_devices = self.core.available_devices
        print(f"Available OpenVINO devices: {available_devices}")
        
        # Load model
        self.model = self.core.read_model(model_path)
        self.compiled_model = self.core.compile_model(self.model, device)
        
        # Get input/output info
        self.input_layer = self.compiled_model.input(0)
        self.output_layer = self.compiled_model.output(0)
        
        print(f"OpenVINO model loaded on {device}")
        print(f"Input shape: {self.input_layer.shape}")
        print(f"Output shape: {self.output_layer.shape}")
    
    def preprocess_image(self, image, target_size=(640, 640)):
        """Preprocess image for OpenVINO inference"""
        # Resize
        image_resized = cv2.resize(image, target_size)
        
        # Convert BGR to RGB and normalize
        image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
        image_norm = image_rgb.astype('float32') / 255.0
        
        # Add batch dimension and transpose to NCHW
        image_batch = image_norm.transpose(2, 0, 1)[None, ...]
        
        return image_batch
    
    def inference(self, image):
        """Run OpenVINO inference"""
        preprocessed = self.preprocess_image(image)
        result = self.compiled_model([preprocessed])[self.output_layer]
        return result
    
    def benchmark(self, image, num_runs=100):
        """Benchmark OpenVINO inference"""
        print(f"Benchmarking OpenVINO on {self.device} ({num_runs} runs)...")
        
        # Warmup
        for _ in range(10):
            self.inference(image)
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            self.inference(image)
        end_time = time.time()
        
        avg_time = (end_time - start_time) / num_runs * 1000  # ms
        fps = 1000 / avg_time
        
        print(f"OpenVINO {self.device} - Average time: {avg_time:.2f} ms, FPS: {fps:.1f}")
        return avg_time, fps

def detect_with_intel_optimization(model_path, image_path, output_path, device='cpu'):
    """Run optimized detection on Intel hardware"""
    # Load and optimize model
    model = optimize_yolo_for_intel(model_path, device)
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return
    
    print(f"Processing image: {image_path}")
    print(f"Using device: {device}")
    
    # Run inference with timing
    start_time = time.time()
    results = model(image, device=device)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.3f} seconds")
    
    # Process and save results
    annotated_image = results[0].plot()
    cv2.imwrite(output_path, annotated_image)
    print(f"Result saved to: {output_path}")
    
    # Print detection summary
    boxes = results[0].boxes
    if boxes is not None:
        print(f"Detected {len(boxes)} objects:")
        for box in boxes:
            conf = box.conf.item()
            cls_id = int(box.cls.item())
            cls_name = model.names[cls_id]
            print(f"  - {cls_name}: {conf:.2f}")

def benchmark_intel_optimizations(model_path, image_path):
    """Compare performance of different Intel optimizations"""
    print("=== Intel Hardware Optimization Benchmark ===")
    
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load test image: {image_path}")
        return
    
    # 1. Standard PyTorch CPU
    print("\n1. Standard PyTorch CPU")
    model_standard = YOLO(model_path)
    
    # Warmup
    for _ in range(5):
        model_standard(image, device='cpu')
    
    # Benchmark
    start_time = time.time()
    num_runs = 20
    for _ in range(num_runs):
        model_standard(image, device='cpu')
    standard_time = (time.time() - start_time) / num_runs * 1000
    print(f"Standard PyTorch CPU: {standard_time:.2f} ms")
    
    # 2. Intel Extension for PyTorch (if available)
    if IPEX_AVAILABLE:
        print("\n2. Intel Extension for PyTorch (CPU)")
        model_ipex = optimize_yolo_for_intel(model_path, 'cpu')
        
        # Warmup
        for _ in range(5):
            model_ipex(image, device='cpu')
        
        # Benchmark
        start_time = time.time()
        for _ in range(num_runs):
            model_ipex(image, device='cpu')
        ipex_time = (time.time() - start_time) / num_runs * 1000
        print(f"Intel Extension CPU: {ipex_time:.2f} ms ({standard_time/ipex_time:.1f}x speedup)")
    
    # 3. ONNX Runtime with OpenVINO (if available)
    if OPENVINO_AVAILABLE:
        print("\n3. ONNX Runtime with OpenVINO")
        try:
            import onnxruntime as ort
            
            # Convert to ONNX first
            onnx_path = model_path.replace('.pt', '_intel.onnx')
            if not os.path.exists(onnx_path):
                model_temp = YOLO(model_path)
                model_temp.export(format='onnx', imgsz=640)
                onnx_path = model_path.replace('.pt', '.onnx')
            
            # Create OpenVINO provider session
            providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
            session = ort.InferenceSession(onnx_path, providers=providers)
            
            # Prepare input
            input_name = session.get_inputs()[0].name
            image_resized = cv2.resize(image, (640, 640))
            image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
            image_norm = image_rgb.astype('float32') / 255.0
            image_batch = image_norm.transpose(2, 0, 1)[None, ...]
            
            # Warmup
            for _ in range(5):
                session.run(None, {input_name: image_batch})
            
            # Benchmark
            start_time = time.time()
            for _ in range(num_runs):
                session.run(None, {input_name: image_batch})
            openvino_time = (time.time() - start_time) / num_runs * 1000
            print(f"ONNX + OpenVINO: {openvino_time:.2f} ms ({standard_time/openvino_time:.1f}x speedup)")
            
        except Exception as e:
            print(f"OpenVINO benchmark failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Intel-Optimized YOLO Inference')
    parser.add_argument('--model', '-m', type=str, default='/workspace/models/yolo11n.pt',
                        help='Path to YOLO model')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input image path')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output image path')
    parser.add_argument('--device', choices=['cpu', 'xpu'], default='cpu',
                        help='Device to use (cpu for Intel CPU, xpu for Intel GPU)')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run comprehensive benchmark')
    parser.add_argument('--openvino', action='store_true',
                        help='Use OpenVINO for inference')
    
    args = parser.parse_args()
    
    if args.benchmark:
        benchmark_intel_optimizations(args.model, args.input)
    elif args.openvino and OPENVINO_AVAILABLE:
        # OpenVINO inference path
        print("OpenVINO inference not fully implemented in this example")
        print("Use --benchmark to see OpenVINO performance comparison")
    else:
        # Standard optimized inference
        detect_with_intel_optimization(args.model, args.input, args.output, args.device)

if __name__ == "__main__":
    main()