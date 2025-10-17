#!/usr/bin/env python3
"""
Quick ONNX Optimization Test
Test different ONNX Runtime providers and settings
"""

import onnxruntime as ort
import numpy as np
import time
import cv2

def test_onnx_providers(onnx_path, image_path):
    """Test different ONNX Runtime providers"""
    print("🔧 Testing ONNX Runtime Optimizations")
    print("=" * 40)
    
    # Load and preprocess image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (640, 640))
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_transposed = np.transpose(image_normalized, (2, 0, 1))
    image_batch = np.expand_dims(image_transposed, axis=0)
    
    providers_to_test = [
        'CPUExecutionProvider',
        ['CPUExecutionProvider'],
    ]
    
    # Test with different CPU optimization settings
    cpu_options = [
        {},  # Default
        {'intra_op_num_threads': 4},
        {'intra_op_num_threads': 8},
        {'inter_op_num_threads': 1, 'intra_op_num_threads': 4},
    ]
    
    for i, cpu_opts in enumerate(cpu_options):
        print(f"\n{i+1}. Testing CPU provider with options: {cpu_opts}")
        
        try:
            # Create session with options
            sess_options = ort.SessionOptions()
            if 'intra_op_num_threads' in cpu_opts:
                sess_options.intra_op_num_threads = cpu_opts['intra_op_num_threads']
            if 'inter_op_num_threads' in cpu_opts:
                sess_options.inter_op_num_threads = cpu_opts['inter_op_num_threads']
            
            session = ort.InferenceSession(onnx_path, sess_options=sess_options, providers=['CPUExecutionProvider'])
            input_name = session.get_inputs()[0].name
            
            # Warmup
            for _ in range(3):
                session.run(None, {input_name: image_batch})
            
            # Benchmark
            num_runs = 10
            start_time = time.time()
            for _ in range(num_runs):
                outputs = session.run(None, {input_name: image_batch})
            avg_time = (time.time() - start_time) / num_runs * 1000
            
            print(f"   Average time: {avg_time:.2f} ms")
            
        except Exception as e:
            print(f"   Failed: {e}")

if __name__ == "__main__":
    onnx_path = "/workspace/models/yolo11n.onnx"
    image_path = "/workspace/data/input/bus.jpg"
    
    print(f"ONNX Runtime version: {ort.__version__}")
    print(f"Available providers: {ort.get_available_providers()}")
    
    test_onnx_providers(onnx_path, image_path)