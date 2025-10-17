#!/usr/bin/env python3
"""
Alternative Image Processing Pipeline
This pipeline focuses on PyTorch and ONNX formats since NCNN tools are not available.
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def run_yolo_detection():
    """Run YOLO detection on sample images"""
    print("1. Running YOLO Detection...")
    try:
        # Import and use ultralytics directly instead of yolo_detection module
        from ultralytics import YOLO
        
        # Load model
        model = YOLO('../models/yolo11n.pt')
        
        # Run detection on all sample images
        sample_images = [
            '../data/input/bus.jpg',
            '../data/input/dog.jpg', 
            '../data/input/zidane.jpg'
        ]
        
        for image_path in sample_images:
            if os.path.exists(image_path):
                print(f"   Processing: {image_path}")
                results = model(image_path, conf=0.5, save=True, project='../data/output')
                print(f"   Detected {len(results[0].boxes)} objects")
            else:
                print(f"   Warning: {image_path} not found")
        
        return True
        
    except Exception as e:
        print(f"❌ YOLO detection failed: {e}")
        return False

def convert_to_onnx():
    """Convert PyTorch model to ONNX format"""
    print("2. Converting PyTorch to ONNX...")
    try:
        from simple_onnx_convert import convert_yolo_simple, verify_onnx_simple
        
        model_path = '../models/yolo11n.pt'
        onnx_path = '../models/yolo11n_optimized.onnx'
        
        # Convert model
        result_path = convert_yolo_simple(model_path, onnx_path, imgsz=640)
        
        if result_path:
            # Verify the converted model
            if verify_onnx_simple(result_path):
                print(f"✅ ONNX conversion successful: {result_path}")
                return result_path
            else:
                print("❌ ONNX verification failed")
                return None
        else:
            print("❌ ONNX conversion failed")
            return None
            
    except Exception as e:
        print(f"❌ ONNX conversion failed: {e}")
        return None

def test_onnx_inference(onnx_path):
    """Test ONNX model inference"""
    print("3. Testing ONNX Inference...")
    try:
        import onnxruntime as ort
        import numpy as np
        from PIL import Image
        import cv2
        
        # Load ONNX model
        session = ort.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        
        print(f"   ONNX model loaded: {input_name} {input_shape}")
        
        # Load and preprocess test image
        test_image = '../data/input/bus.jpg'
        if not os.path.exists(test_image):
            print(f"   Warning: Test image {test_image} not found")
            return False
        
        image = cv2.imread(test_image)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (640, 640))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_transposed = np.transpose(image_normalized, (2, 0, 1))
        image_batch = np.expand_dims(image_transposed, axis=0)
        
        # Run inference and measure time
        start_time = time.time()
        outputs = session.run(None, {input_name: image_batch})
        inference_time = (time.time() - start_time) * 1000
        
        print("✅ ONNX inference successful!")
        print(f"   Inference time: {inference_time:.2f}ms")
        print(f"   Output shape: {outputs[0].shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX inference failed: {e}")
        return False

def benchmark_models():
    """Benchmark PyTorch vs ONNX performance"""
    print("4. Benchmarking Models...")
    try:
        from ultralytics import YOLO
        import time
        
        # Test image
        test_image = '../data/input/bus.jpg'
        if not os.path.exists(test_image):
            print(f"   Warning: Test image {test_image} not found")
            return
        
        # Benchmark PyTorch model
        print("   Benchmarking PyTorch model...")
        pytorch_model = YOLO('../models/yolo11n.pt')
        
        pytorch_times = []
        for i in range(5):
            start_time = time.time()
            pytorch_model(test_image, verbose=False)
            pytorch_times.append((time.time() - start_time) * 1000)
        
        avg_pytorch_time = sum(pytorch_times) / len(pytorch_times)
        
        # Benchmark ONNX model if available
        onnx_path = '../models/yolo11n.onnx'
        if not os.path.exists(onnx_path):
            onnx_path = '../models/yolo11n_optimized.onnx'
            
        if os.path.exists(onnx_path):
            print("   Benchmarking ONNX model...")
            onnx_model = YOLO(onnx_path)
            
            onnx_times = []
            for i in range(5):
                start_time = time.time()
                onnx_model(test_image, verbose=False)
                onnx_times.append((time.time() - start_time) * 1000)
            
            avg_onnx_time = sum(onnx_times) / len(onnx_times)
            
            # Display results
            print("\n📊 Performance Comparison:")
            print(f"   PyTorch: {avg_pytorch_time:.2f}ms (avg)")
            print(f"   ONNX:    {avg_onnx_time:.2f}ms (avg)")
            speedup = avg_pytorch_time / avg_onnx_time
            print(f"   ONNX Speedup: {speedup:.2f}x")
        else:
            print(f"   PyTorch: {avg_pytorch_time:.2f}ms (avg)")
            print("   ONNX model not available for comparison")
            
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")

def print_summary():
    """Print pipeline summary"""
    print("\n" + "="*50)
    print("📋 PIPELINE SUMMARY")
    print("="*50)
    
    # Check what files were created
    models_dir = Path('../models')
    output_dir = Path('../data/output')
    
    print("\n🔧 Available Models:")
    if (models_dir / 'yolo11n.pt').exists():
        size = (models_dir / 'yolo11n.pt').stat().st_size / (1024*1024)
        print(f"   ✅ PyTorch: yolo11n.pt ({size:.1f}MB)")
    
    if (models_dir / 'yolo11n.onnx').exists():
        size = (models_dir / 'yolo11n.onnx').stat().st_size / (1024*1024)
        print(f"   ✅ ONNX: yolo11n.onnx ({size:.1f}MB)")
    elif (models_dir / 'yolo11n_optimized.onnx').exists():
        size = (models_dir / 'yolo11n_optimized.onnx').stat().st_size / (1024*1024)
        print(f"   ✅ ONNX: yolo11n_optimized.onnx ({size:.1f}MB)")
    
    print("   ❌ NCNN: Not available (tools missing)")
    
    print("\n📁 Output Files:")
    if output_dir.exists():
        output_files = list(output_dir.glob('*'))
        for file in output_files:
            if file.is_file():
                print(f"   ✅ {file.name}")
    
    print("\n🚀 Next Steps:")
    print("   1. To install NCNN tools properly, rebuild Docker with NCNN support")
    print("   2. For now, use ONNX format for optimized inference")
    print("   3. ONNX models work with OpenVINO for Intel hardware optimization")

def main():
    parser = argparse.ArgumentParser(description='Alternative Image Processing Pipeline')
    parser.add_argument('--mode', choices=['full', 'detect', 'convert', 'benchmark'], 
                        default='full', help='Pipeline mode to run')
    
    args = parser.parse_args()
    
    print("🔄 Starting Alternative Image Processing Pipeline")
    print("="*50)
    
    success_count = 0
    
    if args.mode in ['full', 'detect']:
        if run_yolo_detection():
            success_count += 1
    
    if args.mode in ['full', 'convert']:
        onnx_path = convert_to_onnx()
        if onnx_path:
            success_count += 1
            if test_onnx_inference(onnx_path):
                success_count += 1
    
    if args.mode in ['full', 'benchmark']:
        benchmark_models()
    
    print_summary()
    
    print(f"\n✅ Pipeline completed with {success_count} successful steps")

if __name__ == "__main__":
    main()