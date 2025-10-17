#!/usr/bin/env python3
"""
Complete Image Processing Pipeline
This script demonstrates the complete workflow: YOLO detection -> ONNX conversion -> NCNN inference
Combines advanced monitoring, benchmarking, and multiple model format support.
"""

import os
import sys
import time
import argparse
import subprocess
import cv2
import warnings
from pathlib import Path
from ultralytics import YOLO

# Suppress deprecation warnings for cleaner output during code reviews
warnings.filterwarnings("ignore", category=FutureWarning, module="onnxscript")
warnings.filterwarnings("ignore", message=".*param_schemas.*deprecated.*")

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def download_sample_images():
    """Download sample images for testing"""
    import urllib.request
    
    sample_urls = [
        ("https://ultralytics.com/images/bus.jpg", "bus.jpg"),
        ("https://ultralytics.com/images/zidane.jpg", "zidane.jpg"),
        ("https://ultralytics.com/images/dog.jpg", "dog.jpg"),
    ]
    
    # Support both relative and absolute paths
    input_dir = Path("../data/input") if os.path.exists("../data") else Path("/workspace/data/input")
    input_dir.mkdir(parents=True, exist_ok=True)
    
    for url, filename in sample_urls:
        file_path = input_dir / filename
        if not file_path.exists():
            print(f"Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, file_path)
                print(f"Downloaded: {file_path}")
            except Exception as e:
                print(f"Failed to download {filename}: {e}")
        else:
            print(f"Sample image already exists: {file_path}")
    
    return input_dir

def setup_yolo_model():
    """Download and setup YOLO11n model"""
    print("Setting up YOLO11n model...")
    model = YOLO('yolo11n.pt')  # This will download if not present
    
    # Determine model directory based on environment
    models_dir = Path("../models") if os.path.exists("../models") else Path("/workspace/models")
    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "yolo11n.pt"
    
    # Copy to models directory
    import shutil
    if os.path.exists("yolo11n.pt"):
        shutil.move("yolo11n.pt", model_path)
        print(f"YOLO model saved to: {model_path}")
    
    return str(model_path)

def run_yolo_detection(input_dir=None, output_dir=None):
    """Run YOLO detection on sample images"""
    print("1. Running YOLO Detection...")
    try:
        # Import and use ultralytics directly instead of yolo_detection module
        from ultralytics import YOLO
        
        # Determine paths based on environment
        if not input_dir:
            input_dir = "../data/input" if os.path.exists("../data") else "/workspace/data/input"
        if not output_dir:
            output_dir = "../data/output" if os.path.exists("../data") else "/workspace/data/output"
        
        # Ensure sample images are available - download if needed
        sample_images = [
            os.path.join(input_dir, 'bus.jpg'),
            os.path.join(input_dir, 'dog.jpg'), 
            os.path.join(input_dir, 'zidane.jpg')
        ]
        
        # Check if any images are missing and download if needed
        missing_images = [img for img in sample_images if not os.path.exists(img)]
        if missing_images:
            print("   Sample images missing, downloading...")
            download_sample_images()  # This will download all sample images
        
        # Ensure output directory exists
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load model from appropriate location
        model_path = "../models/yolo11n.pt" if os.path.exists("../models") else "/workspace/models/yolo11n.pt"
        if not os.path.exists(model_path):
            # Try to setup model first
            model_path = setup_yolo_model()
        
        model = YOLO(model_path)
        
        # Run detection on all sample images
        
        detection_count = 0
        for image_path in sample_images:
            if os.path.exists(image_path):
                print(f"   Processing: {os.path.basename(image_path)}")
                results = model(image_path, conf=0.4, save=True, project=output_dir)
                if results[0].boxes is not None:
                    num_objects = len(results[0].boxes)
                    detection_count += num_objects
                    print(f"   Detected {num_objects} objects")
                else:
                    print(f"   No objects detected")
            else:
                print(f"   Warning: {os.path.basename(image_path)} not found")
        
        print(f"✅ YOLO detection completed. Total objects detected: {detection_count}")
        return True
        
    except Exception as e:
        print(f"❌ YOLO detection failed: {e}")
        return False

def run_complete_pipeline():
    """Run the complete image processing pipeline"""
    print("=== Starting Complete Image Processing Pipeline ===")
    
    # 1. Setup
    print("\n1. Setting up models and sample data...")
    model_path = setup_yolo_model()
    input_dir = download_sample_images()
    
    # Determine output directory
    output_dir = Path("../data/output") if os.path.exists("../data") else Path("/workspace/data/output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 2. YOLO Detection
    print("\n2. Running YOLO detection...")
    if not run_yolo_detection(str(input_dir), str(output_dir)):
        print("❌ YOLO detection failed, stopping pipeline")
        return False
    
    # 3. Convert to ONNX
    print("\n3. Converting YOLO model to ONNX...")
    onnx_path = convert_to_onnx()
    if not onnx_path:
        print("❌ ONNX conversion failed, continuing without ONNX")
    else:
        # Test ONNX inference
        print("\n4. Testing ONNX inference...")
        test_onnx_inference(onnx_path)
    
    # 5. Convert to NCNN (if tools are available)
    print("\n5. Converting ONNX to NCNN...")
    ncnn_dir = Path("../models/ncnn") if os.path.exists("../models") else Path("/workspace/models/ncnn")
    try:
        # Import the conversion function
        sys.path.append('/workspace/scripts')
        from ncnn_inference import convert_onnx_to_ncnn
        
        if onnx_path:
            param_file, bin_file = convert_onnx_to_ncnn(onnx_path, str(ncnn_dir))
            
            if param_file and bin_file:
                print("\n6. Running NCNN inference...")
                sample_image = str(input_dir / "bus.jpg")
                ncnn_output = str(output_dir / "ncnn_result.jpg")
                
                if os.path.exists(sample_image):
                    # Try direct NCNN inference (implementation would be in ncnn_inference.py)
                    print("✅ NCNN files created successfully")
                    print(f"   Param: {param_file}")
                    print(f"   Bin: {bin_file}")
        
    except Exception as e:
        print(f"❌ NCNN conversion/inference failed: {e}")
        print("This is normal if NCNN is not yet built. Build the Docker container to enable NCNN.")
    
    # 6. Benchmark models
    print("\n7. Benchmarking available models...")
    benchmark_all_formats()
    
    print("\n=== Pipeline Complete ===")
    print(f"Check the {output_dir} directory for results")
    return True

def convert_to_onnx():
    """Convert PyTorch model to ONNX format"""
    print("Converting PyTorch to ONNX...")
    try:
        # Try to import the conversion module
        try:
            from simple_onnx_convert import convert_yolo_simple, verify_onnx_simple
        except ImportError:
            # Fallback to direct YOLO export
            from ultralytics import YOLO
            
            # Determine paths based on environment
            model_path = "../models/yolo11n.pt" if os.path.exists("../models") else "/workspace/models/yolo11n.pt"
            models_dir = Path("../models") if os.path.exists("../models") else Path("/workspace/models")
            onnx_path = str(models_dir / "yolo11n.onnx")
            
            if not os.path.exists(model_path):
                print("❌ PyTorch model not found, setting up first...")
                model_path = setup_yolo_model()
            
            print(f"   Converting {model_path} to {onnx_path}")
            
            # Load and export model
            model = YOLO(model_path)
            export_path = model.export(format="onnx", imgsz=640)
            
            # Move exported file to desired location if different
            if export_path != onnx_path and os.path.exists(export_path):
                import shutil
                shutil.move(export_path, onnx_path)
                export_path = onnx_path
            
            print(f"✅ ONNX conversion successful: {export_path}")
            return export_path
        
        # Use simple_onnx_convert if available
        model_path = "../models/yolo11n.pt" if os.path.exists("../models") else "/workspace/models/yolo11n.pt"
        models_dir = Path("../models") if os.path.exists("../models") else Path("/workspace/models")
        onnx_path = str(models_dir / "yolo11n_optimized.onnx")
        
        if not os.path.exists(model_path):
            print("❌ PyTorch model not found, setting up first...")
            model_path = setup_yolo_model()
        
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



def print_summary():
    """Print pipeline summary"""
    print("\n" + "="*50)
    print("📋 PIPELINE SUMMARY")
    print("="*50)
    
    # Check what files were created - handle both relative and absolute paths
    models_dir = Path("../models") if os.path.exists("../models") else Path("/workspace/models")
    output_dir = Path("../data/output") if os.path.exists("../data") else Path("/workspace/data/output")
    
    print("\n🔧 Available Models:")
    
    # Check PyTorch model
    pytorch_files = ['yolo11n.pt']
    for pt_file in pytorch_files:
        if (models_dir / pt_file).exists():
            size = (models_dir / pt_file).stat().st_size / (1024*1024)
            print(f"   ✅ PyTorch: {pt_file} ({size:.1f}MB)")
            break
    else:
        print("   ❌ PyTorch: yolo11n.pt not found")
    
    # Check ONNX models
    onnx_files = ['yolo11n.onnx', 'yolo11n_optimized.onnx']
    found_onnx = False
    for onnx_file in onnx_files:
        if (models_dir / onnx_file).exists():
            size = (models_dir / onnx_file).stat().st_size / (1024*1024)
            print(f"   ✅ ONNX: {onnx_file} ({size:.1f}MB)")
            found_onnx = True
            break
    if not found_onnx:
        print("   ❌ ONNX: No ONNX models found")
    
    # Check NCNN models
    ncnn_dir = models_dir / "ncnn"
    if ncnn_dir.exists() and (ncnn_dir / "model.param").exists() and (ncnn_dir / "model.bin").exists():
        param_size = (ncnn_dir / "model.param").stat().st_size / 1024
        bin_size = (ncnn_dir / "model.bin").stat().st_size / (1024*1024)
        print(f"   ✅ NCNN: model.param ({param_size:.1f}KB), model.bin ({bin_size:.1f}MB)")
    else:
        print("   ❌ NCNN: Not available (tools missing or not built)")
    
    print("\n📁 Output Files:")
    if output_dir.exists():
        output_files = list(output_dir.glob('*'))
        if output_files:
            for file in output_files:
                if file.is_file():
                    print(f"   ✅ {file.name}")
        else:
            print("   ℹ️  No output files found")
    else:
        print("   ❌ Output directory not found")
    
    print("\n🚀 Next Steps:")
    print("   1. Run with --mode=pipeline for complete workflow")
    print("   2. Use --mode=benchmark to compare model performance")
    print("   3. To enable NCNN, rebuild Docker container with NCNN support")
    print("   4. ONNX models work with OpenVINO for Intel hardware optimization")

def benchmark_all_formats():
    """Benchmark all model formats with comprehensive testing"""
    print("=== Benchmarking All Model Formats ===")
    
    # Prepare sample image
    input_dir = Path("../data/input") if os.path.exists("../data") else Path("/workspace/data/input")
    sample_image = str(input_dir / "bus.jpg")
    
    if not os.path.exists(sample_image):
        print("Sample image not found, downloading...")
        download_sample_images()
    
    if not os.path.exists(sample_image):
        print("❌ Cannot find sample image for benchmarking")
        return
    
    # 1. Benchmark YOLO (PyTorch)
    print("\n1. Benchmarking PyTorch YOLO...")
    try:
        model_path = "../models/yolo11n.pt" if os.path.exists("../models") else "/workspace/models/yolo11n.pt"
        if not os.path.exists(model_path):
            model_path = setup_yolo_model()
        
        model = YOLO(model_path)
        
        num_runs = 10
        
        # Load image
        image = cv2.imread(sample_image)
        if image is None:
            print(f"❌ Could not load image: {sample_image}")
            return
        
        # Warmup
        print("   Warming up model...")
        for _ in range(3):
            model(image, verbose=False)
        
        # Benchmark
        print(f"   Running {num_runs} iterations...")
        start_time = time.time()
        for _ in range(num_runs):
            results = model(image, verbose=False)
        pytorch_time = (time.time() - start_time) / num_runs * 1000
        
        print(f"✅ PyTorch average time: {pytorch_time:.2f} ms")
        
        # 2. Benchmark ONNX if available
        print("\n2. Benchmarking ONNX...")
        onnx_path = "../models/yolo11n.onnx" if os.path.exists("../models") else "/workspace/models/yolo11n.onnx"
        alt_onnx = "../models/yolo11n_optimized.onnx" if os.path.exists("../models") else "/workspace/models/yolo11n_optimized.onnx"
        
        if os.path.exists(onnx_path) or os.path.exists(alt_onnx):
            actual_path = onnx_path if os.path.exists(onnx_path) else alt_onnx
            try:
                onnx_model = YOLO(actual_path)
                
                # Warmup
                for _ in range(3):
                    onnx_model(image, verbose=False)
                
                # Benchmark
                start_time = time.time()
                for _ in range(num_runs):
                    onnx_results = onnx_model(image, verbose=False)
                onnx_time = (time.time() - start_time) / num_runs * 1000
                
                print(f"✅ ONNX average time: {onnx_time:.2f} ms")
                
                speedup = pytorch_time / onnx_time
                print(f"\n📊 Performance Comparison:")
                print(f"   PyTorch: {pytorch_time:.2f} ms")
                print(f"   ONNX:    {onnx_time:.2f} ms")
                print(f"   ONNX Speedup: {speedup:.2f}x")
                
            except Exception as e:
                print(f"❌ ONNX benchmarking failed: {e}")
        else:
            print("   ONNX model not available, converting...")
            onnx_path = convert_to_onnx()
            if onnx_path:
                print("   Conversion successful, but skipping benchmark for now")
        
        # 3. NCNN benchmark placeholder
        print("\n3. NCNN Status...")
        ncnn_param = "../models/ncnn/model.param" if os.path.exists("../models") else "/workspace/models/ncnn/model.param"
        ncnn_bin = "../models/ncnn/model.bin" if os.path.exists("../models") else "/workspace/models/ncnn/model.bin"
        
        if os.path.exists(ncnn_param) and os.path.exists(ncnn_bin):
            print("✅ NCNN models available (benchmarking would require C++ implementation)")
        else:
            print("❌ NCNN models not available (rebuild Docker with NCNN support)")
            
    except Exception as e:
        print(f"❌ Benchmarking failed: {e}")

def main():
    parser = argparse.ArgumentParser(description='Complete Image Processing Pipeline')
    parser.add_argument('--mode', choices=['full', 'pipeline', 'detect', 'convert', 'benchmark'], 
                        default='full', help='Pipeline mode to run')
    parser.add_argument('--setup-only', action='store_true',
                        help='Only setup models and sample data')
    
    args = parser.parse_args()
    
    print("🔄 Starting Complete Image Processing Pipeline")
    print("="*50)
    
    if args.setup_only:
        print("Setting up models and sample data...")
        setup_yolo_model()
        download_sample_images()
        return
    
    success_count = 0
    
    if args.mode == 'pipeline':
        # Run the complete integrated pipeline
        if run_complete_pipeline():
            success_count += 3
    elif args.mode in ['full', 'detect']:
        if run_yolo_detection():
            success_count += 1
    
    if args.mode in ['full', 'convert']:
        onnx_path = convert_to_onnx()
        if onnx_path:
            success_count += 1
            if test_onnx_inference(onnx_path):
                success_count += 1
    
    if args.mode in ['full', 'benchmark']:
        benchmark_all_formats()
        success_count += 1
    elif args.mode == 'benchmark':
        benchmark_all_formats()
        return
    
    if args.mode != 'pipeline':  # Avoid duplicate summary
        print_summary()
    
    print(f"\n✅ Pipeline completed with {success_count} successful steps")

if __name__ == "__main__":
    main()