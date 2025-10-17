#!/usr/bin/env python3
"""
PyTorch to ONNX Model Conversion Script
This script demonstrates how to convert PyTorch models (especially YOLO) to ONNX format.
"""

import torch
import onnx
import onnxruntime
import numpy as np
import argparse
from pathlib import Path
from ultralytics import YOLO
import onnxsim

def convert_yolo_to_onnx(model_path, output_path, imgsz=640, batch_size=1, simplify=True):
    """Convert YOLO model to ONNX format"""
    print(f"Converting YOLO model: {model_path}")
    
    # Load YOLO model
    model = YOLO(model_path)
    
    # Export to ONNX
    onnx_path = model.export(
        format='onnx',
        imgsz=imgsz,
        batch=batch_size,
        simplify=simplify,
        opset=11
    )
    
    print(f"ONNX model saved to: {onnx_path}")
    
    # Verify the model
    verify_onnx_model(onnx_path)
    
    return onnx_path

def convert_pytorch_to_onnx(model, dummy_input, output_path, input_names=None, output_names=None, dynamic_axes=None):
    """Convert a generic PyTorch model to ONNX"""
    print(f"Converting PyTorch model to ONNX: {output_path}")
    
    # Set model to evaluation mode
    model.eval()
    
    # Default input/output names
    if input_names is None:
        input_names = ['input']
    if output_names is None:
        output_names = ['output']
    
    # Export the model
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes
    )
    
    print(f"ONNX model saved to: {output_path}")
    
    # Verify the model
    verify_onnx_model(output_path)
    
    return output_path

def simplify_onnx_model(onnx_path, output_path=None):
    """Simplify ONNX model using onnx-simplifier"""
    if output_path is None:
        base_path = Path(onnx_path)
        output_path = base_path.parent / f"{base_path.stem}_simplified{base_path.suffix}"
    
    print(f"Simplifying ONNX model: {onnx_path}")
    
    # Load and simplify
    model = onnx.load(onnx_path)
    model_simplified, check = onnxsim.simplify(model)
    
    if check:
        onnx.save(model_simplified, output_path)
        print(f"Simplified model saved to: {output_path}")
    else:
        print("Simplification failed!")
        return None
    
    return output_path

def verify_onnx_model(onnx_path):
    """Verify ONNX model and print model info"""
    print(f"Verifying ONNX model: {onnx_path}")
    
    try:
        # Load and check the model
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print("✓ ONNX model is valid")
        
        # Print model info
        print("\nModel Information:")
        print(f"  IR Version: {model.ir_version}")
        print(f"  Producer: {model.producer_name}")
        print(f"  Graph name: {model.graph.name}")
        
        # Print inputs
        print("\nInputs:")
        for input_tensor in model.graph.input:
            shape = [dim.dim_value for dim in input_tensor.type.tensor_type.shape.dim]
            print(f"  {input_tensor.name}: {shape}")
        
        # Print outputs
        print("\nOutputs:")
        for output_tensor in model.graph.output:
            shape = [dim.dim_value for dim in output_tensor.type.tensor_type.shape.dim]
            print(f"  {output_tensor.name}: {shape}")
        
        # Test with ONNX Runtime
        session = onnxruntime.InferenceSession(onnx_path)
        print("✓ ONNX Runtime can load the model")
        
    except Exception as e:
        print(f"✗ Model verification failed: {e}")
        return False
    
    return True

def benchmark_onnx_model(onnx_path, input_shape=(1, 3, 640, 640), num_runs=100):
    """Benchmark ONNX model inference speed"""
    print(f"Benchmarking ONNX model: {onnx_path}")
    
    # Create ONNX Runtime session
    session = onnxruntime.InferenceSession(onnx_path)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Create dummy input
    dummy_input = np.random.randn(*input_shape).astype(np.float32)
    
    # Warmup runs
    for _ in range(10):
        session.run(None, {input_name: dummy_input})
    
    # Benchmark
    import time
    start_time = time.time()
    for _ in range(num_runs):
        session.run(None, {input_name: dummy_input})
    end_time = time.time()
    
    avg_time = (end_time - start_time) / num_runs * 1000  # ms
    fps = 1000 / avg_time
    
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Estimated FPS: {fps:.1f}")
    
    return avg_time, fps

def main():
    parser = argparse.ArgumentParser(description='PyTorch to ONNX Conversion')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to PyTorch model (YOLO .pt file)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output ONNX file path')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size for YOLO models (default: 640)')
    parser.add_argument('--batch-size', type=int, default=1,
                        help='Batch size (default: 1)')
    parser.add_argument('--simplify', action='store_true',
                        help='Simplify the ONNX model')
    parser.add_argument('--benchmark', action='store_true',
                        help='Benchmark the converted model')
    
    args = parser.parse_args()
    
    # Convert model
    if args.model.endswith('.pt'):
        # YOLO model
        onnx_path = convert_yolo_to_onnx(
            args.model, 
            args.output, 
            imgsz=args.imgsz, 
            batch_size=args.batch_size,
            simplify=args.simplify
        )
    else:
        print("Only YOLO .pt models are supported in this example")
        return
    
    # Optional simplification
    if args.simplify and onnx_path:
        simplify_onnx_model(onnx_path)
    
    # Optional benchmarking
    if args.benchmark and onnx_path:
        benchmark_onnx_model(onnx_path, input_shape=(args.batch_size, 3, args.imgsz, args.imgsz))

if __name__ == "__main__":
    main()