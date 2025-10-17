#!/usr/bin/env python3
"""
Simple PyTorch to ONNX Converter (without onnxscript dependency)
This script provides a simpler ONNX conversion method that avoids version conflicts.
"""

import argparse
import onnx
import onnxruntime
from ultralytics import YOLO

def convert_yolo_simple(model_path, output_path, imgsz=640):
    """Simple YOLO to ONNX conversion without onnxscript"""
    print(f"Converting YOLO model: {model_path}")
    
    # Load YOLO model
    model = YOLO(model_path)
    
    # Try export with minimal dependencies
    try:
        # First try with opset 11 (most compatible)
        onnx_path = model.export(
            format='onnx',
            imgsz=imgsz,
            opset=11,
            simplify=False,  # Disable simplification to avoid onnxscript
            dynamic=False,   # Disable dynamic shapes
        )
        
        print(f"✅ ONNX model saved to: {onnx_path}")
        return onnx_path
        
    except Exception as e1:
        print(f"First attempt failed: {e1}")
        
        # Try with even older opset
        try:
            onnx_path = model.export(
                format='onnx',
                imgsz=imgsz,
                opset=9,  # Very compatible opset
                simplify=False,
                dynamic=False,
            )
            print(f"✅ ONNX model saved to: {onnx_path} (opset 9)")
            return onnx_path
            
        except Exception as e2:
            print(f"Second attempt failed: {e2}")
            print("❌ ONNX export failed with both opset 11 and 9")
            return None

def verify_onnx_simple(onnx_path):
    """Simple ONNX verification"""
    try:
        # Load model
        model = onnx.load(onnx_path)
        onnx.checker.check_model(model)
        print(f"✅ ONNX model is valid: {onnx_path}")
        
        # Test with ONNX Runtime
        session = onnxruntime.InferenceSession(onnx_path)
        input_name = session.get_inputs()[0].name
        input_shape = session.get_inputs()[0].shape
        print("✅ ONNX Runtime can load the model")
        print(f"   Input: {input_name} {input_shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ ONNX verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Simple PyTorch to ONNX Conversion')
    parser.add_argument('--model', '-m', type=str, required=True,
                        help='Path to PyTorch model (YOLO .pt file)')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output ONNX file path')
    parser.add_argument('--imgsz', type=int, default=640,
                        help='Image size (default: 640)')
    
    args = parser.parse_args()
    
    if not args.model.endswith('.pt'):
        print("❌ Only YOLO .pt models are supported")
        return
    
    # Convert model
    onnx_path = convert_yolo_simple(args.model, args.output, args.imgsz)
    
    if onnx_path:
        # Verify the model
        verify_onnx_simple(onnx_path)
        
        # Move to desired output path if different
        if onnx_path != args.output:
            import shutil
            shutil.move(onnx_path, args.output)
            print(f"✅ Moved ONNX model to: {args.output}")
    else:
        print("❌ ONNX conversion failed")

if __name__ == "__main__":
    main()