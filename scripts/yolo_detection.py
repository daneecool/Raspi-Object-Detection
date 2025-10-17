#!/usr/bin/env python3
"""
YOLO11n Object Detection Script
This script demonstrates how to use YOLO11n for object detection on images.
"""

import os
import cv2
import argparse
from pathlib import Path
from ultralytics import YOLO
import time

def setup_yolo_model(model_path=None):
    """Initialize YOLO11n model"""
    if model_path and os.path.exists(model_path):
        print(f"Loading custom model from {model_path}")
        model = YOLO(model_path)
    else:
        print("Loading YOLO11n model (will download if not present)")
        model = YOLO('yolo11n.pt')  # This will download the model if not present
    return model

def detect_objects(model, image_path, output_path, conf_threshold=0.5):
    """Run object detection on a single image"""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not load image {image_path}")
        return None
    
    print(f"Processing image: {image_path}")
    
    # Run inference
    start_time = time.time()
    results = model(image, conf=conf_threshold)
    inference_time = time.time() - start_time
    
    print(f"Inference time: {inference_time:.3f} seconds")
    
    # Process results
    annotated_image = results[0].plot()
    
    # Save result
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
    
    return results

def batch_detect(model, input_dir, output_dir, conf_threshold=0.5):
    """Run object detection on all images in a directory"""
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported image extensions
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    # Find all image files
    image_files = [f for f in input_path.iterdir() 
                   if f.suffix.lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} images to process")
    
    for image_file in image_files:
        output_file = output_path / f"detected_{image_file.name}"
        detect_objects(model, str(image_file), str(output_file), conf_threshold)

def main():
    parser = argparse.ArgumentParser(description='YOLO11n Object Detection')
    parser.add_argument('--input', '-i', type=str, required=True,
                        help='Input image file or directory')
    parser.add_argument('--output', '-o', type=str, required=True,
                        help='Output image file or directory')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to custom YOLO model (optional)')
    parser.add_argument('--conf', '-c', type=float, default=0.5,
                        help='Confidence threshold (default: 0.5)')
    
    args = parser.parse_args()
    
    # Initialize model
    model = setup_yolo_model(args.model)
    
    # Check if input is a directory or single file
    if os.path.isdir(args.input):
        batch_detect(model, args.input, args.output, args.conf)
    elif os.path.isfile(args.input):
        detect_objects(model, args.input, args.output, args.conf)
    else:
        print(f"Error: Input path {args.input} does not exist")

if __name__ == "__main__":
    main()