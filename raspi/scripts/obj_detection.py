# from ultralytics import YOLO (This is use to train model)

# # Load a COCO-pretrained YOLO11n model
# model = YOLO("yolo11n.pt")

# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=320)

# # Run inference with the YOLO11n model on the 'bus.jpg' image
# results = model("test.jpg")

# ------------------------------------------------------------------ #

# Import required libraries
import cv2
import numpy as np
from ultralytics import YOLO
import time
import platform
import os

print(f"OpenCV version: {cv2.__version__}")
print(f"Platform: {platform.platform()}")
print(f"Python version: {platform.python_version()}")

#------------------------------------------------------------------- #

# Check if running on Raspberry Pi
def is_raspberry_pi():
    try:
        # Check /proc/device-tree/model
        try:
            with open('/proc/device-tree/model', 'r') as f:
                model = f.read().strip('\x00')
                if 'Raspberry Pi' in model:
                    return True, model
        except:
            pass
    except:
        pass
    return False, "Not detected"

is_pi, detection_info = is_raspberry_pi()
if is_pi:
    print(f"  Running on Raspberry Pi")
    print(f"  Device: {detection_info}")
else:
    print("  Not detected as Raspberry Pi - code will still work")
    print(f"  Detection result: {detection_info}")

# Download and load YOLOv8 model with NCNN export
print("Loading YOLO11n model...")

# Load YOLOv8 nano model (lightweight for Raspberry Pi)
model = YOLO('yolo11n.pt')

# Export to NCNN format for optimized inference on Raspberry Pi
print("Exporting model to NCNN format for Raspberry Pi optimization...")
try:
    ncnn_model = model.export(format='ncnn')
    print(" Model exported to NCNN format successfully")
    
    # Load the NCNN model
    model_ncnn = YOLO(ncnn_model)
    print(" NCNN model loaded successfully")
except Exception as e:
    print(f" NCNN export failed: {e}")
    print("Using PyTorch model instead")
    model_ncnn = model

# ---------------------------------------------------------- #

# Camera setup and detection function
def setup_camera(camera_index=1):
    """
    Setup USB webcam (typically camera index 1 on Raspberry Pi)
    """
    print(f"Attempting to connect to camera {camera_index}...")
    
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f" Camera {camera_index} not available, trying camera 0...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise ValueError("No camera found!")
        camera_index = 0
    
    # Optimize camera settings for Raspberry Pi
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)  # Lower FPS for better performance
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer lag
    
    print(f"Check Camera {camera_index} connected successfully")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    
    return cap, camera_index

def detect_people(frame, model, confidence_threshold=0.5):
    """
    Detect people in frame using YOLO model
    """
    results = model(frame, verbose=False)
    
    people_count = 0
    annotated_frame = frame.copy()
    
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                # Class 0 is 'person' in COCO dataset
                if int(box.cls[0]) == 0 and float(box.conf[0]) > confidence_threshold:
                    people_count += 1
                    
                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    
                    # Draw bounding box in RED
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
                    # Add label in GREEN
                    label = f"Person: {confidence:.2f}"
                    cv2.putText(annotated_frame, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return annotated_frame, people_count

# ----------------------------------------------------------------------------------------- #

# Main detection loop
def run_people_detection(duration_seconds=0, camera_index=0):
    """
    Run real-time people detection
    
    Args:
        duration_seconds: How long to run detection (0 for indefinite)
        camera_index: Camera index (0 for USB webcam, 1 for Pi camera)
    """
    try:
        # Setup camera
        cap, actual_camera_index = setup_camera(camera_index)
        
        print(f"\n Starting people detection...")
        print(f"Using camera {actual_camera_index}")
        print("Press 'q' to quit, 's' to save current frame")
        print("-" * 50)
        
        start_time = time.time()
        frame_count = 0
        total_people_detected = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Perform detection
            annotated_frame, people_count = detect_people(frame, model_ncnn)
            
            # Update statistics
            frame_count += 1
            total_people_detected += people_count
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Add performance info to frame
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"People: {people_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Camera: {actual_camera_index}", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow('People Detection - Raspberry Pi', annotated_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n Stopping detection...")
                break
            elif key == ord('s'):
                filename = f"detection_frame_{int(time.time())}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f"Frame saved as {filename}")
            
            # Check duration limit
            if duration_seconds > 0 and elapsed_time > duration_seconds:
                print(f"\n Duration limit ({duration_seconds}s) reached")
                break
        
        # Cleanup and statistics
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"\n   Detection Statistics:")
        print(f"   + Total frames processed: {frame_count}")
        print(f"   + Average FPS: {fps:.2f}")
        print(f"   + Total people detections: {total_people_detected}")
        print(f"   + Runtime: {elapsed_time:.1f} seconds")
        
    except Exception as e:
        print(f"Error during detection: {e}")
        cv2.destroyAllWindows()

# Test camera connection first
print("Testing camera connection...")
try:
    test_cap, camera_idx = setup_camera(1)  # Try USB webcam first
    test_cap.release()
    print(f" Camera test successful - will use camera {camera_idx}")
except Exception as e:
    print(f" Camera test failed: {e}")
    print("Please check your USB webcam connection")

# --------------------------------------------------------------------------------------------- #

# Display Environment Check and Headless Mode
def check_display_environment():
    """Check if we have a display environment available"""
    print(" Checking display environment...")
    
    # Check DISPLAY variable
    display = os.environ.get('DISPLAY')
    print(f"DISPLAY environment variable: {display}")
    
    # Check if we're in SSH
    ssh_client = os.environ.get('SSH_CLIENT')
    ssh_connection = os.environ.get('SSH_CONNECTION')
    print(f"SSH_CLIENT: {ssh_client}")
    print(f"SSH_CONNECTION: {ssh_connection}")
    
    # Check desktop environment
    desktop = os.environ.get('DESKTOP_SESSION')
    xdg_session = os.environ.get('XDG_SESSION_TYPE')
    print(f"Desktop session: {desktop}")
    print(f"XDG session type: {xdg_session}")
    
    return display is not None

def run_headless_detection(duration_seconds=0, camera_index=0, save_interval=5):
    """
    Run detection without GUI - saves images instead of displaying
    
    Args:
        duration_seconds: How long to run detection
        camera_index: Camera index (0 for USB webcam, 1 for Pi camera)
        save_interval: Save an image every N seconds
    """
    cap = None
    annotated_frame = None
    fps = 0
    elapsed_time = 0
    
    try:
        # Setup camera
        cap, actual_camera_index = setup_camera(camera_index)
        
        print("\n Starting HEADLESS people detection...")
        print(f" Using camera {actual_camera_index}")
        print(f" Saving images every {save_interval} seconds")
        print("-" * 50)
        
        start_time = time.time()
        frame_count = 0
        total_people_detected = 0
        last_save_time = start_time
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read from camera")
                break
            
            # Perform detection
            annotated_frame, people_count = detect_people(frame, model_ncnn)
            
            # Update statistics
            frame_count += 1
            total_people_detected += people_count
            
            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            # Add performance info to frame
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"People: {people_count}", (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(annotated_frame, f"Time: {elapsed_time:.1f}s", (10, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Save image periodically
            current_time = time.time()
            if current_time - last_save_time >= save_interval:
                filename = f"detection_{int(current_time)}_people_{people_count}.jpg"
                cv2.imwrite(filename, annotated_frame)
                print(f" Saved: {filename} (People: {people_count}, FPS: {fps:.1f})")
                last_save_time = current_time
            
            # Print status every 30 frames
            if frame_count % 30 == 0:
                print(f"  Frame {frame_count}, People: {people_count}, FPS: {fps:.1f}")
            
            # Check duration limit
            if duration_seconds > 0 and elapsed_time > duration_seconds:
                print(f"\n Duration limit ({duration_seconds}s) reached")
                break
        
        # Save final frame if we have one
        if annotated_frame is not None:
            final_filename = f"detection_final_{int(time.time())}.jpg"
            cv2.imwrite(final_filename, annotated_frame)
            print(f"   + Final frame saved as: {final_filename}")
        
        # Cleanup and statistics
        if cap is not None:
            cap.release()
        
        print("\n    Detection Statistics:")
        print(f"   • Total frames processed: {frame_count}")
        print(f"   • Average FPS: {fps:.2f}")
        print(f"   • Total people detections: {total_people_detected}")
        print(f"   • Runtime: {elapsed_time:.1f} seconds")
        
    except Exception as e:
        print(f" Error during detection: {e}")
        if cap is not None:
            cap.release()

def run_smart_detection(duration_seconds=30, camera_index=1):
    """
    Automatically choose between GUI and headless mode based on environment
    """
    has_display = check_display_environment()
    
    if has_display:
        print(" Display environment detected - Using GUI mode")
        try:
            # Test if cv2.imshow works
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('Display Test', test_img)
            cv2.waitKey(100)
            cv2.destroyAllWindows()
            print(" OpenCV display test successful")
            
            # Run normal detection with GUI
            run_people_detection(duration_seconds, camera_index)
            
        except Exception as e:
            print(f" GUI mode failed: {e}")
            print(" Falling back to headless mode...")
            run_headless_detection(duration_seconds, camera_index)
    else:
        print("  No display environment - Using headless mode")
        run_headless_detection(duration_seconds, camera_index)

# --------------------------------------------------------------------------------------------- #

# AUTO-START THE DETECTION
print("\n" + "="*60)
print(" STARTING OBJECT DETECTION")
print("="*60)

# Run the smart detection that auto-detects best mode
if __name__ == "__main__":
    # Check display environment first
    check_display_environment()
    
    print("\n Available modes:")
    print("1. Smart Detection (Auto-detect best mode)")
    print("2. GUI Mode (Force window display)")
    print("3. Headless Mode (Save images only)")
    
    # Automatically start smart detection
    print("\n Auto-starting Smart Detection...")
    run_smart_detection(duration_seconds=0, camera_index=1) # default time seconds=0(no time limit), can be adjust, eg: seconds=30, 30sec later 'exit' command will be execute and quit the entire program