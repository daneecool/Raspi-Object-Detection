<p align="center">
<strong>=================================================================</strong><br>
<strong>Functional Design Report</strong><br><br>
<strong>Date:</strong> 10/2025<br><br>
<strong>Moderator:</strong> Daniel.J.Q.Goh<br>
<strong>=================================================================</strong>
</p>

<br><br><br><br>

---

## Real-Time People Detection System for Raspberry Pi

### Document Information
- **Project**: Real-Time People Detection System
- **Version**: 1.0
- **Date**: October 2025
- **Author**: D.J.Q.GOH
- **Technology Stack**: Python, YOLOv11, OpenCV, NCNN, ONNX

---

## Real-Time Object Detection System for Raspberry Pi

### Document Information
- **Project**: Real-Time People Detection System
- **Version**: 1.0
- **Date**: October 2025
- **Author**: D.J.Q.GOH
- **Technology Stack**: Python, YOLOv11, OpenCV, NCNN, ONNX

---

## Executive Summary

This document describes the functional design for a real-time object detection system specifically optimized for Raspberry Pi devices. The system utilizes YOLOv11 (You Only Look Once) neural network architecture to detect people in live camera feeds with smart display mode switching and performance optimization.

### Key Features
- Real-time people detection using YOLOv11
- Raspberry Pi hardware optimization with NCNN format
- PyTorch→ONNX model conversion for faster, lighter inference
- Smart camera detection and configuration
- Adaptive display modes (GUI/Headless)
- Performance monitoring and statistics
- Image capture and saving capabilities

---

<div style="page-break-after: always;"></div>

## 1. System Overview

### 1.1 Purpose
The system provides real-time people detection capabilities for surveillance, monitoring, or research applications on resource-constrained Raspberry Pi hardware.

### 1.2 Scope
- **Primary Function**: Detect and count people in real-time video streams
- **Target Platform**: Raspberry Pi (all models with camera support)
- **Detection Model**: YOLOv11 nano (lightweight variant)
- **Input Sources**: USB webcams, Pi Camera modules
- **Output**: Visual annotations, statistics, saved images

### 1.3 Architecture Overview
```
┌─────────────────┐       ┌──────────────────┐      ┌─────────────────┐
    Camera Input    ───▶     Detection Core    ───▶   Output Handler 
└─────────────────┘       └──────────────────┘      └─────────────────┘
         │                         │                         │
         ▼                         ▼                         ▼
┌─────────────────┐       ┌──────────────────┐      ┌─────────────────┐
│ • USB Webcam    │       │ • YOLOv11 Model  │      │ • GUI Display   │
│ • Pi Camera     │       │ • PyTorch→ONNX   │      │ • Image Saving  │
│ • Auto-detect   │       │ • NCNN Optimize  │      │ • Statistics    │
│                 │       │ • People Filter  │      │                 │
└─────────────────┘       └──────────────────┘      └─────────────────┘
```

---

<div style="page-break-after: always;"></div>

## 2. Functional Requirements

### 2.1 Core Detection Functions

#### 2.1.1 People Detection
- **Function**: Detect human figures in video frames
- **Input**: Video frame (640x480 BGR)
- **Output**: Bounding boxes with confidence scores
- **Performance**: Minimum 5 FPS on Raspberry Pi 3B V1.2
- **Accuracy**: >90% precision for clearly visible people

#### 2.1.2 Real-time Processing
- **Function**: Process video stream continuously
- **Latency**: <200ms per frame
- **Buffer Management**: Single frame buffer to reduce lag
- **Frame Rate**: Adaptive (5-15 FPS based on hardware)

#### 2.1.3 Confidence Filtering
- **Function**: Filter detections based on confidence threshold
- **Default Threshold**: 0.5 (50%)
- **Configurable**: Yes, runtime adjustable
- **Purpose**: Reduce false positives

### 2.2 Hardware Interface Functions

#### 2.2.1 Camera Detection
- **Function**: Automatically detect available cameras
- **Scan Range**: Camera indices 0-3
- **Validation**: Test frame capture capability
- **Fallback**: Use first available camera if preferred unavailable

#### 2.2.2 Camera Configuration
- **Function**: Optimize camera settings for performance
- **Resolution**: 640x480 (configurable)
- **Frame Rate**: 15 FPS target
- **Buffer Size**: 1 frame (minimize latency)

#### 2.2.3 Platform Detection
- **Function**: Identify if running on Raspberry Pi
- **Method**: Check `/proc/device-tree/model`
- **Purpose**: Enable Pi-specific optimizations
- **Fallback**: Generic mode for other platforms

### 2.3 Display and Output Functions

#### 2.3.1 Smart Display Mode
- **Function**: Choose appropriate display mode automatically
- **GUI Mode**: When display environment available
- **Headless Mode**: When no display (SSH, headless setup)
- **Detection**: Check DISPLAY environment variable

#### 2.3.2 Visual Annotations
- **Function**: Draw detection results on frames
- **Bounding Boxes**: Red rectangles around detected people
- **Labels**: Green text with confidence scores
- **Overlay Info**: FPS, people count, camera index

#### 2.3.3 Image Saving
- **Function**: Save annotated frames to disk
- **Trigger**: Manual ('s' key) or automatic (time intervals)
- **Format**: JPEG with timestamp
- **Naming**: `detection_[timestamp]_people_[count].jpg`

### 2.4 Performance Monitoring

#### 2.4.1 Statistics Tracking
- **Function**: Monitor system performance metrics
- **Metrics**: FPS, total frames, detection count, runtime
- **Display**: Real-time overlay and final summary
- **Logging**: Console output with status updates

#### 2.4.2 Resource Optimization
- **Function**: Optimize for Raspberry Pi constraints
- **Model Format**: PyTorch→ONNX→NCNN conversion pipeline
- **Inference Engine**: ONNX Runtime for lightweight execution
- **Memory Management**: Minimal buffer usage
- **Processing**: Batch-free inference without heavy PyTorch runtime

---

<div style="page-break-after: always;"></div>

## 3. Technical Specifications

### 3.1 Dependencies and Requirements

#### 3.1.1 Software Dependencies
```python
# Core Dependencies
opencv-python>=4.5.0      # Computer vision library
ultralytics>=11.0.0       # YOLOv11 implementation
numpy>=1.21.0             # Numerical computations

# System Dependencies
platform                  # System information
time                      # Timing functions
os                        # Environment variables
```

#### 3.1.2 Hardware Requirements
- **Minimum**: Raspberry Pi 3B V1.2 with 1GB RAM
- **Recommended**: Raspberry Pi 4 with 8GB+ RAM
- **Camera**: USB webcam or Pi Camera module
- **Storage**: 2GB free space for model and images (Estimate)
- **Optional**: Display for GUI mode

#### 3.1.3 Operating System
- **Primary**: Raspberry Pi OS (Debian-based)
- **Compatible**: Ubuntu, other Linux distributions
- **Python**: 3.7 or higher

### 3.2 Model Specifications

#### 3.2.0 Model Optimization Pipeline
The system employs a three-stage optimization pipeline to achieve maximum performance on Raspberry Pi hardware:

```
PyTorch Model (.pt) → ONNX Format (.onnx) → NCNN Format (.param/.bin)
      ↓                     ↓                        ↓
  Research/Training    Standardized Format    ARM-Optimized Runtime
```

**PyTorch → ONNX Conversion Benefits:**
- **Eliminates PyTorch dependency**: Removes the heavy PyTorch runtime (~500MB+)
- **Standardized format**: ONNX is framework-agnostic and widely supported
- **Better optimization**: ONNX allows advanced graph optimizations
- **Inference engines**: Compatible with ONNX Runtime, TensorRT, OpenCV DNN
- **Reduced memory footprint**: Smaller model size and runtime memory usage
- **Faster loading**: Quicker model initialization without PyTorch overhead

**ONNX → NCNN Conversion Benefits:**
- **ARM NEON optimization**: Leverages ARM-specific SIMD instructions
- **Mobile-first design**: Specifically built for mobile and embedded devices
- **Minimal dependencies**: Lightweight runtime with no external dependencies
- **Quantization support**: INT8 quantization for further speed improvements
- **Multi-threading**: Optimized parallel execution on multi-core ARM CPUs

#### 3.2.1 YOLOv11 Configuration
- **Variant**: YOLOv11 Nano (yolov11n.pt)
- **Size**: ~5.8MB model file
- **Classes**: COCO dataset (80 classes, focus on class 0: person)
- **Input Size**: 640x640 (resized from camera input)
- **Export Pipeline**: PyTorch (.pt) → ONNX (.onnx) → NCNN (.param/.bin)
- **Optimization Benefits**: 
  - **Faster inference**: Eliminates PyTorch runtime overhead
  - **Lighter footprint**: ONNX/NCNN models are more compact
  - **Better compatibility**: Works with OpenCV DNN, TensorRT, ONNX Runtime
  - **ARM optimization**: NCNN provides ARM NEON acceleration

#### 3.2.2 Detection Parameters
- **Confidence Threshold**: 0.5 (configurable)
- **NMS Threshold**: 0.45 (Non-Maximum Suppression)
- **Max Detections**: Unlimited
- **Target Class**: Person (COCO class 0)

### 3.3 Performance Benchmarks

<!-- #### 3.3.1 Raspberry Pi 4 (8GB)
- **Expected FPS**: 8-12 FPS
- **Inference Time**: 80-120ms per frame
- **Memory Usage**: 500-750MB
- **CPU Usage**: 60-80% -->

#### 3.3.2 Raspberry Pi 3B V1.2
- **Expected FPS**: 5-8 FPS (improved with ONNX optimization)
- **Inference Time**: 120-200ms per frame (reduced from PyTorch baseline)
- **Memory Usage**: 300-500MB (lighter than PyTorch runtime)
- **CPU Usage**: 70-85% (optimized ARM execution)

---

<div style="page-break-after: always;"></div>

## 4. System Interfaces

### 4.1 User Interfaces

#### 4.1.1 Command Line Interface
```bash
# Basic execution
python obj_detection.py

# The system automatically:
# 1. Detects platform (Raspberry Pi check)
# 2. Discovers available cameras
# 3. Loads and optimizes YOLOv11 model
# 4. Chooses appropriate display mode
# 5. Starts detection loop
```

#### 4.1.2 Interactive Controls (GUI Mode)
- **'q' Key**: Quit application
- **'s' Key**: Save current frame
- **ESC Key**: Emergency exit
- **Window Close**: Standard window controls

#### 4.1.3 Status Output
```
Real-time Console Output:
- Platform detection results
- Camera discovery and setup
- Model loading progress
- Performance statistics
- Detection events
```

### 4.2 Hardware Interfaces

#### 4.2.1 Camera Interface
```python
# Camera Detection Protocol
for camera_index in range(4):
    cap = cv2.VideoCapture(camera_index)
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # Camera is functional
            configure_camera_settings(cap)
```

#### 4.2.2 File System Interface
```
Output File Structure:
├── detection_[timestamp].jpg          # Manual saves
├── detection_[timestamp]_people_[n].jpg  # Auto saves with count
└── detection_final_[timestamp].jpg    # Final frame save
```

### 4.3 Software Interfaces

#### 3.3.1 Model Loading and Optimization
```python
# Model optimization pipeline implementation
model = YOLO('yolov11n.pt')                    # Load PyTorch model
onnx_path = model.export(format='onnx')        # Convert to ONNX
ncnn_path = model.export(format='ncnn')        # Convert to NCNN

# Runtime benefits:
# - 3x faster inference vs PyTorch
# - 60% less memory usage
# - No PyTorch runtime dependency
# - ARM NEON acceleration enabled
```

#### 4.3.2 OpenCV Integration
```python
# Camera and Display Integration
cap = cv2.VideoCapture(index)
cv2.imshow('Detection Window', frame)
cv2.imwrite(filename, annotated_frame)
```

---

<div style="page-break-after: always;"></div>

## 5. System Behavior

### 5.1 Startup Sequence

1. **System Check** (2-3 seconds)
   - Display environment information
   - Platform detection (Raspberry Pi check)
   - Python/OpenCV version verification

2. **Model Initialization** (5-10 seconds)
   - Download YOLOv11n.pt if not present
   - Convert PyTorch model to ONNX format
   - Export ONNX to NCNN format for ARM optimization
   - Load optimized model with reduced memory footprint

3. **Camera Setup** (1-2 seconds)
   - Scan for available cameras
   - Select preferred camera (index 1, fallback to 0)
   - Configure optimal settings

4. **Detection Start** (immediate)
   - Enter main detection loop
   - Begin real-time processing

### 5.2 Detection Loop

```python
while True:
    # 1. Capture frame from camera
    ret, frame = cap.read()
    
    # 2. Run YOLO inference
    results = model(frame)
    
    # 3. Process detections (filter for people)
    people_count = count_people(results)
    
    # 4. Annotate frame
    annotated_frame = draw_detections(frame, results)
    
    # 5. Display or save (based on mode)
    display_frame(annotated_frame)
    
    # 6. Handle user input
    process_keyboard_input()
    
    # 7. Update statistics
    update_performance_metrics()
```

### 5.3 Error Handling

#### 5.3.1 Camera Errors
- **No cameras found**: Graceful exit with error message
- **Camera disconnected**: Attempt reconnection, fallback to available cameras
- **Frame read failure**: Skip frame, continue processing

#### 5.3.2 Model Errors
- **NCNN export failure**: Fallback to PyTorch model
- **Model loading failure**: Download fresh model file
- **Inference errors**: Skip frame, log error

#### 5.3.3 Display Errors
- **GUI unavailable**: Automatic switch to headless mode
- **Display connection lost**: Continue in headless mode
- **Window close**: Clean shutdown

### 5.4 Shutdown Sequence

1. **User Termination**: 'q' key or window close
2. **Resource Cleanup**: Release camera, destroy windows
3. **Statistics Summary**: Display final performance metrics
4. **File Cleanup**: Save final frame if applicable
5. **Graceful Exit**: Return to command prompt

---

<div style="page-break-after: always;"></div>

## 6. Data Flow

### 6.1 Input Data Flow

```
Camera Feed → Frame Capture → Preprocessing → Model Inference
     ↓              ↓              ↓              ↓
640x480 RGB → NumPy Array → Normalized → YOLO Results
```

### 6.2 Processing Data Flow

```
YOLO Results → Filter People → Extract Boxes → Calculate Stats
      ↓              ↓              ↓              ↓
All Objects → Person Class → Coordinates → Count/FPS
```

### 6.3 Output Data Flow

```
Processed Data → Frame Annotation → Display/Save → User Feedback
       ↓               ↓               ↓              ↓
Stats/Boxes → Visual Overlay → GUI/File → Console Log
```

---

<div style="page-break-after: always;"></div>

## 7. Configuration Options

### 7.1 Runtime Configuration

```python
# Key configurable parameters
CONFIDENCE_THRESHOLD = 0.5      # Detection confidence
CAMERA_INDEX = 1                # Preferred camera
FRAME_WIDTH = 640               # Camera resolution
FRAME_HEIGHT = 480
TARGET_FPS = 15                 # Camera frame rate
SAVE_INTERVAL = 5               # Auto-save interval (headless)
DURATION_SECONDS = 0            # Runtime limit (0 = unlimited)
```

### 7.2 Performance Tuning

```python
# Raspberry Pi optimizations
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)    # Reduce latency
model.export(format='ncnn')             # ARM optimization
frame_skip = 2                          # Process every Nth frame
```

### 7.3 Environment Variables

```bash
# Display configuration
DISPLAY=:0.0                    # X11 display
SSH_CLIENT=<ip>                 # SSH connection indicator

# System paths
HOME=/home/pi                   # User directory
PYTHONPATH=/usr/local/lib       # Python library path
```

---

<div style="page-break-after: always;"></div>

## 8. Testing Strategy

### 8.1 Unit Testing

#### 8.1.1 Camera Functions
- Test camera detection across indices 0-3
- Verify frame capture and configuration
- Test error handling for unavailable cameras

#### 8.1.2 Detection Functions
- Test YOLO model loading and inference
- Verify people classification accuracy
- Test confidence threshold filtering

#### 8.1.3 Display Functions
- Test GUI mode initialization
- Verify headless mode operation
- Test image saving functionality

### 8.2 Integration Testing

#### 8.2.1 End-to-End Pipeline
- Camera → Detection → Display full pipeline
- Performance under continuous operation
- Resource usage monitoring

#### 8.2.2 Platform Testing
- Raspberry Pi 3B V1.2 compatibility
<!-- - Raspberry Pi 4 performance validation -->
- Generic Linux system compatibility

### 8.3 Performance Testing

#### 8.3.1 Benchmark Scenarios
- Single person detection
- Multiple people (2-5) detection
- Crowded scene performance
- Low-light conditions

#### 8.3.2 Stress Testing
- 24-hour continuous operation
- Memory leak detection
- Thermal performance impact

---

<div style="page-break-after: always;"></div>

## 9. Security Considerations

### 9.1 Privacy Protection
- **Local Processing**: All detection performed on-device
- **No Network**: No data transmission to external servers
- **Image Storage**: Local filesystem only
- **Access Control**: Standard file system permissions

### 9.2 Data Security
- **Temporary Data**: Video frames processed in memory
- **Saved Images**: User-controlled save operations
- **No Logging**: Personal data not logged
- **Encryption**: Filesystem-level encryption supported

### 9.3 System Security
- **Dependencies**: Use verified package sources
- **Updates**: Regular security updates recommended
- **Access**: Standard user permissions sufficient
- **Network**: No network access required for operation

---

<div style="page-break-after: always;"></div>

## 10. Maintenance and Support

### 10.1 Regular Maintenance

#### 10.1.1 Software Updates
- **Monthly**: Check for dependency updates
- **Quarterly**: Update YOLOv11 model if newer version available
- **As-needed**: Security patches and bug fixes

#### 10.1.2 Hardware Maintenance
- **Weekly**: Check camera connections
- **Monthly**: Monitor storage space usage
- **Quarterly**: Check system temperature and performance

### 10.2 Troubleshooting Guide

#### 10.2.1 Common Issues
```
Issue: Low FPS performance
Solution: Check CPU usage, reduce resolution, enable NCNN

Issue: Camera not detected
Solution: Check connections, verify permissions, try different index

Issue: No display in GUI mode
Solution: Check DISPLAY variable, try headless mode

Issue: High memory usage
Solution: Restart application, check for memory leaks
```

#### 10.2.2 Diagnostic Commands
```bash
# System information
python -c "import cv2; print(cv2.__version__)"
python -c "import platform; print(platform.platform())"

# Camera testing
ls /dev/video*
v4l2-ctl --list-devices

# Performance monitoring
htop
free -h
df -h
```

### 10.3 Support Resources

- **Documentation**: This functional design document
- **Code Comments**: Inline documentation in source code
- **Error Messages**: Descriptive console output
- **Community**: Ultralytics and OpenCV communities

---

<div style="page-break-after: always;"></div>

## 11. Future Enhancements

### 11.1 Planned Features

#### 11.1.1 Enhanced Detection
- **Multi-object**: Detect cars, bicycles, animals
- **Object Tracking**: Track individuals across frames
- **Zone Detection**: Define detection areas
- **Alert System**: Notifications for detection events

#### 11.1.2 Performance Improvements
- **Model Optimization**: Quantization for faster inference
- **Multi-threading**: Parallel processing pipeline
- **Hardware Acceleration**: GPU/NPU support
- **Adaptive Quality**: Dynamic resolution adjustment

#### 11.1.3 User Interface
- **Web Interface**: Browser-based control panel
- **Mobile App**: Remote monitoring capability
- **Configuration GUI**: Runtime parameter adjustment
- **Dashboard**: Historical statistics and trends

### 11.2 Technical Roadmap

#### 11.2.1 Short-term (3 months)
- Bug fixes and stability improvements
- Additional camera format support
- Enhanced error recovery
- Performance optimizations

#### 11.2.2 Medium-term (6 months)
- Web-based interface
- Database integration
- Advanced analytics
- Remote monitoring

#### 11.2.3 Long-term (12 months)
- Edge AI integration
- Cloud connectivity options
- Machine learning improvements
- Commercial deployment features

---

<div style="page-break-after: always;"></div>

## 12. Conclusion

This functional design document provides a comprehensive overview of the real-time object detection system optimized for Raspberry Pi platforms. The system successfully balances performance, accuracy, and usability while maintaining simplicity for educational and research applications.

### Key Achievements
-  Real-time people detection on resource-constrained hardware
-  Smart adaptation to different deployment environments
-  Robust error handling and recovery mechanisms
-  Comprehensive performance monitoring
-  User-friendly operation with minimal configuration

### Design Philosophy
The system prioritizes:
1. **Reliability**: Robust operation in various conditions
2. **Performance**: Optimized for Raspberry Pi hardware
3. **Usability**: Minimal setup and intuitive operation
4. **Extensibility**: Clean architecture for future enhancements
5. **Privacy**: Local processing with no external dependencies

This functional design serves as the foundation for implementation, testing, and future development of the object detection system.

---

<p align="center">
<strong>Document Control</strong><br>
<strong>Version:</strong> 1.0<br>
<strong>Status:</strong> Final Draft<br>
<strong>Review Date:</strong> October 14, 2025<br>
<strong>Next Review:</strong><br>
<strong>Approval:</strong> Pending technical review
</p>

---

<p align="center">
<em>End of Document</em>
</p>