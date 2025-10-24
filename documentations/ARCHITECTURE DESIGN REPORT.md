<p align="center">
<strong>=================================================================</strong><br>
<strong>Architecture Design Document</strong><br><br>
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

## 1. System Overview

### 1.1 Purpose
The Real-Time People Detection System is designed to provide efficient, real-time human detection capabilities on Raspberry Pi hardware using state-of-the-art computer vision techniques.

### 1.2 Key Objectives
- **Performance**: Achieve real-time detection (10-15 FPS) on Raspberry Pi 3B V1.2
- **Accuracy**: Reliable people detection with configurable confidence thresholds
- **Flexibility**: Support both GUI and headless deployment modes
- **Efficiency**: Optimize resource usage through PyTorch→ONNX→NCNN pipeline acceleration
- **Usability**: Provide intuitive controls and automated environment detection

---

<div style="page-break-after: always;"></div>

## 2. System Architecture

### 2.1 High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    User Interface Layer                          │
├──────────────────────────────────────────────────────────────────┤
│  GUI Mode              │  Headless Mode     │  Smart Detection   │
│  ┌─────────────────┐   │  ┌──────────────┐  │  ┌─────────────┐   │
│  │ Live Video      │   │  │ Image Saves  │  │  │ Auto-Select │   │
│  │ Display         │   │  │ Periodic     │  │  │ Best Mode   │   │
│  │ Keyboard Input  │   │  │ Console Log  │  │  │             │   │
│  └─────────────────┘   │  └──────────────┘  │  └─────────────┘   │
├──────────────────────────────────────────────────────────────────┤
│                    Application Logic Layer                       │
├──────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌──────────┐ │
│  │ Detection   │  │ Camera      │  │ Display     │  │ Config   │ │
│  │ Engine      │  │ Management  │  │ Environment │  │ Manager  │ │
│  │             │  │             │  │ Detection   │  │          │ │
│  └─────────────┘  └─────────────┘  └─────────────┘  └──────────┘ │
├──────────────────────────────────────────────────────────────────┤
│                    AI/ML Processing Layer                        │
├──────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │ YOLOv11     │  │ PyTorch→ONNX│  │ Post-       │               │
│  │ Model       │  │ →NCNN       │  │ Processing  │               │
│  │ Loading     │  │ Pipeline    │  │ & Filtering │               │
│  └─────────────┘  └─────────────┘  └─────────────┘               │
├──────────────────────────────────────────────────────────────────┤
│                    Hardware Abstraction Layer                    │
├──────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐               │
│  │ Camera      │  │ Display     │  │ File System │               │
│  │ Interface   │  │ Interface   │  │ I/O         │               │
│  │ (OpenCV)    │  │ (OpenCV)    │  │             │               │
│  └─────────────┘  └─────────────┘  └─────────────┘               │
└──────────────────────────────────────────────────────────────────┘
```

<div style="page-break-after: always;"></div>

### 2.2 Component Architecture

#### 2.2.1 Core Components

```
PeopleDetectionSystem (Main Class)
├── Initialization & Setup
│   ├── print_system_info()
│   ├── detect_raspberry_pi()
│   └── load_model()
├── Camera Management
│   └── setup_camera()
├── Detection Engine
│   └── detect_people_in_frame()
├── Environment Management
│   └── check_display_environment()
├── Execution Modes
│   ├── run_gui_detection()
│   ├── run_headless_detection()
│   └── run_smart_detection()
└── Main Orchestrator
    └── run()
```

---

<div style="page-break-after: always;"></div>

## 3. Detailed Component Design

### 3.1 Detection Engine

#### 3.1.1 YOLOv11 Model Pipeline
```
Input Frame (640x480)
        ↓
[ Preprocessing ]
        ↓
[ YOLOv11 Inference ]
        ↓
[ PyTorch→ONNX→NCNN Optimization ]
        ↓
[ Detection Results ]
        ↓
[ Post-processing ]
        ↓
[ Person Filtering ]
        ↓
[ Bounding Box Drawing ]
        ↓
Output Frame (Annotated)
```

#### 3.1.2 Model Optimization Strategy
- **Primary**: PyTorch→ONNX→NCNN conversion pipeline for maximum ARM optimization
- **ONNX Benefits**: Eliminates heavy PyTorch runtime, enables faster inference
- **NCNN Benefits**: ARM NEON acceleration, mobile-optimized execution
- **Fallback**: PyTorch backend if conversion fails
- **Model Size**: Nano variant for resource efficiency
- **Precision**: FP16 inference when available

<div style="page-break-after: always;"></div>

### 3.1.3 Model Conversion Pipeline

The system employs a sophisticated three-stage optimization pipeline specifically designed for ARM-based edge devices:

```
PyTorch Model (.pt) → ONNX Format (.onnx) → NCNN Format (.param/.bin)
      ↓                     ↓                        ↓
  Training Format      Inference Format         ARM-Optimized
```

**Stage 1: PyTorch → ONNX**
- **Purpose**: Convert from training format to inference-optimized format
- **Benefits**: 
  - Eliminates 500MB+ PyTorch runtime dependency
  - Creates framework-agnostic representation
  - Enables graph-level optimizations
  - Reduces memory footprint by 60%
  - Compatible with multiple inference engines (ONNX Runtime, TensorRT, OpenCV DNN)

**Stage 2: ONNX → NCNN**
- **Purpose**: Convert to ARM-optimized mobile inference format
- **Benefits**:
  - ARM NEON SIMD instruction utilization
  - Mobile-first architecture design
  - Minimal external dependencies
  - INT8 quantization support
  - Multi-threaded ARM CPU optimization
  - 3x faster inference compared to PyTorch on ARM

**Performance Impact:**
- **Inference Speed**: 150-300ms → 80-120ms per frame
- **Memory Usage**: 400-750MB → 300-500MB
- **CPU Efficiency**: Better ARM instruction utilization
- **Startup Time**: Faster model loading without PyTorch

<div style="page-break-after: always;"></div>

### 3.2 Camera Management System

#### 3.2.1 Camera Discovery Flow
```
Camera Setup Request
        ↓
Try Camera Index 1 (USB)
        ↓
    Success? ──No──→ Try Camera Index 0 (Pi Camera)
        ↓Yes                    ↓
Configure Settings         Success? ──No──→ Raise Error
        ↓                       ↓Yes
Return Camera Object    Configure Settings
                               ↓
                       Return Camera Object
```

<div style="page-break-after: always;"></div>

#### 3.2.2 Camera Configuration
- **Resolution**: 640x480 (optimized for Pi performance)
- **FPS**: 15 (balanced performance/quality)
- **Buffer Size**: 1 (reduced latency)
- **Format**: BGR (OpenCV native)

### 3.3 Display Environment Detection

#### 3.3.1 Environment Detection Logic
```
Check Display Environment
        ↓
┌─────────────────────────┐
│ Environment Variables   │
│ - DISPLAY               │
│ - SSH_CLIENT            │
│ - DESKTOP_SESSION       │
│ - XDG_SESSION_TYPE      │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│ OpenCV Display Test     │
│ - Create test window    │
│ - Verify functionality  │
│ - Clean up resources    │
└─────────────────────────┘
        ↓
Decision: GUI vs Headless
```

---

<div style="page-break-after: always;"></div>

## 4. Data Flow Architecture

### 4.1 Real-Time Processing Pipeline

```
Camera Capture
        ↓
Frame Buffer (1 frame)
        ↓
┌─────────────────────────┐
│ Detection Processing    │
│ ┌─────────────────────┐ │
│ │ YOLOv11 Inference   │ │
│ │ NCNN Acceleration   │ │
│ │ Person Detection    │ │
│ │ Confidence Filter   │ │
│ └─────────────────────┘ │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│ Annotation Processing   │
│ ┌─────────────────────┐ │
│ │ Bounding Boxes      │ │
│ │ Confidence Labels   │ │
│ │ Performance Stats   │ │
│ │ Frame Information   │ │
│ └─────────────────────┘ │
└─────────────────────────┘
        ↓
┌─────────────────────────┐
│ Output Processing       │
│ ┌─────────────────────┐ │
│ │ GUI: Display Window │ │
│ │ Headless: Save File │ │
│ │ Statistics: Console │ │
│ └─────────────────────┘ │
└─────────────────────────┘
```

<div style="page-break-after: always;"></div>

### 4.2 Memory Management

#### 4.2.1 Memory Usage Profile
- **Model Loading**: ~200MB (YOLOv11n + NCNN)
- **Frame Buffers**: ~5MB (multiple frame copies)
- **Processing**: ~50MB (inference workspace)
- **Application**: ~20MB (Python runtime)
- **Total Estimated**: ~275MB peak usage

#### 4.2.2 Memory Optimization Strategies
- Single frame buffer to minimize copies
- Immediate release of processed frames
- PyTorch→ONNX conversion eliminates heavy runtime dependencies
- NCNN quantization for reduced model size
- Efficient numpy array management

---

<div style="page-break-after: always;"></div>

## 5. Performance Architecture

### 5.1 Performance Optimization Layers

#### 5.1.1 Hardware Optimization
```
Raspberry Pi 3B V1.2 (ARM Cortex-A53)
        ↓
PyTorch Model (.pt)
        ↓
ONNX Conversion (Framework-agnostic)
        ↓
NCNN Framework (ARM NEON)
        ↓
YOLOv11n (Optimized Model)
        ↓
Camera Settings (640x480@15fps)
        ↓
Memory Management (Efficient Buffers)
```

#### 5.1.2 Software Optimization
- **Model**: Nano variant selection for speed
- **Conversion Pipeline**: PyTorch→ONNX→NCNN for optimal ARM performance
- **Runtime**: Lightweight ONNX/NCNN execution without PyTorch overhead
- **Backend**: NCNN for ARM acceleration
- **Resolution**: Balanced 640x480 resolution
- **Preprocessing**: Minimal image transformations
- **Post-processing**: Efficient bounding box operations

<div style="page-break-after: always;"></div>

### 5.2 Performance Monitoring

#### 5.2.1 Real-Time Metrics
- **FPS**: Frames processed per second
- **Detection Count**: People detected per frame
- **Processing Time**: Inference latency per frame
- **Memory Usage**: Runtime memory consumption

#### 5.2.2 Performance Targets
- **Target FPS**: 10-15 FPS on Raspberry Pi 4
- **Detection Latency**: <100ms per frame
- **Memory Usage**: <500MB total
- **Accuracy**: >85% person detection rate

---

<div style="page-break-after: always;"></div>

## 6. Security and Error Handling

### 6.1 Error Handling Strategy

#### 6.1.1 Graceful Degradation
```
Primary System Failure
        ↓
Automatic Fallback
        ↓
┌─────────────────────────┐
│ NCNN Fails              │
│ ↓                       │
│ ONNX Runtime Fallback   │
│ ↓                       │
│ PyTorch Backend         │
└─────────────────────────┘
┌─────────────────────────┐
│ USB Camera Fails        │
│ ↓                       │
│ Pi Camera Fallback      │
└─────────────────────────┘
┌─────────────────────────┐
│ GUI Fails               │
│ ↓                       │
│ Headless Mode           │
└─────────────────────────┘
```

#### 6.1.2 Resource Management
- Automatic camera resource cleanup
- Model memory deallocation
- OpenCV window management
- Exception handling with recovery

### 6.2 System Reliability

#### 6.2.1 Fault Tolerance
- Multiple model backend support
- Camera redundancy (USB/Pi camera)
- Display mode fallbacks
- Interrupt signal handling

---

<div style="page-break-after: always;"></div>

## 7. Configuration Management

### 7.1 Configuration Architecture

```
config.py (Central Configuration)
├── Camera Settings
│   ├── Resolution (640x480)
│   ├── FPS (15)
│   └── Buffer Size (1)
├── Detection Settings
│   ├── Model (yolo11n.pt)
│   ├── Confidence (0.5)
│   └── NCNN Enable (True)
├── Display Settings
│   ├── Window Name
│   ├── Font Properties
│   └── Color Schemes
└── File Settings
    ├── Naming Conventions
    └── Save Formats
```

### 7.2 Runtime Configuration
- Environment variable detection
- Automatic hardware adaptation
- Dynamic mode selection
- Performance parameter tuning

---

<div style="page-break-after: always;"></div>

## 8. Deployment Architecture

### 8.1 Installation Requirements

#### 8.1.1 Hardware Requirements
- **Minimum**: Raspberry Pi 3B+ (1GB RAM)
- **Recommended**: Raspberry Pi 4 (4GB+ RAM)
- **Camera**: USB webcam or Pi camera module
- **Storage**: 8GB+ microSD card

#### 8.1.2 Software Dependencies
```
Python 3.7+
├── ultralytics (YOLOv11)
├── opencv-python (Computer Vision)
├── numpy (Array Operations)
└── System Libraries
    ├── libgtk-3-dev (GUI Support)
    └── python3-tk (Display Interface)
```

### 8.2 Deployment Modes

#### 8.2.1 Development Deployment
- Local execution with GUI
- Real-time debugging
- Performance profiling
- Interactive testing

#### 8.2.2 Production Deployment
- Headless operation
- Automated startup
- Log file generation
- Remote monitoring

---

<div style="page-break-after: always;"></div>

## 9. Future Enhancements

### 9.1 Planned Architecture Extensions

#### 9.1.1 Network Integration
- Remote monitoring capabilities
- Cloud-based model updates
- Multi-device coordination
- Data synchronization

#### 9.1.2 Advanced Features
- Multi-object detection
- Person tracking and identification
- Behavior analysis
- Alert system integration

### 9.2 Scalability Considerations

#### 9.2.1 Horizontal Scaling
- Multi-camera support
- Distributed processing
- Load balancing
- Result aggregation

#### 9.2.2 Vertical Scaling
- Hardware acceleration (Coral TPU)
- Model optimization
- Algorithm improvements
- Performance tuning

---

<div style="page-break-after: always;"></div>

## 10. Conclusion

This architecture provides a robust, efficient, and scalable foundation for real-time people detection on Raspberry Pi hardware. The modular design ensures maintainability while the optimization strategies deliver the performance required for practical deployment scenarios.

The system's ability to automatically adapt to different environments and gracefully handle failures makes it suitable for both development and production use cases.

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