# Raspberry Pi 3B v1.2 Object Detection Setup 🍓

This folder contains Docker and testing setup specifically optimized for **Raspberry Pi 3B v1.2** with **1GB RAM** and **ARM64 architecture**.

## 🎯 Pi-Specific Optimizations

### Hardware Constraints Addressed:
- **1GB RAM** - Memory-optimized Docker container (750MB limit)
- **ARM Cortex-A53** - ARM64 base image with piwheels packages
- **Limited CPU** - Reduced inference resolution and frame skipping
- **No Intel optimizations** - Removed Intel-specific libraries

### Performance Expectations on Pi 3B:
- **Inference time**: 500-2000ms (vs 64ms on Intel)
- **FPS**: 1-3 fps (vs 15+ fps on Intel)
- **Memory usage**: ~600MB total
- **Model load time**: 10-30 seconds

## 🚀 Quick Start

### 1. Setup Script (Recommended)
```bash
# Navigate to the raspi folder
cd raspi/

# Make setup script executable
chmod +x setup_raspi.sh

# Run complete Pi setup and testing
./setup_raspi.sh
```

### 2. Manual Setup
```bash
# Navigate to the raspi folder
cd raspi/

# Build Pi-optimized image
docker-compose build

# Start Pi container
docker-compose up -d

# Run Pi test suite
docker-compose exec raspi-detection python3 /workspace/tests/test_raspi_detection.py

# Test your obj_detection.py (copy it to raspi/scripts/ first)
docker-compose exec raspi-detection python3 /workspace/scripts/obj_detection.py
```

## 📁 Pi-Specific Files

```
raspi/
├── Dockerfile                    # ARM64 optimized Dockerfile
├── docker-compose.yml           # Pi container configuration
├── requirements.raspi.txt       # Minimal Pi dependencies
├── setup_raspi.sh              # Automated Pi setup script
├── README.md                   # This file
├── scripts/                    # Copy your obj_detection.py here
└── tests/
    └── test_raspi_detection.py # Pi-specific test suite
```

## 🧪 Test Suite for Pi

The Pi test suite (`test_raspi_detection.py`) includes 8 specialized tests:

1. **Pi Environment Check** - Verify ARM architecture and memory
2. **Ultralytics Import** - Memory-optimized import testing
3. **Model Loading** - Pi memory constraint validation
4. **Camera Detection** - Pi camera interface testing
5. **Inference Speed** - Pi-realistic performance testing
6. **Memory Stress** - Pi memory limit validation
7. **Integration Test** - obj_detection.py compatibility
8. **Performance Benchmark** - Overall Pi performance metrics

### Expected Test Results:
- **Inference time**: < 2000ms ✅
- **Memory increase**: < 400MB ✅
- **Model load time**: < 30s ✅
- **FPS**: > 0.5 ✅

## 🔧 Pi Configuration Details

### Docker Container Specs:
```yaml
# Resource limits for Pi 3B
mem_limit: 750m          # 75% of 1GB RAM
memswap_limit: 750m      # No swap
cpus: "4.0"              # All 4 ARM cores
```

### Python Dependencies:
```
ultralytics>=8.0.0      # Core YOLO
opencv-python>=4.5.0    # Computer vision
numpy>=1.21.0           # Numerical computing
psutil>=5.8.0           # System monitoring
```

### YOLO Model Optimizations:
- **Model**: YOLO11n (smallest/fastest)
- **Input size**: 320x240 (vs 640x640)
- **Confidence**: 0.6 (vs 0.4)
- **Device**: CPU only
- **Frame skip**: Every 3rd frame

## 📊 Performance Comparison

| Metric | Intel x86_64 | Pi 3B ARM64 | Optimization |
|--------|-------------|-------------|--------------|
| Inference | 64ms | 500-2000ms | Frame skipping |
| Memory | 531MB | 600-750MB | Minimal packages |
| FPS | 15+ | 1-3 | Lower resolution |
| Load time | 0.1s | 10-30s | Model caching |

## 🛠️ Troubleshooting

### Common Pi Issues:

1. **"Out of memory" errors**:
   ```bash
   # Check memory usage
   docker-compose exec raspi-detection free -h
   
   # Reduce model size or image resolution
   ```

2. **Slow inference**:
   ```bash
   # This is normal for Pi 3B - try:
   # - Lower confidence threshold
   # - Smaller input images
   # - Frame skipping
   ```

3. **Docker build fails**:
   ```bash
   # Pi 3B builds are slow (15-30 minutes)
   # Ensure stable internet connection
   # Use piwheels for faster package installation
   ```

4. **Camera not detected**:
   ```bash
   # Enable camera in raspi-config
   sudo raspi-config
   
   # Check camera module
   vcgencmd get_camera
   ```

### Debug Mode:
```bash
# Enter Pi container for debugging
docker-compose exec raspi-detection bash

# Check system resources
htop
free -h
cat /proc/cpuinfo
```

## 🔄 Setting Up Your obj_detection.py

To use your main obj_detection.py with this Pi setup:

1. **Copy your script to the Pi folder**:
   ```bash
   # From the main project directory
   cp scripts/obj_detection.py raspi/scripts/
   ```

2. **Build and run Pi container**:
   ```bash
   cd raspi/
   docker-compose build
   docker-compose up -d
   ```

3. **Test your script on Pi**:
   ```bash
   docker-compose exec raspi-detection python3 /workspace/scripts/obj_detection.py
   ```

## 📈 Optimization Tips for Pi

1. **Reduce input resolution**:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
   ```

2. **Increase confidence threshold**:
   ```python
   results = model(image, conf=0.6)  # Higher than 0.4
   ```

3. **Skip frames**:
   ```python
   if frame_count % 3 == 0:  # Process every 3rd frame
       run_detection()
   ```

4. **Use headless mode**:
   ```python
   # Save images instead of displaying
   cv2.imwrite(f'output_{timestamp}.jpg', annotated_frame)
   ```

## 🎉 Success Indicators

Your Pi setup is working correctly when:
- ✅ All 8 tests pass in test suite
- ✅ obj_detection.py runs without errors
- ✅ Inference time < 2 seconds
- ✅ Memory usage < 750MB
- ✅ Camera detection works (if connected)

## 🔗 Related Files

- `../scripts/obj_detection.py` - Your main detection script (copy to raspi/scripts/)
- `../Dockerfile` - Intel-optimized version
- `../docker-compose.yml` - Intel container config
- `../tests/complete_test_suite.py` - Intel test suite

---

**Note**: This Pi setup is specifically optimized for Raspberry Pi 3B v1.2. For Pi 4 with more RAM, you can increase memory limits and potentially use ONNX optimization.