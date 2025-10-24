<p align="center">
<strong>=================================================================</strong><br>
<strong>README</strong><br><br>
<strong>Date:</strong> 10/2025<br><br>
<strong>Moderator:</strong> Daniel.J.Q.Goh<br>
<strong>=================================================================</strong>
</p>

<br><br><br><br>

!!! / This section contains the main objectives to complete the entire project

# Real-Time People Detection System for Raspberry Pi

A professional, clean implementation of real-time people detection using YOLOv11 with NCNN optimization for Raspberry Pi.

## Features

**Smart Environment Detection** - Automatically detects if GUI or headless mode should be used  
**YOLOv11 Nano Model** - Latest YOLO version optimized for Raspberry Pi performance  
**NCNN Backend** - Hardware acceleration for better inference speed  
**Dual Camera Support** - Works with USB webcams and Pi camera  
**GUI & Headless Modes** - Live display or image saving  
**Professional Code Structure** - Clean, documented, and maintainable  

## Requirements

```bash
pip install ultralytics opencv-python numpy
```

## Quick Start

### 1. **Simple Run**
Ultralytics makes it easy to download and use an off-the-shelf YOLO model. These models are trained on the COCO dataset and can detect 80 common objects, such as “person”, “car”, “chair”, and so on. To download a __YOLO11n__ detection model, issue:
```bash
yolo detect predict model=yolo11n.pt
```
To run the __object detection module__, run the following command to execute the program:
Before running the __object detection module__, create and activate a virtual environment:
```bash
python3 -m venv raspi
source raspi/bin/activate
```
Then run the following command to execute the program:
```bash
python3 obj_detection.py
```
To exit the virtual environment when you are done, run:
```bash
deactivate
```
The system will automatically:
- Detect your environment (GUI/headless)
- Load and optimize the YOLO model
- Test camera connection
- Start detection in the best mode

### 2. **Manual Mode Selection**

**GUI Mode (with window display):**
```python
detector = PeopleDetectionSystem()
detector.run_gui_detection(duration_seconds=0, camera_index=0)
```

**Headless Mode (saves images):**
```python
detector = PeopleDetectionSystem()
detector.run_headless_detection(duration_seconds=0, camera_index=0, save_interval=5)
```

**Smart Auto-Detection:**
```python
detector = PeopleDetectionSystem()
detector.run_smart_detection(duration_seconds=0, camera_index=0)
```

## Configuration

Edit `config.py` to customize:
- Camera settings (resolution, FPS)
- Detection parameters (confidence threshold)
- Display options (colors, fonts)
- File naming conventions

## Controls

### GUI Mode:
- **'q'** - Quit detection
- **'s'** - Save current frame
- **Window close** - Stop detection

### Headless Mode:
- **Ctrl+C** - Stop detection
- Images auto-saved every 5 seconds
- Final frame saved on exit

## Output Files

**Headless Mode generates:**
- `detection_[timestamp]_people_[count].jpg` - Periodic saves
- `detection_final_[timestamp].jpg` - Final frame

**GUI Mode generates (when 's' pressed):**
- `detection_frame_[timestamp].jpg` - Manual saves

## Troubleshooting

### No Camera Window Opens:
- **SSH Users**: Connect with `ssh -X pi@your_pi_ip`
- **Headless Pi**: System auto-switches to headless mode
- **Missing Display**: Install `sudo apt install python3-opencv libgtk-3-dev`

### Performance Issues:
- Lower resolution in `config.py`
- Reduce FPS settings
- Ensure NCNN export succeeded

### Camera Not Found:
- Check USB connection
- Try different camera index (0 instead of 1)
- Verify camera permissions

## System Architecture

```
PeopleDetectionSystem (Main Class)
├── print_system_info()          # System information
├── detect_raspberry_pi()        # Hardware detection
├── load_model()                 # YOLOv11 model loading & NCNN export
├── setup_camera()               # Camera initialization
├── detect_people_in_frame()     # Core detection logic
├── check_display_environment()  # Display capability check
├── run_gui_detection()          # GUI mode with live display
├── run_headless_detection()     # Headless mode with image saving
├── run_smart_detection()        # Auto-mode selection
└── run()                        # Main execution flow
```

## Performance Stats

**Typical Performance on Raspberry Pi 3B V1.2:**
- **Resolution**: 640x480
- **FPS**: 10-15 (depending on detections)
- **Model**: YOLOv11n with NCNN
- **Memory**: ~750MB(0.75GB) RAM usage

## License

This code is provided as-is for educational and development purposes.

## Support

For issues or improvements, check:
1. Camera connection and permissions
2. Display environment setup
3. Required package installation
4. Model download completion
5. Raspberry Pi 3B V1.2 CPU and RAM bottleneck during model training
---

<p align="center">
<em>End of Document</em>
</p>