# 🛠️ Installation & Running

Both scripts are independent. Follow these steps to install dependencies and run each script:

## 1. Install Dependencies

```bash
pip install opencv-python numpy ultralytics psutil
```

Download the YOLOv11 Nano model file (`yolo11n.pt`) and place it in the same directory as the script or specify the correct path in your code.

Connect your camera (USB webcam or Pi camera) if you want to use detection features.

---

## 2. Run Benchmark Test Suite

To run the Raspberry Pi benchmark and validation tests:

```bash
python3 tests/test_raspi_detection.py
```

To auto-install missing dependencies (recommended for first run):

```bash
python3 tests/test_raspi_detection.py --install-deps
```

---

## 3. Run Real-Time People Detection

To run the real-time people detection program:

```bash
python3 scripts/obj_detection.py
```

---
# <p align="center">
   <img src="https://img.icons8.com/color/96/000000/raspberry-pi.png" alt="Raspberry Pi" width="60"/>
   <img src="https://img.icons8.com/color/96/000000/camera.png" alt="Camera" width="60"/>
   <img src="https://img.icons8.com/color/96/000000/artificial-intelligence.png" alt="AI" width="60"/>
</p>

# Real-Time People Detection System for Raspberry Pi

## 📋 Overview
This project provides a complete solution for real-time people detection on Raspberry Pi 3B v1.2 using a camera and deep learning models. It includes:
- `obj_detection.py`: The main detection program for live people detection using a camera.
- `test_raspi_detection.py`: A comprehensive benchmark and validation test suite for Raspberry Pi hardware and detection pipeline.

## ✨ Features
- Real-time detection of people using YOLOv11 Nano model
- Optimized for Raspberry Pi ARM64 architecture and low memory
- Supports both GUI (window display) and headless (image saving) modes
- Automatic environment detection for best performance
- Benchmark tests for speed, memory, and integration

---

## 🚀 Usage

### 🧪 1. Benchmark & Validation: `test_raspi_detection.py`
This script runs a full suite of tests to validate your Raspberry Pi setup and the detection pipeline.

**Run all tests:**
```bash
python3 raspi/tests/test_raspi_detection.py
```

**Auto-install dependencies (recommended for first run):**
```bash
python3 raspi/tests/test_raspi_detection.py --install-deps
```

**What it does:**
- Checks Pi hardware and dependencies
- Validates model loading and memory usage
- Tests camera integration
- Measures inference speed and performance
- Verifies integration with `obj_detection.py`
- Prints a summary of results and readiness for deployment

---

### 🎥 2. Real-Time Detection: `obj_detection.py`
This is the main program for live people detection using a camera on Raspberry Pi.

**Run the detection program:**
```bash
python3 raspi/scripts/obj_detection.py
```

**Modes:**
- **Smart Detection (default):** Automatically chooses GUI or headless mode based on environment
- **GUI Mode:** Displays live video with detection overlays
- **Headless Mode:** Saves annotated images periodically (for SSH or no display)

**How it works:**
- Loads YOLOv11 Nano model (optimized for Pi)
- Sets up camera (USB or Pi camera)
- Detects people in each frame, draws bounding boxes and labels
- Displays or saves annotated frames
- Prints detection statistics (FPS, people count, runtime)

**Custom options:**
- Change detection duration or camera index by editing the script or passing arguments

---

## 🛠 Requirements
- Raspberry Pi 3B v1.2 (or compatible ARM64 device)
- Python 3.8+
- Camera (USB webcam or Pi camera)
- Required Python packages: `opencv-python`, `numpy`, `ultralytics`, `psutil`
- YOLOv11 Nano model file (`yolo11n.pt`)

---

## ⚡ Quick Start
1. Clone the repository and navigate to the project folder.
2. Ensure your camera is connected.
3. (Optional) Build and run the Docker container for a reproducible environment.
4. Run the benchmark test suite:
   ```bash
   python3 raspi/tests/test_raspi_detection.py --install-deps
   ```
5. Run the real-time detection program:
   ```bash
   python3 raspi/scripts/obj_detection.py
   ```

---

## 🧩 Troubleshooting
- If you see errors about missing packages, use the `--install-deps` option.
- For camera issues, check your connection and try both camera index 0 and 1.
- For Docker usage, see the documentation and ensure permissions for `/dev/video0`.

---

## 📚 Documentation
- See `raspi/tests/TEST_DOCUMENTATION.md` for detailed test case descriptions and results.
- See `raspi/scripts/obj_detection.py` for code comments and usage details.

---

## 📝 License
MIT License

---

## 👤 Author
Daniel J.Q. Goh

---

<p align="center">
   <img src="https://img.icons8.com/color/48/000000/github.png" alt="GitHub" width="32"/>
</p>

For questions or bug reports, please open an issue on GitHub.