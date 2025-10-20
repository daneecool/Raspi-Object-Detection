## Docker usage for Model_Training

This document explains how to build and run the project's Docker image and common troubleshooting steps. It assumes you are working from the project root (`/mnt/c/Users/daniel-goh/Desktop/Model_Training` in WSL).

### Prerequisites
- Docker installed and running on your machine
- (Optional) `docker-compose` if you prefer compose-based workflows
- For GPU acceleration: NVIDIA drivers + Docker NVIDIA support (see notes below)

### Files of interest
- `Dockerfile` — builds an image that installs Python, packages from `requirements.txt`, and NCNN
- `.dockerignore` — excludes local venvs, data, and models from build context

### Build the image
From the project root run:

```bash
cd /mnt/c/Users/daniel-goh/Desktop/Model_Training
docker build -t model_training:latest .
```

Notes:
- We added a fallback `pip3 install --no-cache-dir ultralytics opencv-python numpy --upgrade` in the `Dockerfile` so those packages are present even if `requirements.txt` changes.
- Use `.dockerignore` to keep build context small — make sure large folders like local virtualenvs and `data/` are ignored.

### Run the container and execute the training script
To run the training script inside the container and mount the current project into `/workspace`:

```bash
cd /mnt/c/Users/daniel-goh/Desktop/Model_Training
docker run --rm -it \
  -v "$(pwd)":/workspace \
  -w /workspace \
  model_training:latest \
  /bin/bash -lc "python yolo11n_training.py"
```

To start an interactive shell inside the container (useful for debugging):

```bash
docker run --rm -it -v "$(pwd)":/workspace -w /workspace model_training:latest /bin/bash
```

Inside the container you can run `python yolo11n_training.py` or inspect `/workspace`.

### Mounting data and models
- The recommended pattern is to keep large datasets and model weights outside the image and mount them at runtime. Example:

```bash
docker run --rm -it \
  -v "$(pwd)":/workspace \
  -v "/absolute/path/to/local/data":/workspace/data \
  -v "/absolute/path/to/local/models":/workspace/models \
  -w /workspace model_training:latest /bin/bash -lc "python yolo11n_training.py"
```

### Ports
- The `Dockerfile` exposes ports `8888` and `5000` if you run services inside the container. Map them with `-p` if needed:

```bash
docker run -p 8888:8888 -p 5000:5000 ...
```

### GPU support
- The default `Dockerfile` uses `ubuntu:22.04`. For NVIDIA GPU support you can switch to an NVIDIA CUDA base image and enable GPU at runtime. Example runtime flag (modern Docker):

```bash
docker run --gpus all ...
```

On older setups use the NVIDIA Container Toolkit and `--runtime=nvidia`.

### Rebuild when dependencies change
If you change `requirements.txt` or the `Dockerfile`, rebuild:

```bash
docker build -t model_training:latest --no-cache .
```

### Troubleshooting
- If import errors happen inside the container, start an interactive shell (`/bin/bash`) and run `python -c "import ultralytics, cv2, numpy; print(ultralytics.__version__)"` to verify installations.
- If builds are slow, ensure `.dockerignore` excludes large files and local venv directories (e.g. `.venv`, `.wsl_venv`).
- If you see permission errors when mounting Windows paths in WSL, run Docker Desktop and ensure file sharing/mounting is enabled.

### Example Docker Compose snippet
If you prefer compose, add a service to your `docker-compose.yml` like:

```yaml
services:
  model_training:
    image: model_training:latest
    build: .
    volumes:
      - ./:/workspace
      - ./data:/workspace/data
      - ./models:/workspace/models
    working_dir: /workspace
    command: /bin/bash -lc "python yolo11n_training.py"
```

### Security note
- Do not include credentials or tokens inside the image or repository. Use build-time secrets or mount credentials at runtime if necessary.

---

If you want, I can also add a small `docker-compose.service` file or extend the README with examples for running experiments and saving outputs to a host folder. Tell me which example you prefer and I will add it.
