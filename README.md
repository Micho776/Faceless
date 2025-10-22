# FACELESS — Live Facial Recognition (Python)

This project demonstrates a live facial recognition system using Python, OpenCV, and `face_recognition` (dlib). It features a continuous learning system that automatically improves recognition accuracy over time by collecting high-confidence samples.

## Core Features

- **Live Recognition:** Recognizes known faces from a live webcam feed.
- `🎓` **Continuous Auto-Improvement:** The system gets smarter every time you use it. It automatically saves high-confidence (85%+) detections and merges them into the model.
- `🎯` **Face Tracking:** Reduces CPU usage by 50-70% by only re-detecting faces every N frames.
- `🚀` **GPU Acceleration (CUDA):** Automatically uses an NVIDIA GPU (if available) with the CNN model for higher accuracy detection.
- `👤` **Age & Gender Estimation:** Can optionally estimate age range and gender, displaying it alongside the person's name.
- `📸` **Multi-Camera Support:** Can list available cameras and be configured to use any connected camera.

## Project Structure

```bash
FACELESS/
├── known_faces/          # Add photos here
│   ├── Person1/
│   │   ├── photo1.jpg
│   │   └── photo2.jpg
│   └── Person2.jpg
├── data/
│   ├── encodings.pickle  # Main database
│   └── backups/          # Auto backups
├── learning_samples/     # Auto-collected samples
│   ├── Person1/
│   └── Person2/
├── logs/                 # Recognition logs
├── models/               # For age/gender models
├── encode_faces.py       # Script to encode known faces
├── recognize_live.py     # Main recognition script
├── recognize_live_enhanced.py # Enhanced script with Age/Gender
└── *.bat                 # Easy launchers
```

## Setup and Installation

### 1. Use the Conda Environment

This project uses the `facerec` Conda environment which has all dependencies installed (dlib, face_recognition, opencv-python, numpy, pillow).

You can run scripts without activating the environment (recommended):

```powershell
# Example:
C:\Users\user\miniconda3\condabin\conda.bat run -n facerec python encode_faces.py
```

Or activate the environment first:

```powershell
& C:\Users\user\miniconda3\shell\condabin\conda-hook.ps1
conda activate facerec
# Then run normally:
python encode_faces.py
```

### 2. Install Enhanced Dependencies (for Tracking)

The enhanced version needs `opencv-contrib-python` for tracking.

```powershell
C:\Users\user\miniconda3\condabin\conda.bat run -n facerec pip install opencv-contrib-python
```

### 3. Download Age/Gender Models (Optional)

If you want age/gender estimation, run the download script. This will download 4 pre-trained models (~200MB total) to the `models/` directory.

```powershell
C:\Users\user\miniconda3\condabin\conda.bat run -n facerec python download_models.py
```

## Quick Start Guide (Daily Use)

These launchers are the easiest way to use the system.

### 1. Add New People

1. Add 3-5 photos of a new person to a folder inside `known_faces/` (e.g., `known_faces/[person_name]/`).
2. Run `.\run_encoder.bat` to update the face database.
3. Done! The system will now recognize them.

### 2. Run Recognition

```bash
.\run_recognition.bat
```

- This starts the live webcam feed. Learning is enabled by default.
- Press **'q'** to quit.
- Press **'s'** to manually save the current frame for training.

### 3. Improve Model (After Sessions)

```bash
.\improve_model.bat
```

- Run this after every few sessions to merge all the new learning samples into the main database.

## How It Works: Auto-Improvement Workflow

The system gets smarter every time you run `run_recognition.bat`.

```bash
┌─────────────────────────────────────────────────────────────┐
│  1. START RECOGNITION                                       │
│     Double-click run_recognition.bat                        │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  2. LIVE RECOGNITION + LEARNING                             │
│     • Recognizes faces with confidence %                    │
│     • Auto-saves samples when confidence ≥ 85%              │
│     • Press 's' to manually save good frames                │
│     • Press 'q' when done                                   │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  3. AUTO-IMPROVEMENT (Happens Automatically!)               │
│     • Detects new learning samples                          │
│     • Creates backup of current model                       │
│     • Merges new samples into database                      │
│     • Clears learning samples                               │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│  4. NEXT SESSION IS BETTER!                                 │
│     • More encodings = higher accuracy                      │
│     • Better recognition from different angles              │
│     • Adapts to lighting changes, expressions               │
└─────────────────────────────────────────────────────────────┘
```

**Manual Control (Optional):** If you want to review samples before merging, edit `run_recognition.bat`, remove the `--learn` flag, and then manually run `python improve_model.py` when ready.

## Advanced Features Deep Dive

### 🚀 GPU Acceleration (CUDA)

- Uses the CNN face detection model for higher accuracy.
- Automatically uses an NVIDIA GPU if CUDA is available, falling back to CPU if not.
- To use, run with the `--model cnn` flag (see `recognize_live_enhanced.py`).

### 🎯 Face Tracking

- Tracks detected faces between frames, reducing CPU usage by 50-70%.
- Only re-detects faces every N frames (configurable with `--detect-interval`).
- Enable with the `--track` flag.

### 👤 Age & Gender Estimation

- Estimates age range (8 categories) and gender (Male/Female).
- Displays this info alongside the person's name.
- Requires downloading the models (see setup).
- Can be disabled with `--no-age-gender` for better performance.
- **Age Ranges:** (0-2), (4-6), (8-12), (15-20), (25-32), (38-43), (48-53), (60-100).

## Camera Selection

### 1. List Available Cameras

Run this script to see all connected cameras and their IDs.

```bash
.\list_cameras.bat
```

Output example:

```text
Camera 0: 1280x720 @ 30fps  (Built-in webcam)
Camera 1: 1920x1080 @ 60fps (External USB camera)
```

### 2. Use Different Camera

Edit `run_recognition.bat` and change the `--camera` flag:

```batch
--camera 0  →  --camera 1
```

Or run the script manually with the flag:

```bash
python recognize_live.py --camera 1
```

## Command-Line Reference

The project appears to have two main recognition scripts with different options.

### `recognize_live.py` (Main Script)

These options are available for the main script.

```bash
python recognize_live.py [OPTIONS]

--camera 0           Camera index (default: 0)
--model hog          Detection model: hog/cnn (default: hog)
--track              Enable face tracking (recommended)
--fps                Show FPS counter
--scale 0.4          Processing scale (lower=faster)
--confidence 0.6     Recognition threshold (0.0-1.0)
--log                Enable logging
--learn              Enable continuous learning
--list-cameras       List available cameras
```

### `recognize_live_enhanced.py` (Enhanced Script)

These options are available for the enhanced script.

```text
--model {hog,cnn}         Face detection model (default: hog)
                          hog = CPU-optimized, faster
                          cnn = GPU-accelerated, more accurate

--track                   Enable face tracking between frames
                          Reduces CPU usage significantly

--detect-interval N       Frames between full detection when tracking
                          Default: 30 (higher = less CPU, may lose faces)

--no-age-gender           Disable age/gender estimation

--fps                     Show FPS counter on screen

--camera N                Camera device index (default: 0)
```

## Performance

### Performance Tips

- Use `--track` for significantly better performance (lower CPU).
- Use `--scale 0.25` for the fastest speed (lower quality).
- Use `--scale 0.5` for a balanced speed/quality.
- The HOG model is faster (CPU), while the CNN model is more accurate (GPU).
- Add more photos per person (different angles, lighting) for better accuracy.

### Performance Comparison

**With/Without Tracking**:

| Mode                      | CPU Usage | FPS   | Accuracy                      |
| ------------------------- | --------- | ----- | ----------------------------- |
| Without Tracking          | \~40-60%  | 15-25 | Highest (detects every frame) |
| With Tracking (`--track`) | \~15-25%  | 40-60 | Very Good (occasional misses) |

**HOG (CPU) vs. CNN (GPU)**:

| Model | Speed                  | Accuracy  | GPU          | Best For                  |
| ----- | ---------------------- | --------- | ------------ | ------------------------- |
| HOG   | Fast (\~30 FPS on CPU) | Good      | Not required | Real-time on any machine  |
| CNN   | Very Fast with CUDA    | Excellent | Recommended  | Highest accuracy with GPU |

## Backups and Restore

Every time the model is improved (either automatically or via `improve_model.bat`), a timestamped backup of the _previous_ model is created in `data/backups/`.

```text
data/backups/
  ├── encodings_backup_20251023_010000.pickle
  └── encodings_backup_20251023_020000.pickle
```

To restore a backup:

```bash
copy data\backups\encodings_backup_YYYYMMDD_HHMMSS.pickle data\encodings.pickle
```

## GPU/CUDA Setup (Windows)

For using the `cnn` model with GPU acceleration.

### 1. Check for NVIDIA GPU

```powershell
nvidia-smi
```

### 2. Install CUDA Toolkit

1. Download from: [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
2. Install CUDA 11.8 or 12.1 (recommended)
3. Restart your computer

### 3. Verify CUDA in Python

```powershell
C:\Users\miche\miniconda3\condabin\conda.bat run -n facerec python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}')"
```

_(Note: This requires `torch` to be installed in the `facerec` environment as shown in FEATURES.md)_

## Troubleshooting

- **No images found:** Add images to `known_faces/` and rerun `run_encoder.bat` (or `python encode_faces.py`).
- **Webcam doesn't open:** Ensure no other app is using the camera and check Windows privacy settings.
- **"Tracker not found" error:** Install `opencv-contrib-python` (see setup steps).
- **Age/gender models not found:** Run `python download_models.py` to download the models.
- **CNN model is slow:** Make sure you have an NVIDIA GPU with CUDA installed. If not, use `--model hog` for CPU-optimized detection.
- **Tracking loses faces:** Decrease `--detect-interval` (e.g., `--detect-interval 15`). For maximum accuracy, disable tracking.
- **High CPU usage with tracking:** Increase `--detect-interval` (e.g., `--detect-interval 60`).

---

## Author

**Made by Micho** 🚀
