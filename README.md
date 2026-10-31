# Human Pose Skeleton Video Generator

A local Streamlit app that takes any video and produces `pose_result.mp4`. The new video
shows only human pose skeletons: white joints and white limbs on a pure black background.
It works with multiple people per frame.

Built with Streamlit, OpenCV, NumPy, Ultralytics YOLO26 (`yolo26n-pose.pt`) and
imageio-ffmpeg. Runs fully offline after the first model download. No API key or paid service.

## How it works

1. The uploaded video is read frame by frame with OpenCV.
2. Each frame goes through the YOLO26 nano pose model, which predicts 17 COCO keypoints
   (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) for every person.
3. A new black frame of the same size is created. Keypoints with confidence >= 0.5 are drawn
   as white dots and connected with white lines using the COCO skeleton.
4. Frames are encoded as MP4 at the original resolution and frame rate, so the output has
   the same number of frames and the same duration as the input.
   - Primary encoder: imageio-ffmpeg (H.264), which plays in the browser preview.
   - Fallback encoder: OpenCV `mp4v`, used automatically if imageio-ffmpeg is unavailable.
     Its files play in VLC and most desktop players but usually not in the browser preview.

## Requirements

- Windows 10/11, Python 3.14 (Python 3.12 as fallback)
- Internet connection on the first run only (to install packages and download the ~7 MB
  `yolo26n-pose.pt` weights, which Ultralytics fetches automatically)

## Setup (VS Code terminal, PowerShell)

Open the `PoseEstimation` folder in VS Code and open a terminal with `` Ctrl+` ``.

### Option A: Python 3.14 (try first)

```powershell
python -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`--system-site-packages` lets the environment reuse packages already installed globally
(for example `torch` and `numpy`), so the environment stays small. This matters because
the project folder is synced by OneDrive.

### Option B: Python 3.12 (only if a package fails to install in Option A)

```powershell
deactivate
Remove-Item -Recurse -Force .venv
py -3.12 -m venv .venv --system-site-packages
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### If activation is blocked

If you see "running scripts is disabled on this system", run this in the same terminal and
activate again (it only affects the current terminal session):

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

### Select the interpreter in VS Code

Press `Ctrl+Shift+P` -> "Python: Select Interpreter" -> choose `.\.venv\Scripts\python.exe`.

### Verify the installation

```powershell
python -c "import ultralytics, streamlit, cv2, imageio_ffmpeg; print(ultralytics.__version__, streamlit.__version__, cv2.__version__)"
```

The ultralytics version must be 8.4.0 or newer for YOLO26.

## Run

With the virtual environment active (the prompt starts with `(.venv)`):

```powershell
streamlit run app.py
```

The browser opens at http://localhost:8501. Upload a video, click **Generate Pose Video**,
wait for the progress bar, then preview and download `pose_result.mp4`. The file is also
saved in the project folder and is overwritten on each run. Stop the app with `Ctrl+C`.

For videos larger than 200 MB:

```powershell
streamlit run app.py --server.maxUploadSize 1024
```

## Settings (top of `app.py`)

| Setting          | Default            | Effect                                               |
|------------------|--------------------|------------------------------------------------------|
| `PRIMARY_MODEL`  | `yolo26n-pose.pt`  | Use `yolo26s-pose.pt` etc. for accuracy over speed   |
| `DETECTION_CONF` | `0.25`             | Lower = more people found, more false positives      |
| `KEYPOINT_CONF`  | `0.5`              | Lower = more joints drawn, more jittery/wrong joints |

If `yolo26n-pose.pt` cannot be loaded (for example, an outdated ultralytics), the app shows a
warning and uses `yolo11n-pose.pt` instead.

## Limitations

- The output has no audio (it only contains skeletons).
- Variable-frame-rate videos (common on phones) are written at their average frame rate,
  so the timing can drift slightly.
- Videos with odd pixel dimensions keep their exact size but use a pixel format that the
  browser preview may not play; the downloaded file is still valid.
- Processing runs on the CPU unless a CUDA GPU and the CUDA build of PyTorch are installed.
  Expect a few to a few dozen frames per second with the nano model.
- Pose estimation is weaker on small or distant people, heavy occlusion, crowded scenes,
  motion blur and unusual poses. Test with such clips and compare thresholds or model
  sizes to evaluate this.

## Project structure

```
PoseEstimation/
├── app.py            # Streamlit UI + video processing
├── requirements.txt  # Python dependencies
├── README.md
└── pose_result.mp4   # created after the first run
```
