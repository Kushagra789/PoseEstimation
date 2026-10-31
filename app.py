"""Streamlit app that turns any video into a white-on-black human pose skeleton video."""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

try:
    import imageio_ffmpeg
except ImportError:  # fall back to OpenCV's MP4 writer
    imageio_ffmpeg = None

PRIMARY_MODEL = "yolo26n-pose.pt"   # YOLO26 nano pose (needs ultralytics >= 8.4.0)
FALLBACK_MODEL = "yolo11n-pose.pt"  # previous-generation nano pose model
DETECTION_CONF = 0.25               # minimum confidence for a person detection
KEYPOINT_CONF = 0.5                 # minimum confidence for a joint to be drawn
OUTPUT_PATH = Path(__file__).parent / "pose_result.mp4"
VIDEO_TYPES = ["mp4", "mov", "avi", "mkv", "webm", "m4v", "mpg", "mpeg", "wmv"]
WHITE = (255, 255, 255)

# COCO 17-keypoint skeleton: pairs of joint indices joined by a line.
# 0 nose, 1-2 eyes, 3-4 ears, 5-6 shoulders, 7-8 elbows, 9-10 wrists,
# 11-12 hips, 13-14 knees, 15-16 ankles (left, right).
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),
    (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]


@st.cache_resource(show_spinner="Loading YOLO pose model (downloaded once on first run)...")
def load_model():
    """Load YOLO26 nano pose; fall back to YOLO11 nano pose if it cannot be loaded."""
    try:
        return YOLO(PRIMARY_MODEL), PRIMARY_MODEL, None
    except Exception as exc:  # e.g. outdated ultralytics package or failed download
        note = (
            f"Could not load {PRIMARY_MODEL} ({exc}). Using {FALLBACK_MODEL} instead. "
            "Run 'pip install -U ultralytics' to enable YOLO26."
        )
        return YOLO(FALLBACK_MODEL), FALLBACK_MODEL, note


def draw_skeletons(result, width, height, line_thickness, joint_radius):
    """Return a black frame with every detected person's skeleton drawn in white."""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    keypoints = result.keypoints
    if keypoints is None or len(keypoints) == 0:
        return canvas

    xy = keypoints.xy.cpu().numpy()  # (people, 17, 2) pixel coordinates
    conf = keypoints.conf.cpu().numpy() if keypoints.conf is not None else np.ones(xy.shape[:2])

    for person_xy, person_conf in zip(xy, conf):
        visible = (person_conf >= KEYPOINT_CONF) & (person_xy[:, 0] > 0) & (person_xy[:, 1] > 0)
        points = [(int(round(x)), int(round(y))) for x, y in person_xy]
        for a, b in SKELETON:
            if visible[a] and visible[b]:
                cv2.line(canvas, points[a], points[b], WHITE, line_thickness, cv2.LINE_AA)
        for i in np.flatnonzero(visible):
            cv2.circle(canvas, points[i], joint_radius, WHITE, -1, cv2.LINE_AA)
    return canvas


class Mp4Writer:
    """Writes BGR frames to MP4 at the exact input size.

    Uses H.264 via imageio-ffmpeg (plays in browsers); falls back to OpenCV's
    mp4v codec (plays in desktop players, usually not in browsers).
    """

    def __init__(self, path, width, height, fps):
        if imageio_ffmpeg is not None:
            try:
                # yuv420p is what browsers play but needs even dimensions; odd sizes use yuv444p.
                even = width % 2 == 0 and height % 2 == 0
                self._ffmpeg = imageio_ffmpeg.write_frames(
                    str(path),
                    (width, height),
                    pix_fmt_in="bgr24",
                    pix_fmt_out="yuv420p" if even else "yuv444p",
                    fps=fps,
                    codec="libx264",
                    quality=7,
                    macro_block_size=1,  # never resize the frames
                    output_params=["-movflags", "+faststart"],
                )
                self._ffmpeg.send(None)  # start the ffmpeg process
                self.backend = "imageio-ffmpeg (H.264)"
                self.browser_friendly = even
                return
            except Exception:
                pass  # e.g. bundled ffmpeg binary missing; use OpenCV below

        self._cv = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self._cv.isOpened():
            raise RuntimeError("Could not create the output video with imageio-ffmpeg or OpenCV.")
        self._ffmpeg = None
        self.backend = "OpenCV (mp4v)"
        self.browser_friendly = False

    def write(self, frame):
        if self._ffmpeg is not None:
            self._ffmpeg.send(frame.tobytes())
        else:
            self._cv.write(frame)

    def close(self):
        if self._ffmpeg is not None:
            self._ffmpeg.close()
        else:
            self._cv.release()


def process_video(model, input_path, output_path, on_progress):
    """Run pose estimation frame by frame and write the skeleton-only video."""
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise ValueError("OpenCV could not open this video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 1000:  # missing or broken metadata
        fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    writer = None
    frames_done = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if writer is None:
                # Size comes from the decoded frame, so rotated phone videos are handled.
                height, width = frame.shape[:2]
                writer = Mp4Writer(output_path, width, height, fps)
                line_thickness = max(2, round(min(width, height) / 240))
                joint_radius = line_thickness + 2

            result = model(frame, conf=DETECTION_CONF, verbose=False)[0]
            writer.write(draw_skeletons(result, width, height, line_thickness, joint_radius))

            frames_done += 1
            on_progress(frames_done, total_frames)
    finally:
        cap.release()
        if writer is not None:
            writer.close()

    if frames_done == 0:
        raise ValueError("No frames could be read from this video.")

    return {
        "width": width,
        "height": height,
        "fps": fps,
        "frames": frames_done,
        "encoder": writer.backend,
        "browser_friendly": writer.browser_friendly,
    }


def main():
    st.set_page_config(page_title="Pose Skeleton Video", page_icon="🦴", layout="centered")
    st.title("Human Pose Skeleton Video Generator")
    st.write(
        "Upload a video to generate a new video that shows only the human pose skeletons "
        "(white joints and limbs on a black background) for every person detected."
    )

    uploaded = st.file_uploader("Upload a video", type=VIDEO_TYPES)

    # Forget the previous result when a different file is uploaded.
    if uploaded is not None and st.session_state.get("source_id") != uploaded.file_id:
        st.session_state.source_id = uploaded.file_id
        st.session_state.pop("result", None)

    if st.button("Generate Pose Video", type="primary", disabled=uploaded is None):
        model, model_name, warning = load_model()
        if warning:
            st.warning(warning)

        progress = st.progress(0.0, text="Starting...")

        def on_progress(done, total):
            if done % 5 and done != total:  # update the UI every 5 frames
                return
            if total > 0:
                progress.progress(min(done / total, 1.0), text=f"Processing frame {done} of {total}")
            else:
                progress.progress(0.0, text=f"Processing frame {done}")

        suffix = Path(uploaded.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            input_path = Path(tmp.name)

        try:
            info = process_video(model, input_path, OUTPUT_PATH, on_progress)
        except Exception as exc:
            progress.empty()
            st.error(f"Processing failed: {exc}")
        else:
            progress.progress(1.0, text="Done")
            st.session_state.result = {**info, "model": model_name}
        finally:
            input_path.unlink(missing_ok=True)

    result = st.session_state.get("result")
    if result and OUTPUT_PATH.exists():
        st.success(f"Pose video saved as {OUTPUT_PATH.name}")
        st.caption(
            f"{result['width']}x{result['height']} · {result['fps']:.2f} fps · "
            f"{result['frames']} frames · {result['frames'] / result['fps']:.1f} s · "
            f"model: {result['model']} · encoder: {result['encoder']}"
        )
        if not result["browser_friendly"]:
            st.info("The browser preview may not play this file (odd pixel dimensions or "
                    "OpenCV fallback encoder). The downloaded file still plays in VLC "
                    "and most desktop players.")

        video_bytes = OUTPUT_PATH.read_bytes()
        st.video(video_bytes, format="video/mp4")
        st.download_button(
            "Download pose_result.mp4",
            data=video_bytes,
            file_name="pose_result.mp4",
            mime="video/mp4",
        )


main()
