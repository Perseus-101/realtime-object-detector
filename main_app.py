import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import os
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import threading
import time
import numpy as np
import av

# --- Model Loading ---
@st.cache_resource
def load_yolo_model(model_name):
    """Loads the YOLOv8 model from cache."""
    return YOLO(model_name)

# --- Video Processing Class for WebRTC ---
class YOLOVideoProcessor(VideoProcessorBase):
    def __init__(self, model, confidence_threshold, save_output):
        self.model = model
        self.confidence_threshold = confidence_threshold
        self.save_output = save_output
        self.frames = []
        self.lock = threading.Lock()
        self.p_time = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """Processes a single frame and constructs a side-by-side view."""
        img = frame.to_ndarray(format="bgr24")
        original_frame = img.copy()
        
        # --- FPS Calculation ---
        c_time = time.time()
        fps = 1 / (c_time - self.p_time) if (c_time - self.p_time) > 0 else 0
        self.p_time = c_time
        
        # Inference and plotting
        results = self.model(img, conf=self.confidence_threshold, verbose=False)
        processed_frame = results[0].plot() # .plot() draws boxes and labels
        
        # Draw FPS on the processed frame
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # If saving is enabled, store only the processed frame
        if self.save_output:
            with self.lock:
                self.frames.append(processed_frame)

        # --- Create the side-by-side layout ---
        # Ensure frames are the same height
        h, w, _ = original_frame.shape
        processed_frame = cv2.resize(processed_frame, (w, h))

        # Combine video frames horizontally
        combined_frame = np.hstack((original_frame, processed_frame))

        return av.VideoFrame.from_ndarray(combined_frame, format="bgr24")

# --- Main Application ---
def main():
    st.set_page_config(page_title="Real-Time Object Detection", layout="wide")
    st.title("Real-Time Object Detection")
    st.write("Upload a video file or use your webcam for real-time object detection.")

    st.sidebar.header("Configuration")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    save_output = st.sidebar.checkbox("Save Output Video", key="save_output")
    
    try:
        model = load_yolo_model("yolov8n.pt")
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        return

    st.sidebar.success("YOLOv8 model loaded successfully.")

    source_option = st.sidebar.radio("Select Input Source", ["Upload Video File", "Webcam"])

    # Use a placeholder to ensure the UI is cleared when switching modes
    main_placeholder = st.container()

    if source_option == "Upload Video File":
        with main_placeholder:
            process_uploaded_video(model, confidence_threshold, save_output)
    elif source_option == "Webcam":
        with main_placeholder:
            process_webcam(model, confidence_threshold, save_output)

def process_uploaded_video(model, confidence_threshold, save_output):
    """Handles the logic for processing an uploaded video file."""
    uploaded_file = st.file_uploader("Choose a video...", type=["mp4", "avi", "mov", "mkv"])
    
    if uploaded_file is not None:
        if st.button("Start Processing Video"):
            process_and_display_video(uploaded_file, model, confidence_threshold, save_output)

def process_and_display_video(video_source, model, confidence_threshold, save_output):
    """Core function to process and display video from a file path."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
        tfile.write(video_source.read())
        video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original Video")
        original_frame_placeholder = st.empty()
    with col2:
        st.header("Detections")
        processed_frame_placeholder = st.empty()
    
    fps_placeholder = st.empty()

    video_writer = None
    output_path = ""
    if save_output:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as out_tfile:
            output_path = out_tfile.name
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    p_time = 0
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        
        original_frame_placeholder.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        results = model(frame, conf=confidence_threshold, verbose=False)
        processed_frame = results[0].plot()
        
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        fps_placeholder.text(f"FPS: {int(fps)}")

        if video_writer:
            video_writer.write(processed_frame)
        
        processed_frame_placeholder.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), use_container_width=True)

    cap.release()
    if video_writer:
        video_writer.release()
    os.remove(video_path)
    
    st.success("Video processing complete.")
    if save_output and os.path.exists(output_path):
        with open(output_path, 'rb') as f:
            st.download_button("Download Processed Video", f, file_name="processed_video.mp4")
        os.remove(output_path)

def process_webcam(model, confidence_threshold, save_output):
    """Handles the logic for processing the webcam feed with a side-by-side view."""
    st.header("Webcam Live Feed")
    st.write("Click 'Start' to begin detection.")
    
    # Use columns to create the headers, mimicking the video file layout
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original")
    with col2:
        st.header("Detections")

    processor = YOLOVideoProcessor(model, confidence_threshold, save_output)
    
    ctx = webrtc_streamer(
        key="yolo_webcam_combined",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=lambda: processor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    if save_output and not ctx.state.playing and len(processor.frames) > 0:
        st.info("Saving webcam video...")
        output_path = ""
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as out_tfile:
            output_path = out_tfile.name
        
        frame_height, frame_width, _ = processor.frames[0].shape
        fps = 20
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

        for frame in processor.frames:
            video_writer.write(frame)
        
        video_writer.release()
        
        with open(output_path, 'rb') as f:
            st.download_button("Download Webcam Video", f, file_name="webcam_video.mp4")
        os.remove(output_path)
        processor.frames.clear()

if __name__ == "__main__":
    main()
