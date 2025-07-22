import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import time
import os

# Main function to run the Streamlit application for YOLOv8 object detection.
def main():
    st.set_page_config(page_title="Real-Time Object Detection", layout="wide")
    st.title("Real-Time Object Detection")
    st.write("Upload a video file or use your webcam for real-time object detection.")

    # --- Session State Initialization ---
    if 'run_webcam' not in st.session_state:
        st.session_state.run_webcam = False

    # --- Sidebar for options ---
    st.sidebar.header("Configuration")
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
    save_output = st.sidebar.checkbox("Save Output Video")
    
    # --- Model Selection ---
    model_name = "yolov8n.pt" 
    try:
        @st.cache_resource
        def load_yolo_model(model_name):
            return YOLO(model_name)
        model = load_yolo_model(model_name)
    except Exception as e:
        st.error(f"Error loading YOLO model: {e}")
        st.info("Please ensure you have the yolov8n.pt file or an internet connection to download it.")
        return

    st.sidebar.success(f"Successfully loaded YOLO model: {model_name}")

    # --- Input Source Selection ---
    source_option = st.sidebar.radio("Select Input Source", ["Upload Video File", "Webcam"])

    # --- Video File Upload ---
    if source_option == "Upload Video File":
        st.session_state.run_webcam = False
        
        uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov", "mkv"])
        if uploaded_file is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
                tfile.write(uploaded_file.read())
                video_path = tfile.name
            
            if st.button("Start Processing Video"):
                process_video(video_path, model, confidence_threshold, save_output)
                os.remove(video_path)
        else:
            st.info("Please upload a video file to begin processing.")

    # --- Webcam ---
    elif source_option == "Webcam":
        st.sidebar.header("Webcam Control")
        if st.sidebar.button("Start Webcam"):
            st.session_state.run_webcam = True
        
        if st.sidebar.button("Stop Webcam"):
            st.session_state.run_webcam = False
        
        if st.session_state.run_webcam:
            st.info("Webcam feed is running...")
            process_video(0, model, confidence_threshold, save_output)
        else:
            st.info("Webcam is stopped. Click 'Start Webcam' in the sidebar to begin.")

# Processes a video source for object detection and optionally saves the output.
def process_video(source, model, confidence_threshold, save_output=False):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        st.error("Error: Could not open video source.")
        return

    # --- Video Writer Initialization ---
    video_writer = None
    output_path = ""
    if save_output:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Get FPS from video file, use a default for webcam
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # --- Create a temporary file for the output video ---
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            output_path = tfile.name
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # --- UI Placeholders ---
    col1, col2 = st.columns(2)
    with col1:
        st.header("Original")
        original_frame_placeholder = st.empty()
    with col2:
        st.header("Detections")
        processed_frame_placeholder = st.empty()
    fps_placeholder = st.empty()
    
    p_time = 0
    
    while cap.isOpened():
        if source == 0 and not st.session_state.get('run_webcam', False):
            break

        success, frame = cap.read()
        if not success:
            if source != 0:
                st.write("Video processing finished.")
            break

        c_time = time.time()
        fps = 1 / (c_time - p_time) if (c_time - p_time) > 0 else 0
        p_time = c_time
        
        results = model(frame, stream=True, verbose=False)
        processed_frame = frame.copy()

        for r in results:
            boxes = r.boxes
            for box in boxes:
                conf = box.conf[0]
                
                if conf >= confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cls_id = int(box.cls[0])
                    class_name = model.names[cls_id]

                    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                    label = f'{class_name} {conf:.2f}'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                    label_y = y1 - 10 if y1 - 10 > 10 else y1 + 10
                    cv2.rectangle(processed_frame, (x1, label_y - label_size[1]), (x1 + label_size[0], label_y + 5), (255, 0, 255), cv2.FILLED)
                    cv2.putText(processed_frame, label, (x1, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        # --- Write frame to video file ---
        if video_writer:
            video_writer.write(processed_frame)

        # --- Display Frames and FPS ---
        original_frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
        original_frame_placeholder.image(original_frame_rgb, channels="RGB", use_container_width=True)
        processed_frame_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
        fps_placeholder.text(f"FPS: {int(fps)}")

    cap.release()
    if video_writer:
        video_writer.release()
    cv2.destroyAllWindows()

    # --- Provide Download Link ---
    if save_output and os.path.exists(output_path):
        st.success("Processing complete. You can now download the video.")
        with open(output_path, 'rb') as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
        # Clean up the temporary file after providing the download link
        os.remove(output_path)
    elif not save_output:
        st.success("Processing complete.")

if __name__ == "__main__":
    main()
