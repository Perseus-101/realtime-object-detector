# **Real-Time Object Detection with YOLOv8**

<p align="center">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT">
<img src="https://img.shields.io/badge/Python-3.9%2B-blue.svg" alt="Python Version">
<img src="https://img.shields.io/badge/Streamlit-Ready-brightgreen" alt="Streamlit Ready">
</p>

<p align="center">
<a style="font-size: 2rem; font-weight: bold;" href="https://realtime-object-detector.streamlit.app/">View Live App</a>
</p>
A web application built with Streamlit that uses a pre-trained YOLOv8 model to perform real-time object detection on video files or a live webcam feed.

## **✨ Features**

* **🎥 Multiple Input Sources**: Process either an uploaded video file (.mp4, .avi, .mov) or a live webcam feed.  
* **⚡ Real-Time Detection**: Utilizes the powerful YOLOv8 model for fast and accurate object detection.  
* **🎚️ Adjustable Confidence**: A sidebar slider allows you to filter out detections below a certain confidence threshold.  
* **🖼️ Side-by-Side View**: Displays the original video and the processed video with bounding boxes simultaneously.  
* **⏱️ FPS Counter**: Shows the current processing speed in Frames Per Second.  
* **💾 Save and Download**: Option to save the processed video output and download it directly from the app.

## **🛠️ Tech Stack**

* **Python**: The core programming language.  
* **Streamlit**: For creating the interactive web application UI.  
* **OpenCV**: For video capture and image processing.  
* **Ultralytics (YOLOv8)**: For the object detection model and inference.  
* **Conda**: For environment and package management.

## **⚙️ How to Run Locally**

Follow these steps to set up and run the project on your local machine.

### **1\. Clone the Repository**

First, clone this repository to your local machine:

```bash
git clone https://github.com/Perseus-101/realtime-object-detector.git 
cd realtime-object-detector
```

### **2\. Create a Conda Environment**

It is highly recommended to use a Conda environment to manage dependencies.

```bash
# Create a new environment with Python 3.9  
conda create --name yolo_app python=3.9

# Activate the environment  
conda activate yolo_app
```

### **3\. Install Dependencies**

Install all the necessary Python libraries from the requirements.txt file.

```bash
pip install -r requirements.txt
```

### **4\. Run the Streamlit App**

Once the dependencies are installed, you can run the application with the following command:

```bash
streamlit run main_app.py
```

Your web browser should automatically open to the application's URL (usually http://localhost:8501).

## **📜 License**

This project is licensed under the MIT License \- see the [LICENSE](https://www.google.com/search?q=LICENSE) file for details.