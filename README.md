# Car tracker

This project uses OpenCV to recognize and track toy cars in video footage. The program can identify different car types based on their visual features, track their trajectories, and determine the order in which they cross a designated finish line.

## Table of Contents
1. [Project Structure](#project-structure)
2. [Features](#features)
3. [Dependencies](#dependencies)
4. [Getting Started](#getting-started)
5. [Additional Notes](#additional-notes)

## Project Structure

* **`main.py:`** main script for executing the program,
* **`car_recognition.py:`** module containing functions for recognizing car features on image,
* **`car_tracking.py:`** module responsible for tracking car movements in video frames and detecting finish line crossings,
* **`utils.py:`** module with helper functions for image processing tasks,
* **`config.py`:** configuration file containing parameters for image/video paths and car features,
* **`data`:** folder with the image and videos.

## Features

* **Car Recognition:** Identifies car features (color, shape, contours) on the image,
* **Trajectory Tracking:** Tracks car movements throughout the video frames,
* **Finish Line Detection:** Detects when a car crosses the designated finish line and outputs the order of crossings based on car features.

## Dependencies
* Python 3.11
* opencv-python
* numpy


## Getting Started
1. **Clone the repository:**
```bash
git clone https://github.com/bog134/Car-tracker.git
```
2. **Install dependencies:**
```bash
pip install -r requirements.txt
```
3. **Run the program:**
```bash
python3 main.py
```

## Additional notes
* The project is designed to work with specific toy cars and video footage. 
