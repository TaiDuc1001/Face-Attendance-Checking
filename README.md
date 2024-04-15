# Real-Time Face Identification Project

This project is a real-time face identification system that uses computer vision techniques to identify faces from a live camera feed.

## Project Structure

- `detect_on_camera.py`: This script captures video from a camera, detects faces in the video, and identifies the faces using encoded images.
- `encoder.py`: This script encodes the images for the face identification system and uploads them to Firebase.
- `main.py`: This is the main script that captures a frame from the camera, identifies faces in the frame, and prints the results.
- `pickles/`: This directory contains the pickle files for the encoded images.
- `secret/`: This directory contains sensitive information such as Firebase service account key and DroidCam IP.
- `Images/`: This directory contains the images of students to be identified.

## How to Run

1. Run `encoder.py` to encode the images for the face identification system and upload them to Firebase.
2. Run `main.py` to start the face identification system.

## Dependencies

- OpenCV
- Pickle
- os
- time
- numpy
- face_recognition
- firebase_admin

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.