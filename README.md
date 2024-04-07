# Face Identification Project

This project is a face identification system that uses computer vision techniques to identify faces in images.

## Project Structure

- `draw_boxes_from_annotations.py`: This script reads face annotations from a pickle file and draws bounding boxes around the faces in the images.
- `encoder.py`: This script encodes the images for the face identification system.
- `main.py`: This is the main script that runs the face identification system.
- `process_annotation.py`: This script processes the annotation files and saves them as pickle files for later use.
- `pickles/`: This directory contains the pickle files for the encoded images and the annotations.
- `WIDER_train/`: This directory contains the training images for the face identification system.

## How to Run

1. Run `process_annotation.py` to process the annotation files and save them as pickle files.
2. Run `encoder.py` to encode the images for the face identification system.
3. Run `main.py` to start the face identification system.

## Dependencies

- OpenCV
- Pickle
- os
- random

## Contributing

Contributions are welcome. Please open an issue or submit a pull request.
