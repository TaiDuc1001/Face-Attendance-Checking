# Real-Time Face Identification Project

This project is a real-time face identification system that uses computer vision techniques to identify faces from a live camera feed.

## Project Structure

- `encoder.py`: This script encodes the images from `storage/` folder or new students in `stage/` for the face identification system.
- `face_detector.py`: This file extract faces from target picture and save them in `extracted-faces/` folder that will be used by `main.py`.
- `main.py`: This is the main script that take faces from `extracted-faces` then compare with old students in `data.pt`.
- `helper.py`: This one contains timing decorator to get time executed of target functions.
- `config.py`: This one contains constants of paths and for configuration such as `THRESHOLD`.

## How to Run
1. Create folder `stage` to store new students' images. Remember to put each of them in a subfolder with their names and inside is 2-5 images.
2. Create folder `test` to store images that are manually captured (students) to identify. Such as a picture of students in class looking at camera.
3. Create folder `storage` to store students' images. Folder are structured as the `stage`.
4. Run `encoder.py` to extract the embedding feature vectors from students in `storage`. NOTE: if there is new students you want to include, put them in `stage` and run `python encoder.py --isNew` to analyze `stage` folder and concatenate new embedding features to the old `data.pt`.
5. Run `face_detector.py` to extract faces in `test` folder and put them in `extracted-faces`. Press `q` to continue to next face.
6. Run `main.py` to compare faces in `extracted-faces` with original database and print out which person is it.


## Dependencies
Run `pip install ./requirements.txt`

## License
This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
