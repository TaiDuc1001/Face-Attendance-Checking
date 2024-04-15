# Real-Time Face Identification Project

This project is a real-time face identification system that uses computer vision techniques to identify faces from a live camera feed.

## Project Structure

- `encoder.py`: This script encodes the images for the face identification system and uploads them to Firebase.
- `main.py`: This is the main script that captures a frame from the camera, identifies faces in the frame, and prints the results.
- `Images/`: This directory contains the images of students to be identified.

## How to Run
If there is new student:
1. Put their images on `Images/` folder.
2. Manually add their information on `info.csv`.
3. Run `encoder.py` to encode the images for the face identification system and upload them to Firebase.
4. Run `main.py` to start the face identification system. 
   
If not:
1. Run `main.py` to detect faces, the `total_attendance` on firebase will automatically update.
2. If you want to download all data on firebase, change `FIREBASE_TO_CSV` in `upload_data_to_database.py` to True and run that file (it will update your csv file). 

## Dependencies
Run `pip install ./requirements.txt`

## License

This project is licensed under the Apache 2.0 License - see the [LICENSE](LICENSE) file for details.
