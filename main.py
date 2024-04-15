import cv2
import time
import face_recognition
import pickle
import numpy as np
import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import db

droidcam_ip_file = "secret/droidcam_ip.txt"
with open(droidcam_ip_file, "r") as f:
    droidcam_ip = f.readline().strip()


# Init database and bucket in FireBase
cred = credentials.Certificate("secret/serviceAccountKey.json")
firebase_admin.initialize_app(
	cred,
	{
		"databaseURL": "https://face-identification-real-time-default-rtdb.firebaseio.com/",
		"storageBucket": "face-identification-real-time.appspot.com"
	}
)
bucket = storage.bucket()

# Initialize the camera
cap = cv2.VideoCapture(droidcam_ip)

# Check if the camera is opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Wait for 2 seconds
print("Waiting for 4 seconds before capturing...")
time.sleep(4)

# Capture a frame
ret, frame = cap.read()

# If frame is captured successfully, save it to a file
if ret:
    # Define the file name and path
    file_name = "captured_image.jpg"
    
    # Save the captured frame to a file
    cv2.imwrite(file_name, frame)
    print(f"Image saved as {file_name}")

# Release the camera
cap.release()

print("Loading encoded file...")
with open("pickles/EncodedImages.pickle", "rb") as file:
    students_encode_with_IDs = pickle.load(file)
encoded_list, student_IDs = students_encode_with_IDs
print("Encoded file loaded successfully!")

# Convert the frame to RGB (as face_recognition uses RGB format)
rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Find face locations and encodings in the frame
face_locations = face_recognition.face_locations(rgb_frame)
encoded_faces = face_recognition.face_encodings(rgb_frame, face_locations)

if face_locations:
    # Comparing task
    for encoded_face, face_location in zip(encoded_faces, face_locations):
        matches = face_recognition.compare_faces(encoded_list, encoded_face)
        face_distances = face_recognition.face_distance(encoded_list, encoded_face)
        match_index = np.argmin(face_distances)

        if matches[match_index]:
            student_id = student_IDs[match_index]
            student_info = db.reference(f"Students/{student_id}").get()
            ref = db.reference(f"Students/{student_id}")
            student_info["total_attendance"] += 1
            ref.child("total_attendance").set(student_info["total_attendance"])
            print(f"Found {student_info['name']}.")
            top, right, bottom, left = face_location
            frame = cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

# Display the processed frame for 10 seconds or until 'q' key is pressed
cv2.imshow('Processed Frame', frame)
key = cv2.waitKey(10000) & 0xFF  # wait for 10 seconds
if key == ord('q'):  # if 'q' is pressed, exit
    cv2.destroyAllWindows()
else:  # otherwise, wait for the 'q' key
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cv2.destroyAllWindows()
