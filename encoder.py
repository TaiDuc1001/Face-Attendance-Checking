import face_recognition
import os
import cv2
import time
import pickle
import firebase_admin
from firebase_admin import credentials, storage
from firebase_admin import db

# Init database and bucket in FireBase
cred = credentials.Certificate("secret/serviceAccountKey.json")
firebase_admin.initialize_app(
	cred,
	{
		"databaseURL": "https://face-identification-real-time-default-rtdb.firebaseio.com/",
		"storageBucket": "face-identification-real-time.appspot.com"
	}
)

folder_path = 'Images'
images_path = os.listdir(folder_path)


images_list = []
students_ID = []
for path in images_path:
	images_list.append(cv2.imread(os.path.join(folder_path, path)))
	students_ID.append(os.path.splitext(path)[0])

encoded_list = []

start = time.time()

# Encode part
for img in images_list:
	img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	encoded = face_recognition.face_encodings(img)[0]
	encoded_list.append(encoded)

end = time.time()
print(f"Encoded time: {end - start}")

# Combine student encodes with IDs
student_encoded_with_ID = [encoded_list, students_ID]

with open("pickles/EncodedImages.pickle", "wb") as file:
	pickle.dump(student_encoded_with_ID, file)