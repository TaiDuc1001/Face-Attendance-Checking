import face_recognition
import numpy as np
import os
import cv2
import pickle
from datetime import datetime
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
bucket = storage.bucket()

# Color constant
GREEN = (0, 255, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREY = (100, 100, 100)

# Coordinates constant
CAMERA_DISPLAY_Y = 162
CAMERA_DISPLAY_X = 55
MODE_DISPLAY_Y = 44
MODE_DISPLAY_X = 808
STUDENT_IMAGE_DISPLAY_Y = 175
STUDENT_IMAGE_DISPLAY_X = 909
TOTAL_ATTENDANCE_DISPLAY = (861, 125)
MAJOR_DISPLAY = (1006, 550)
STUDENT_ID_DISPLAY = (1006, 493)
STANDING_DISPLAY = (910, 625)
YEAR_DISPLAY = (1025, 625)
STARTING_YEAR = (1125, 625)
NAME_DISPLAY_Y = 808
NAME_DISPLAY_X = 445

# Object dimension (height & width) constant
CAMERA_HEIGHT = 480
CAMERA_WIDTH = 640
MODE_HEIGHT = 633
MODE_WIDTH = 414
STUDENT_IMAGE_HEIGHT = 216
STUDENT_IMAGE_WIDTH = 216

# Configuration constants
TOTAL_ATTENDANCE_FONTSIZE = 1
MAJOR_FONTSIZE = 0.5
STUDENT_ID_FONTSIZE = 0.5
STANDING_FONTSIZE = 0.6
YEAR_FONTSIZE = 0.6
STARTING_YEAR_FONTSIZE = 0.6
NAME_FONTSIZE = 1

cap = cv2.VideoCapture('http://192.168.1.5:4747/video')
background = cv2.imread("Resources/background.png") # Read background

mode_type = 0
counter = 0
student_id = -1

# Find paths of mode images
folder_mode_path = 'Resources/Modes'
mode_path_list = os.listdir(folder_mode_path)
img_mode_list = []
for path in mode_path_list:
    img_mode_list.append(cv2.imread(os.path.join(folder_mode_path, path)))

# Load encode file
print("Loading encoded file...")
with open("pickles/EncodedImages.pickle", "rb") as file:
    students_encode_with_IDs = pickle.load(file)
encoded_list, student_IDs = students_encode_with_IDs
print("Encoded file loaded successfully!")

while True:
    success, img = cap.read()

    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    # Place camera in background
    background[CAMERA_DISPLAY_Y:CAMERA_DISPLAY_Y + CAMERA_HEIGHT, CAMERA_DISPLAY_X:CAMERA_DISPLAY_X + CAMERA_WIDTH] = img
    background[MODE_DISPLAY_Y:MODE_DISPLAY_Y + MODE_HEIGHT, MODE_DISPLAY_X:MODE_DISPLAY_X + MODE_WIDTH] = img_mode_list[mode_type]
    
    # Detecting face
    face_current_frame = face_recognition.face_locations(imgS)
    encoded_current_frame = face_recognition.face_encodings(imgS, face_current_frame)

    if face_current_frame:
        # Comparing task
        for encoded_face, face_location in zip(encoded_current_frame, face_current_frame):
            matches = face_recognition.compare_faces(encoded_list, encoded_face)
            face_distance = face_recognition.face_distance(encoded_list, encoded_face)
            match_index = np.argmin(face_distance)

            if matches[match_index]:
                y1, x2, y2, x1 = face_location

                x1 = x1 * 4 + CAMERA_DISPLAY_X
                x2 = x2 * 4 + CAMERA_DISPLAY_X
                y1= y1 * 4 + CAMERA_DISPLAY_Y
                y2 = y2 * 4 + CAMERA_DISPLAY_Y

                p1, p2 = (x1, y1), (x2, y2)
                background = cv2.rectangle(background, p1, p2, GREEN, 2)
                student_id = student_IDs[match_index]

                if counter == 0:
                    counter = 1
                    mode_type = 1

        if counter != 0:
            if counter == 1:
                # Get information of student
                student_info = db.reference(f"Students/{student_id}").get()

                # Get the image of student from storage
                blob = bucket.get_blob(f"Images/{student_id}.png")
                array = np.frombuffer(blob.download_as_string(), np.uint8)
                student_image = cv2.imdecode(array, cv2.COLOR_BGRA2BGR)

                # Get time
                datetime_object = datetime.strptime(student_info["last_attendance_time"],
                                                "%Y-%m-%d %H:%M:%S")
                second_elapsed = (datetime.now() - datetime_object).total_seconds()

                if second_elapsed > 30:
                    # Update data of total attendance
                    ref = db.reference(f"Students/{student_id}")
                    student_info["total_attendance"] += 1
                    ref.child("total_attendance").set(student_info["total_attendance"])

                    # Update last_attendance_time
                    ref.child("last_attendance_time").set(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                else:
                    mode_type = 3
                    counter = 0
                    background[MODE_DISPLAY_Y:MODE_DISPLAY_Y + MODE_HEIGHT, MODE_DISPLAY_X:MODE_DISPLAY_X + MODE_WIDTH] = img_mode_list[mode_type]

            if mode_type != 3:       
                if 10 < counter < 20:
                    mode_type = 2
                
                background[MODE_DISPLAY_Y:MODE_DISPLAY_Y + MODE_HEIGHT, MODE_DISPLAY_X:MODE_DISPLAY_X + MODE_WIDTH] = img_mode_list[mode_type]

                if counter <= 10:
                    # Display student image
                    background[STUDENT_IMAGE_DISPLAY_Y:STUDENT_IMAGE_DISPLAY_Y + STUDENT_IMAGE_HEIGHT, STUDENT_IMAGE_DISPLAY_X:STUDENT_IMAGE_DISPLAY_X + STUDENT_IMAGE_WIDTH] = student_image

                    # Display total_attendance text
                    cv2.putText(background, str(student_info["total_attendance"]), TOTAL_ATTENDANCE_DISPLAY, cv2.FONT_HERSHEY_COMPLEX, TOTAL_ATTENDANCE_FONTSIZE, WHITE, 1)
                    
                    # Display major text
                    cv2.putText(background, str(student_info["major"]), MAJOR_DISPLAY, cv2.FONT_HERSHEY_COMPLEX, MAJOR_FONTSIZE, WHITE, 1)
                    
                    # Display id text
                    cv2.putText(background, str(student_id), STUDENT_ID_DISPLAY, cv2.FONT_HERSHEY_COMPLEX, STUDENT_ID_FONTSIZE, WHITE, 1)
                    
                    # Display standing text
                    cv2.putText(background, str(student_info["standing"]), STANDING_DISPLAY, cv2.FONT_HERSHEY_COMPLEX, STANDING_FONTSIZE, BLACK, 1)
                    
                    # Display year text
                    cv2.putText(background, str(student_info["year"]), YEAR_DISPLAY, cv2.FONT_HERSHEY_COMPLEX, YEAR_FONTSIZE, BLACK, 1)
                    
                    # Display starting_year text
                    cv2.putText(background, str(student_info["starting_year"]), STARTING_YEAR, cv2.FONT_HERSHEY_COMPLEX, STARTING_YEAR_FONTSIZE, BLACK, 1)
                    
                    # Display name text
                    (w, h), _ = cv2.getTextSize(student_info["name"], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (MODE_WIDTH - w) // 2
                    cv2.putText(background, str(student_info["name"]), (NAME_DISPLAY_Y + offset, NAME_DISPLAY_X), cv2.FONT_HERSHEY_COMPLEX, NAME_FONTSIZE, BLACK, 1)
            
                counter += 1

                if counter >= 20:
                    counter = 0
                    mode_type = 0
                    student_info = []
                    student_image = []
                    background[MODE_DISPLAY_Y:MODE_DISPLAY_Y + MODE_HEIGHT, MODE_DISPLAY_X:MODE_DISPLAY_X + MODE_WIDTH] = img_mode_list[mode_type]
    else:
        mode_type = 0
        counter = 0

    # Show main frame
    cv2.imshow("FACE ATTENDANCE CHECKING", background)


	# Break loop or terminate cam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
