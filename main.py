import face_recognition
import numpy as np
import os
import cv2
import pickle

GREEN = (0, 255, 0)

cap = cv2.VideoCapture('http://192.168.1.5:4747/video')
background = cv2.imread("Resources/background.png") # Read background

# Set height and width of camera
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 240)

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
    # img = cv2.resize(img, (320, 240))

    imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    background[162:162+480, 55:55+640] = img  # Place camera in background
    background[44:44+633, 808:808+414] = img_mode_list[0] # Place mode in background

    # Detecting face
    face_current_frame = face_recognition.face_locations(imgS)
    encoded_current_frame = face_recognition.face_encodings(imgS, face_current_frame)

    # Comparing task
    for encoded_face, face_location in zip(encoded_current_frame, face_current_frame):
        matches = face_recognition.compare_faces(encoded_list, encoded_face)
        face_distance = face_recognition.face_distance(encoded_list, encoded_face)
        match_index = np.argmin(face_distance)

        if matches[match_index]:
            y1, x2, y2, x1 = face_location
            x1, x2, y1, y2 = x1*4+55, x2*4+55, y1*4+162, y2*4 + 162
            p1, p2 = (x1, y1), (x2, y2)
            background = cv2.rectangle(background, p1, p2, GREEN, 2)

    # Show main frame
    cv2.imshow("FACE ATTENDANCE CHECKING", background)


	# Break loop or terminate cam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
