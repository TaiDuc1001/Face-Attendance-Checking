import os
import cv2
import pickle

cap = cv2.VideoCapture(0)
background = cv2.imread("Resources/background.png") # Read background

# Set height and width of camera
cap.set(3, 320)
cap.set(4, 240)

# Find paths of mode images
folder_mode_path = 'Resources/Modes'
mode_path_list = os.listdir(folder_mode_path)
img_mode_list = []
for path in mode_path_list:
    img_mode_list.append(cv2.imread(os.path.join(folder_mode_path, path)))

# Load encode file
with open("EncodedImages.pickle", "rb") as file:
    students_encode_with_IDs = pickle.load(file)

encoded_list, student_IDs = students_encode_with_IDs

while True:
    success, img = cap.read()

    background[81:81+240, 27:27+320] = img  # Place camera in background
    background[22:22+317, 404:404+207] = img_mode_list[0] # Place mode in background

    # Show main frame
    cv2.imshow("FACE ATTENDANCE CHECKING", background)


	# Break loop or terminate cam
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
