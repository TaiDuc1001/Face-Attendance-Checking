import random
import os
import cv2
import pickle

# Read annotations from pickle file
TRAIN_ANNOTATION_PATH = "pickles/train_annotations.pickle"
GREEN = (0, 255, 0)

with open(TRAIN_ANNOTATION_PATH, "rb") as file:
	annotations = pickle.load(file)

index = random.randint(0, 100)
image_path = annotations[index]['Image path']
faces = annotations[index]['Face annotations']
print(image_path)
#print(annotations[index]['Face annotations'])

image_path = "WIDER_train/images/" + image_path
image = cv2.imread(image_path)
for face in faces:
	x, y, w, h = face['x'], face['y'], face['w'], face['h']
	cv2.rectangle(image, (x, y), (x+w, y+h), GREEN, 2)

cv2.imshow(image_path, image)
cv2.waitKey(0)
cv2.destroyAllWindows()