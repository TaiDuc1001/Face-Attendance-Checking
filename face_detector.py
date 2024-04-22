import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import shutil
import os
from config import TEST_IMAGES_PATH, EXTRACTED_FACES_PATH
import numpy as np

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained='vggface2').eval()


image_path = f'{TEST_IMAGES_PATH}/1.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
boxes, probs = mtcnn.detect(image_rgb)

shutil.rmtree(EXTRACTED_FACES_PATH, ignore_errors=True)
os.makedirs(EXTRACTED_FACES_PATH, exist_ok=True)

if boxes is not None:
    for i, (box, prob) in enumerate(zip(boxes, probs)):
        if prob >= 0.9:
            x1, y1, x2, y2 = map(int, box)
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
            face = image[y1:y2, x1:x2]

            cv2.imwrite(os.path.join(EXTRACTED_FACES_PATH, f"{i}.jpg"), face)
            cv2.imshow('Extracted Face', face)
            key = cv2.waitKey(5000) & 0xFF
            if key == ord('q'):
                continue

cv2.destroyAllWindows()
