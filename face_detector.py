import cv2
from facenet_pytorch import MTCNN

mtcnn = MTCNN()
image_path = 'Students_In_Class_For_Testing/1.jpg'
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
boxes, _ = mtcnn.detect(image_rgb)

if boxes is not None:
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
        face = image[y1:y2, x1:x2]
        cv2.imshow('Extracted Face', face)
        key = cv2.waitKey(5000) & 0xFF
        if key == ord('q'):
            continue

cv2.destroyAllWindows()
