import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

from facenet_pytorch import MTCNN, InceptionResnetV1
import argparse
import cv2
import shutil
import os
import dlib
import torch
from scripts.config import TEST_IMAGES_PATH, EXTRACTED_FACES_PATH

if __name__ == "__main__":
    mtcnn = MTCNN()
    resnet = InceptionResnetV1(pretrained='vggface2').eval()
    detector = dlib.get_frontal_face_detector()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # === ArgParse ===
    parser = argparse.ArgumentParser(description="Extract faces from an image.")
    parser.add_argument("-f", "--file", type=str, help="File name in TEST_IMAGES_PATH folder")
    parser.add_argument("-m", "--method", type=str, help="Method to detect faces: mtcnn or dlib", default="mtcnn")
    args = parser.parse_args()

    if args.file:
        file_name = args.file
        image_path = os.path.join(TEST_IMAGES_PATH, file_name)
    else:
        image_path = os.path.join(TEST_IMAGES_PATH, "1.jpg")
    # === Read image ===
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(image_rgb)
    detections = detector(image_rgb, 1)

    # In case someone else pull this project
    if not os.path.exists(EXTRACTED_FACES_PATH):
        os.mkdir(EXTRACTED_FACES_PATH)

    shutil.rmtree(EXTRACTED_FACES_PATH, ignore_errors=True)
    if not os.path.exists(EXTRACTED_FACES_PATH):
        os.mkdir(EXTRACTED_FACES_PATH)

    face_counted = 0
    choice = args.method
    if choice == "mtcnn":
        print("Using MTCNN...")
        if boxes is not None:
            for i, (box, prob) in enumerate(zip(boxes, probs)):
                i += 1
                if prob >= 0.9:
                    face_counted += 1
                    x1, y1, x2, y2 = map(int, box)
                    x1, y1 = max(x1, 0), max(y1, 0)
                    x2, y2 = min(x2, image.shape[1]), min(y2, image.shape[0])
                    face = image[y1:y2, x1:x2]

                    cv2.imwrite(os.path.join(EXTRACTED_FACES_PATH, f"{i}.jpg"), face)
                    cv2.imshow('Extracted Face', face)
                    key = cv2.waitKey(5000) & 0xFF
                    if key == ord('q'):
                        continue
    elif choice == "dlib":
        print("Using dlib...")
        for i, d in enumerate(detections):
            i += 1
            x1, y1, x2, y2 = d.left(), d.top(), d.right(), d.bottom()
            face = image[y1:y2, x1:x2]
            # Resize face to match the input size of the resnet model
            face = cv2.resize(face, (160, 160))
            
            # Convert face to tensor
            face_tensor = torch.tensor(face, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
            
            # Inference using resnet model
            embeddings = resnet(face_tensor).detach().cpu().numpy()

            # Saving the face
            cv2.imwrite(os.path.join(EXTRACTED_FACES_PATH, f"{i}.jpg"), face)
            cv2.imshow('Extracted Face', face)
            key = cv2.waitKey(5000) & 0xFF
            if key == ord('q'):
                continue
            face_counted += 1

    print(f"Detected {face_counted} faces. Saved in {EXTRACTED_FACES_PATH}.")
    cv2.destroyAllWindows()
