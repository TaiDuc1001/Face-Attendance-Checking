from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn.functional as F
from PIL import Image
from helper import timing
from config import DATA_PATH, EXTRACTED_FACES_PATH, THRESHOLD
import os
import numpy as np

# Init mtcnn and resnet model
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained="vggface2").eval()

def calculate_score(predictions, targets, alpha):
    cosine_similarity = F.cosine_similarity(predictions, targets).item()
    l2_distance = F.pairwise_distance(predictions, targets).item()
    score = alpha * cosine_similarity + (1 - alpha) * l2_distance
    return score

@timing
def face_match(image_path, data_path):
    img = Image.open(image_path)
    face = mtcnn(img)
    if face is not None:
        embeddings = resnet(face.unsqueeze(0)).detach()
        saved_data = torch.load(data_path)
        embedding_list = saved_data[0]
        name_list = saved_data[1]
        score_list = []
        for old_embeddings in embedding_list:
            score = calculate_score(embeddings, old_embeddings, alpha=0.8)
            score_list.append(score)
            
        idx_min = score_list.index(max(score_list)) if max(score_list) >= THRESHOLD else None

        if idx_min is not None:
            return name_list[idx_min], score_list[idx_min], score_list
        else:
            return None, None, None
    else:
        return None, None, None


image_files = os.listdir(EXTRACTED_FACES_PATH)
for image_file in image_files:
    image_path = os.path.join(EXTRACTED_FACES_PATH, image_file)
    person, score, _ = face_match(image_path=image_path, data_path=DATA_PATH)
    if person is not None:
        print(f"Image {image_file} --- Person: {person} --- Similarity: {score:.4f}\n")
    else:
        print(f"Image {image_file}: no face detected.\n")
    