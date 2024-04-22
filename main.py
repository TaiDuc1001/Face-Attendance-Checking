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

@timing
def face_match(image_path, data_path):
    img = Image.open(image_path)
    face = mtcnn(img)
    if face is not None:
        embeddings = resnet(face.unsqueeze(0)).detach()
        saved_data = torch.load(data_path)
        embedding_list = saved_data[0]
        name_list = saved_data[1]
        difference_list = []
        for old_embedding in embedding_list:
            similarity = F.cosine_similarity(embeddings, old_embedding).item()
            difference_list.append(similarity)
            
        idx_min = difference_list.index(max(difference_list)) if max(difference_list) >= THRESHOLD else None

        if idx_min is not None:
            return name_list[idx_min], difference_list[idx_min], difference_list
        else:
            return None, None, None
    else:
        return None, None, None


image_files = os.listdir(EXTRACTED_FACES_PATH)
for image_file in image_files:
    image_path = os.path.join(EXTRACTED_FACES_PATH, image_file)
    person, difference, _ = face_match(image_path=image_path, data_path=DATA_PATH)
    if person is not None:
        print(f"Image {image_file}:\tPerson: {person}\tSimilarity: {difference:.4f}\n")
    else:
        print(f"Image {image_file}: no face detected.")
    