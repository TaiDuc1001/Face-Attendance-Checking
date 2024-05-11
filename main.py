import torch
from deepface import DeepFace
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch.nn.functional as F

from PIL import Image
from helper import timing
import os
from config import *

# Init mtcnn, resnet and vgg16 model
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained="vggface2").eval()
vggface = DeepFace.represent

# Models dict
model_dict = {
    "resnet": {
        "model": resnet,
        "data_path": RESNET_DATA_PATH,
        "alpha": 0.8,
        "gamma": 0.8,
    },
    "vgg-face": {
        "model": vggface,
        "data_path": VGG_FACE_DATA_PATH,
        "alpha": 0.6,
        "gamma": 0.2,
    }
}

def calculate_score(predictions, targets, alpha):
    predictions = torch.tensor(predictions)
    targets = torch.tensor(targets)
    cosine_similarity = F.cosine_similarity(predictions, targets).item()
    l2_distance = F.pairwise_distance(predictions, targets).item()
    score = alpha * cosine_similarity + (1 - alpha) * l2_distance
    return score

def face_match(image_path, data_path, model_name):
    model = model_dict[model_name]["model"]
    if model_name != "vgg-face":
        img = Image.open(image_path)
        face = mtcnn(img)
        if face is not None:
            face = face.unsqueeze(0)
            embeddings = model(face).detach()
        else:
            return None, None, None

    else:
        embeddings = [DeepFace.represent(image_path, model_name='VGG-Face', enforce_detection=False)[0]["embedding"]]

    # === Load data ===
    saved_data = torch.load(data_path)
    embedding_list = saved_data[0]
    name_list = saved_data[1]

    score_list = []
    for old_embeddings in embedding_list:
        score = calculate_score(embeddings, old_embeddings, alpha=0.8)
        score_list.append(score)
        
    idx_min = score_list.index(max(score_list))

    if idx_min is not None:
        return name_list[idx_min], score_list[idx_min], score_list
    else:
        return None, None, None


class ImageInfo:
    def __init__(self, dataset_path=DATASET_PATH, extracted_faces_path=EXTRACTED_FACES_PATH, model_dict=model_dict):
        self.dataset_path = dataset_path
        self.extracted_faces_path = extracted_faces_path
        self.model_dict = model_dict
        self.voting = {name: {} for name in os.listdir(self.dataset_path)}

    def analyze_images(self):
        image_files = os.listdir(self.extracted_faces_path)
        image_info = {}

        for image_file in image_files:
            image_path = os.path.join(self.extracted_faces_path, image_file)
            total_score = 0
            for model_name in self.model_dict:
                person, score, _ = face_match(image_path=image_path, data_path=self.model_dict[model_name]["data_path"], model_name=model_name)
                if score is not None:
                    total_score += score * self.model_dict[model_name]["gamma"]
                if image_file not in image_info:
                    image_info[image_file] = {}
                a = {
                    "score": score if score is not None else 0,
                    "person": person,
                }
                if model_name not in image_info[image_file]:
                    image_info[image_file][model_name] = {}
                image_info[image_file][model_name] = a
            self.voting[image_file] = total_score
        return image_info
    
    def print_result(self):
        voting = self.analyze_images()
        for image, models in voting.items():
            print(f"Image: {image}")
            persons_for_each_image = [data["person"] for data in models.values()]
            isSame = all(person == persons_for_each_image[0] for person in persons_for_each_image)
            if isSame:
                person = persons_for_each_image[0]
                score = models[next(iter(models))]['score']
            else:
                max_score_model = max(models.values(), key=lambda x: x['score'])
                person = max_score_model['person']
                score = max_score_model['score']
                
            if score > THRESHOLD:
                print(f"Person: {person} --- Score: {score}")
            else:
                print(f"Person: Unknown --- Score: {score}")

if __name__ == "__main__":
    ImageInfo().print_result()