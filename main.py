import torch
import torch.nn.functional as F
from models import model_dict, mtcnn

from PIL import Image
import os
from config import *

class ImageInfo:
    def __init__(self, dataset_path=DATASET_PATH, extracted_faces_path=EXTRACTED_FACES_PATH, model_dict=model_dict):
        self.dataset_path = dataset_path
        self.extracted_faces_path = extracted_faces_path
        self.model_dict = model_dict
        self.voting = {name: {} for name in os.listdir(self.dataset_path)}

    def calculate_score(self, predictions, targets, alpha):
        predictions = torch.tensor(predictions)
        targets = torch.tensor(targets)
        cosine_similarity = F.cosine_similarity(predictions, targets).item()
        l2_distance = F.pairwise_distance(predictions, targets).item()
        score = alpha * cosine_similarity + (1 - alpha) * l2_distance
        return score
    
    def load_available_data(self, data_path):
        data = torch.load(data_path)
        embedding_list = data[0]
        name_list = data[1]
        return embedding_list, name_list

    def face_match(self, image_path, data_path, model_name):
        model = self.model_dict[model_name]["model"]
        if model_name != "vgg-face":
            img = Image.open(image_path)
            face = mtcnn(img)
            if face is not None:
                face = face.unsqueeze(0)
                embeddings = model(face).detach()
            else:
                return None, None

        else:
            embeddings = [model(image_path, model_name='VGG-Face', enforce_detection=False)[0]["embedding"]]

        # === Load data ===
        embedding_list, name_list = self.load_available_data(data_path)

        score_list = []
        for old_embeddings in embedding_list:
            score = self.calculate_score(embeddings, old_embeddings, alpha=model_dict[model_name]["alpha"])
            score_list.append(score)
            
        idx_max = score_list.index(max(score_list))

        if idx_max is not None:
            return name_list[idx_max], score_list[idx_max]
        else:
            return None, None

    def analyze_images(self):
        image_files = os.listdir(self.extracted_faces_path)
        image_info = {}

        for image_file in image_files:
            image_path = os.path.join(self.extracted_faces_path, image_file)
            total_score = 0
            for model_name in self.model_dict:
                person, score = self.face_match(image_path=image_path, data_path=self.model_dict[model_name]["data_path"], model_name=model_name)
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

    def format_result(self, image, person, score, options=[]):
        print(f"Image: {image}")
        if score > THRESHOLD:
            print(f"Person: {person} --- Score: {score}")
        else:
            print(f"Person: Unknown --- Score: {score}")
    
    def print_result(self, options=[]):
        voting = self.analyze_images()
        for image, models in voting.items():
            persons_for_each_image = [data["person"] for data in models.values()]
            isSame = all(person == persons_for_each_image[0] for person in persons_for_each_image)
            if isSame:
                person = persons_for_each_image[0]
                score = models[next(iter(models))]['score']
            else:
                max_score_model = max(models.values(), key=lambda x: x['score'])
                person = max_score_model['person']
                score = max_score_model['score']
            
            self.format_result(image=image, person=person, score=score, options=options)

        return image, person, score


if __name__ == "__main__":
    ImageInfo().print_result()