import torch
import torch.nn.functional as F
from models import model_dict, mtcnn

from PIL import Image
import argparse
import os
import json
from config import *

# === ArgParse ===
parser = argparse.ArgumentParser(description="Compare faces with class target.")
parser.add_argument('--class_code', type=str, help='The class code')
args = parser.parse_args()
class_code = args.class_code

class ImageInfo:
    def __init__(self, dataset_path=DATASET_PATH, extracted_faces_path=EXTRACTED_FACES_PATH, model_dict=model_dict):
        self.dataset_path = dataset_path
        self.extracted_faces_path = extracted_faces_path
        self.model_dict = model_dict
        self.voting = {name: {} for name in os.listdir(self.dataset_path)}
        self.existing_students = self.match_classcode_with_database()

    def match_classcode_with_database(self, json_path=INFO_PATH, class_code=class_code):
        with open(json_path, 'r') as f:
            data = json.load(f)
        existing_students = []
        for student in data:
            if class_code in student["Class Code"]:
                existing_students.append(student)
        return existing_students

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
        id_list = data[1]
        return embedding_list, id_list
    
    def match_name(self, existing_students, id_list):
        indices = []
        for student in existing_students:
            for i, _id in enumerate(id_list):
                if student["ID"] == _id:
                    indices.append(i)
        return indices

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
        embedding_list, id_list = self.load_available_data(data_path)

        # === Match class code with database ===
        score_list = []
        embedding_list = [embedding_list[i] for i in self.match_name(self.existing_students, id_list)]
        id_list = [id_list[i] for i in self.match_name(self.existing_students, id_list)]
        for old_embeddings in embedding_list:
            score = self.calculate_score(embeddings, old_embeddings, alpha=model_dict[model_name]["alpha"])
            score_list.append(score)
            
        idx_max = score_list.index(max(score_list))
        if idx_max is not None:
            return id_list[idx_max], score_list[idx_max]
        else:
            return None, None

    def analyze_images(self):
        image_files = os.listdir(self.extracted_faces_path)
        image_info = {}

        for image_file in image_files:
            image_path = os.path.join(self.extracted_faces_path, image_file)
            total_score = 0
            for model_name in self.model_dict:
                _id, score = self.face_match(image_path=image_path, data_path=self.model_dict[model_name]["data_path"], model_name=model_name)
                if score is not None:
                    total_score += score * self.model_dict[model_name]["gamma"]
                if image_file not in image_info:
                    image_info[image_file] = {}
                a = {
                    "score": score if score is not None else 0,
                    "id": _id,
                }
                if model_name not in image_info[image_file]:
                    image_info[image_file][model_name] = {}
                image_info[image_file][model_name] = a
            self.voting[image_file] = total_score
        return image_info

    def format_result(self, image, _id, score, options=["name", "score"]):
        print(f"Image: {image}")
        if score > THRESHOLD:
            name = [student["Name"] for student in self.existing_students if student["ID"] == _id][0]
            message = f"Person: {name} --- Score: {score}"
            print(message)
        else:
            print(f"Person: Unknown --- Score: {score}")
    
    def print_result(self, options=[]):
        voting = self.analyze_images()
        for image, models in voting.items():
            persons_for_each_image = [data["id"] for data in models.values()]
            isSame = all(person == persons_for_each_image[0] for person in persons_for_each_image)
            if isSame:
                _id = persons_for_each_image[0]
                score = models[next(iter(models))]['score']
            else:
                max_score_model = max(models.values(), key=lambda x: x['score'])
                _id = max_score_model['id']
                score = max_score_model['score']
            
            self.format_result(image=image, _id=_id, score=score, options=options)

if __name__ == "__main__":
    ImageInfo().print_result()