import torch
import torch.nn.functional as F
from models import model_dict, mtcnn

from PIL import Image
from functools import lru_cache
import argparse
import os
import json
from config import *

class ImageInfo:
    def __init__(self, class_code, json_path, dataset_path=DATASET_PATH, extracted_faces_path=EXTRACTED_FACES_PATH, model_dict=model_dict):
        self.dataset_path = dataset_path
        self.extracted_faces_path = extracted_faces_path
        self.model_dict = model_dict
        self.class_code = class_code
        self.voting = {name: {} for name in os.listdir(self.dataset_path)}
        self.existing_students = self.match_classcode_with_database(json_path)

    def match_classcode_with_database(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        existing_students = []
        for student in data:
            if self.class_code in student["Class Code"]:
                existing_students.append(student)
        return existing_students

    def calculate_score(self, predictions, targets, alpha):
        if predictions is None:
            return 0
        predictions = torch.tensor(predictions)
        targets = torch.tensor(targets)
        cosine_similarity = F.cosine_similarity(predictions, targets).item()
        l2_distance = F.pairwise_distance(predictions, targets).item()
        score = alpha * cosine_similarity + (1 - alpha) * l2_distance
        return score

    # @lru_cache(maxsize=1024)
    def load_available_data(self, data_path):
        data = torch.load(data_path)
        original_embedding_list = data[0]
        id_list = data[1]
        original_embedding_list = [original_embedding_list[i] for i in self.match_name(self.existing_students, id_list)]
        id_list = [id_list[i] for i in self.match_name(self.existing_students, id_list)]
        return original_embedding_list, id_list

    def match_name(self, existing_students, id_list):
        indices = []
        for student in existing_students:
            for i, _id in enumerate(id_list):
                if student["ID"] == _id:
                    indices.append(i)
        return indices
    
    def calculate_embedding_target_image_with_correspond_model(self, image_path, model_name):
        model = self.model_dict[model_name]["model"]
        if model_name == "resnet":
            img = Image.open(image_path)
            face = mtcnn(img)
            if face is not None:
                face = face.unsqueeze(0)
                embeddings = model(face).detach()
            else:
                embeddings = None
        else:
            embeddings = [model(image_path, model_name=model_name, enforce_detection=False)[0]["embedding"]]

        return embeddings

    def face_match(self, image_path):
        score_list_for_each_person = []
        for model_name in self.model_dict:
            # Calculate embedding of target face
            embedding = self.calculate_embedding_target_image_with_correspond_model(image_path, model_name)
            original_embedding_list, id_list = self.load_available_data(model_dict[model_name]["data_path"])
            score_list_for_each_person_each_model = []

            for original_embedding in original_embedding_list:
                score = self.calculate_score(embedding, original_embedding, alpha=model_dict[model_name]["alpha"])
                score_list_for_each_person_each_model.append(score)
            score_list_for_each_person.append(score_list_for_each_person_each_model)
        total_score_list_for_each_image = []
        number_of_person = len(score_list_for_each_person[0])
        number_of_model = len(score_list_for_each_person)

        for i in range(number_of_person):
            total_score_for_each_person = 0
            for j in range(number_of_model):
                total_score_for_each_person += score_list_for_each_person[j][i] * model_dict[model_name]["gamma"]
            total_score_list_for_each_image.append(total_score_for_each_person)

        max_score = max(total_score_list_for_each_image)
        idx_max = total_score_list_for_each_image.index(max_score)
        id_of_that_max_score = id_list[idx_max]
        if max_score > THRESHOLD:
            return id_of_that_max_score, max_score
        else:
            return None, max_score

    def analyze_images(self):
        image_files = os.listdir(self.extracted_faces_path)
        image_info = {}

        for image_file in image_files:
            image_path = os.path.join(self.extracted_faces_path, image_file)
            _id, score = self.face_match(image_path=image_path)
            if image_file not in image_info:
                image_info[image_file] = {}
            image_info[image_file] = {
                "score": score,
                "id": _id
            }
        return image_info

    def format_result(self, image, _id, score, options=["name", "score"]):
        print(f"Image: {image}")
        if score is not None:
            name = [student["Name"] for student in self.existing_students if student["ID"] == _id][0]
            message = f"Person: {name} --- Score: {score}"
            print(message)
        else:
            print(f"Person: Unknown --- Score: Unknown")
    
    def print_result(self, options=[]):
        voting = self.analyze_images()
        for image in voting:
            _id = voting[image]["id"]
            score = voting[image]["score"]
            self.format_result(image=image, _id=_id, score=score, options=options)

if __name__ == "__main__":
    # === ArgParse ===
    parser = argparse.ArgumentParser(description="Compare faces with class target.")
    parser.add_argument('--class_code', type=str, help='The class code')
    args = parser.parse_args()
    class_code = args.class_code
    ImageInfo(class_code=class_code, json_path=INFO_PATH).print_result()