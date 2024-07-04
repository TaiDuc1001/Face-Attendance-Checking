import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

from scripts.config import DATABASE_PATH, META_PATH, FEATURES_PATH
from model.models import model_dict
from scripts_experiments.process_lfw import read_metadata

import torch
from tqdm import tqdm
import cv2
from functools import lru_cache

class FindThreshold:
    def __init__(self, data_name, score_path, meta_path=META_PATH, features_path=FEATURES_PATH, database_path=DATABASE_PATH):
        self.data_name = data_name
        self.score_path = score_path
        self.meta_path = meta_path
        self.features_path = features_path
        self.database_path = database_path
        self.data = read_metadata(os.path.join(self.meta_path, f"{self.data_name}.json"))
        self.device = self.get_cuda()
        self.full_code_list = [self.data[i]["Student Code"] for i in range(len(self.data))]

    def get_cuda(self):
        '''
        Get cuda status. Print status
        Return:
            device (String): Device name
        '''
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                print("GPUs detected.")
                return "0,1"
            else:
                print("Found GPU.")
                return "cuda"
        else:
            print("No GPU detected. Using CPU")
            return "cpu"

    def get_code_list_from_class_code(self, class_code):
        '''
        Args:
            class_code (String): Class code of the students
        Return:
            students List<String>: List of student codes in the class
        '''
        students = []
        for i in range(len(self.data)):
            if class_code in self.data[i]["Class Code"]:
                students.append(self.data[i]["Student Code"])
        return students

    def calculate_score(self, y_pred, y_true):
        '''
        Args:
            y_pred (Tensor): Predicted feature
            y_true (Tensor): True feature
        Return:
            score (Tensor): Cosine similarity score
        '''
        try:
            y_pred = y_pred.squeeze()
            y_true = y_true.squeeze()
        except:
            pass
        y_pred = torch.Tensor(y_pred).squeeze().to(self.device)
        y_true = torch.Tensor(y_true).squeeze().to(self.device)
        score = torch.dot(y_pred, y_true) / (torch.norm(y_pred) * torch.norm(y_true))
        return score
    
    def get_score(self, pred, true_feature):
        '''
        Get score between an input (current image) and a student within that class.
        Args:
            pred (Tensor): Predicted feature
            true_feature (List<Tensor>): List of true features of students in the class
        Return:
            average_score (Tensor): Average cosine similarity score
        '''
        scores = [self.calculate_score(pred, true_feature[i]) for i in range(len(true_feature))]
        average_score = sum(scores) / len(scores)
        return average_score

    @lru_cache(maxsize=128)
    def get_indices(self, student_codes):
        '''
        Get indices of students in class from the full list of student codes
        Args:
            student_codes (List<String>): List of student codes in the class
        Return:
            true_code_list (List<int>): List of indices of students in the class, aka true code list
        '''
        true_code_list = []
        for student in student_codes:
            for i, _id in enumerate(self.full_code_list):
                if student == _id:
                    true_code_list.append(i)
        return true_code_list

    @lru_cache(maxsize=128)
    def get_true_features(self, data_path, student_codes):
        '''
        Get true features of students in the class
        Args:
            data_path (String): Path to the data file
            student_codes (List<String>): List of student codes in the class
        Return:
            true_feature_list (List<Tensor>): List of true features of students in the class
            true_code_list (List<String>): List of student codes in the class
        '''
        data = torch.load(data_path)
        true_feature_list = data[0]
        true_feature_list = [true_feature_list[i] for i in self.get_indices(tuple(student_codes))]
        true_code_list = [self.full_code_list[i] for i in self.get_indices(tuple(student_codes))]
        return true_feature_list, true_code_list

    def get_scores_per_input(self, image_path, model_dict, student_codes):
        '''
        Get scores for each student in the class (input -> student)
        Args:
            image_path (String): Path to the image
            model_dict (Dict): Dictionary of models
            student_codes (List<String>): List of student codes in the class
        Return:
            score_list_for_each_person (List<List<Tensor>>): List of scores for each student in the class
            true_code_list (List<String>): List of student codes in the class
        '''
        score_list_for_each_person = []
        for model_name in model_dict:
            pred_feat = self.encode_with_one_model(image_path, model_name)
            true_feature_list, true_code_list = self.get_true_features(os.path.join(self.features_path, self.data_name, f"{model_name}.pt"), tuple(student_codes))
            score_list_for_each_person_each_model = []

            for true_feature in (true_feature_list):
                score = self.get_score(pred_feat, true_feature)
                score_list_for_each_person_each_model.append(score)
            score_list_for_each_person.append(score_list_for_each_person_each_model)
        return score_list_for_each_person, true_code_list
    
    def get_final_scores_per_input(self, score_list_for_each_person, model_dict):
        '''
        Get final scores for each student in the class (input -> student)
        Args:
            score_list_for_each_person (List<List<Tensor>>): List of scores for each student in the class
            model_dict (Dict): Dictionary of models
        Return:
            total_score_list_for_each_image (List<Tensor>): List of final scores for each student in the class
        '''
        total_score_list_for_each_image = []
        number_of_person = len(score_list_for_each_person[0])
        number_of_model = len(score_list_for_each_person)
        for i in range(number_of_person):
            total_score_for_each_person = 0
            for j in range(number_of_model):
                total_score_for_each_person += score_list_for_each_person[j][i] * model_dict[list(model_dict.keys())[j]]["gamma"]
            total_score_list_for_each_image.append(total_score_for_each_person)
        return total_score_list_for_each_image
    
    def encode_with_one_model(self, image_path, model_name):
        '''
        Encode an image with a model
        Args:
            image_path (String): Path to the image
            model_name (String): Name of the model
        Return:
            embeddings (Tensor): Encoded feature
        '''
        model = model_dict[model_name]["model"]
        if model_name == "resnet":
            model = model.to(self.device)
            img = cv2.imread(image_path)
            face = torch.Tensor(img).to(self.device)
            face = face.unsqueeze(0).permute(0, 3, 1, 2)
            embeddings = model(face).detach()
        else:
            embeddings = [model(image_path, model_name=model_name, enforce_detection=False)[0]["embedding"]]
        return embeddings

    def get_pred_code(self, image_path, model_dict, student_codes):
        '''
        Get predicted code for the input image
        Args:
            image_path (String): Path to the image
            model_dict (Dict): Dictionary of models
            student_codes (List<String>): List of student codes in the class
        Return:
            pred_code (String): Predicted student code
            score (Tensor): Score of the prediction
        '''
        score_list_for_each_person, trueCodeList = self.get_scores_per_input(image_path, model_dict, student_codes)
        total_score_list_for_each_image = self.get_final_scores_per_input(score_list_for_each_person, model_dict)
        max_idx = total_score_list_for_each_image.index(max(total_score_list_for_each_image))
        return trueCodeList[max_idx], max(total_score_list_for_each_image)

    def clear_score_file(self):
        '''
        Clear the score file
        '''
        with open(self.score_path, "w") as f:
            f.write("")

    def write_score(self, score, status):
        '''
        Write score to a file
        Args:
            score (Tensor): Score to write
            status (bool): Status of the prediction
        '''
        with open(self.score_path, "a") as f:
            f.write(f"{score},{int(status)}\n")

    def write_scores(self, num_classes, model_dict, isTesting=False, useLoader=True):
        '''
        Write scores to a file
        Args:
            num_classes (int): Number of classes
            model_dict (Dict): Dictionary of models
        '''
        # Clear the score file
        self.clear_score_file()

        for i in range(num_classes):
            num_correct, num_images = 0, 0
            class_code = f"SE{1900 + (i//40)*40 + (i%40)}"
            student_codes = self.get_code_list_from_class_code(class_code)
            student_per_class = student_codes[:10] if isTesting else student_codes
            loader = tqdm(student_per_class, desc=f"Class Code: {class_code}") if useLoader else student_per_class
            for student in loader:
                student_path = os.path.join(self.database_path, self.data_name, student)
                # loader = tqdm(os.listdir(student_path), desc=f"Processing {student}") if useLoader else os.listdir(student_path)
                for image in os.listdir(student_path):
                    image_path = os.path.join(student_path, image)
                    pred_code, score = self.get_pred_code(image_path, model_dict, student_codes)
                    status = pred_code == student
                    num_correct += 1 if status else 0
                    num_images += 1
                    self.write_score(score, status)
            print(f"Num correct: {num_correct}/{num_images} -- Accuracy: {(num_correct / num_images):.4f} -- Class: {class_code}")


if __name__ == "__main__":
    findThres = FindThreshold(data_name="lfw-deepfunneled", score_path="final_scores.txt")
    findThres.write_scores(num_classes=2, model_dict=model_dict, isTesting=True, useLoader=False)
    