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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

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
        self.last_class_code = self.get_final_class_code()
        self.sim_path = self.score_path.split(".")[0] + "_sim" + ".txt"
        self.dis_path = self.score_path.split(".")[0] + "_dis" + ".txt"

    def get_final_class_code(self):
        '''
        Get final class code
        Return:
            digits (int): Digits of the final class code
        '''
        last_code = self.data[-1]["Class Code"][0]
        digits = int(last_code[2:])
        return digits + 1

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
            score_similarity (Tensor): Cosine similarity score
            score_distance (Tensor): Euclidean distance score
        '''
        try:
            y_pred = y_pred.squeeze()
            y_true = y_true.squeeze()
        except:
            pass
        y_pred = torch.Tensor(y_pred).squeeze().to(self.device)
        y_true = torch.Tensor(y_true).squeeze().to(self.device)
        score_similarity = torch.dot(y_pred, y_true) / (torch.norm(y_pred) * torch.norm(y_true))
        score_distance = torch.dist(y_pred, y_true)
        return score_similarity, score_distance
    
    def get_score(self, pred, true_feature, option="avg_after"):
        '''
        Get score between an input (current image) and a student within that class.
        Args:
            pred (Tensor): Predicted feature
            true_feature (List<Tensor>): List of true features of students in the class
        Return:
            avg_score_similarity (Tensor): Average cosine similarity score
            avg_score_distance (Tensor): Average euclidean distance score
        '''
        if option == "avg_after":
            scores = [self.calculate_score(pred, true_feature[i]) for i in range(len(true_feature))]
            avg_score_similarity = sum([score[0] for score in scores]) / len(scores)
            avg_score_distance = sum([score[1] for score in scores]) / len(scores)
        elif option == "avg_before":
            true_feature_tensors = [torch.tensor(f) if not isinstance(f, torch.Tensor) else f for f in true_feature]

            # Now you can safely stack and calculate the mean
            avg_feature = torch.mean(torch.stack(true_feature_tensors), dim=0)
            avg_score_similarity, avg_score_distance = self.calculate_score(pred, avg_feature)
        else:
            raise ValueError("Option must be either 'avg_before' or 'avg_after'")
        
        return avg_score_similarity, avg_score_distance

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

    def get_scores_per_input(self, image_path, model_dict, student_codes, option):
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
        score_similarity_list_for_each_person = []
        score_distance_list_for_each_person = []
        for model_name in model_dict:
            pred_feat = self.encode_with_one_model(image_path, model_name)
            true_feature_list, true_code_list = self.get_true_features(os.path.join(self.features_path, self.data_name, f"{model_name}.pt"), tuple(student_codes))
            score_similarity_list_for_each_person_each_model = []
            score_distance_list_for_each_person_each_model = []

            for true_feature in (true_feature_list):
                score_similarity, score_distance = self.get_score(pred_feat, true_feature, option=option)
                score_similarity_list_for_each_person_each_model.append(score_similarity)
                score_distance_list_for_each_person_each_model.append(score_distance)
            score_similarity_list_for_each_person.append(score_similarity_list_for_each_person_each_model)
            score_distance_list_for_each_person.append(score_distance_list_for_each_person_each_model)
        return score_similarity_list_for_each_person, score_distance_list_for_each_person, true_code_list
    
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

    def get_pred_code(self, image_path, model_dict, student_codes, option="avg_after"):
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
        score_similarity_list_for_each_person, score_distance_list_for_each_person, trueCodeList = self.get_scores_per_input(image_path, model_dict, student_codes, option=option)
        total_score_similarity_list_for_each_image = self.get_final_scores_per_input(score_similarity_list_for_each_person, model_dict)
        total_score_distance_list_for_each_image = self.get_final_scores_per_input(score_distance_list_for_each_person, model_dict)
        max_idx_sim = total_score_similarity_list_for_each_image.index(max(total_score_similarity_list_for_each_image))
        min_idx_dis = total_score_distance_list_for_each_image.index(min(total_score_distance_list_for_each_image))
        return trueCodeList[max_idx_sim], trueCodeList[min_idx_dis], max(total_score_similarity_list_for_each_image), min(total_score_distance_list_for_each_image)

    def clear_score_file(self):
        '''
        Clear the score file
        '''
        with open(self.sim_path, "w") as f:
            f.write("")
        with open(self.dis_path, "w") as f:
            f.write("")

    def write_score(self, score_sim, score_dis, status_sim, status_dis):
        '''
        Write score to a file
        Args:
            score (Tensor): Score to write
            status (bool): Status of the prediction
        '''
        with open(self.sim_path, "a") as f:
            f.write(f"{score_sim},{int(status_sim)}\n")
        with open(self.dis_path, "a") as f:
            f.write(f"{score_dis},{int(status_dis)}\n")

    def write_scores(self, num_classes, model_dict, option, isTesting=False, useLoader=True):
        '''
        Write scores to a file
        Args:
            num_classes (int): Number of classes
            model_dict (Dict): Dictionary of models
        '''
        # Clear the score file
        self.clear_score_file()

        num_classes = num_classes if num_classes is not None else self.last_class_code

        for i in range(num_classes):
            num_correct_sim, num_correct_dis, num_images = 0, 0, 0
            class_code = f"SE{1900 + (i//40)*40 + (i%40)}"
            student_codes = self.get_code_list_from_class_code(class_code)
            try:
                student_per_class = student_codes[:10] if isTesting else student_codes
                loader = tqdm(student_per_class, desc=f"Class Code: {class_code}") if useLoader else student_per_class
                for student in loader:
                    student_path = os.path.join(self.database_path, self.data_name, student)
                    for image in os.listdir(student_path):
                        image_path = os.path.join(student_path, image)
                        pred_code_sim, pred_code_dis, score_sim, score_dis = self.get_pred_code(image_path, model_dict, student_codes, option=option)
                        status_sim = pred_code_sim == student
                        status_dis = pred_code_dis == student
                        num_correct_sim += 1 if status_sim else 0
                        num_correct_dis += 1 if status_dis else 0
                        num_images += 1
                        self.write_score(score_sim, score_dis, status_sim, status_dis)
                print(f"Sim: Num correct: {num_correct_sim}/{num_images} -- Accuracy: {(num_correct_sim / num_images):.4f} -- Class: {class_code}")
                print(f"Dis: Num correct: {num_correct_dis}/{num_images} -- Accuracy: {(num_correct_dis / num_images):.4f} -- Class: {class_code}")
            except:
                print(f"Error in class: {class_code}")
    def get_score_from_file(self):
        '''
        Get scores from a file
        Return:
            scores_sim (List<(float, int)>): List of scores and status of the prediction (similarity)
            scores_dis (List<(float, int)>): List of scores and status of the prediction (distance)
        '''
        def read_score(score_path):
            with open(score_path, "r") as f:
                lines = f.readlines()
            scores = [(float(line.split(",")[0]), int(line.split(",")[1])) for line in lines]
            return scores
        
        scores_sim = read_score(self.sim_path)
        scores_dis = read_score(self.dis_path)
        return scores_sim, scores_dis

    def read_scores_to_dataframe(self):
        '''
        Read scores from a file to a DataFrame
        Return:
            sim_df (DataFrame): DataFrame containing similarity scores
            dis_df (DataFrame): DataFrame containing distance scores
        '''
        sim_df = pd.read_csv(self.sim_path, header=None, names=['score', 'status'])
        dis_df = pd.read_csv(self.dis_path, header=None, names=['score', 'status'])
        return sim_df, dis_df

    def count_thres_one(self, df, thresholds):
        '''
        Read the DataFrame and return the number of lines where 'thres-0.0' value is 1.
        Args:
            df (DataFrame): The DataFrame to process.
        Return:
            accuracies (List<float>): List of accuracies for each threshold value.
        '''
        accuracies = []
        for thres in thresholds:
            count = df[df[f'thres-{thres}'] == 1].shape[0]
            accuracy = count / df.shape[0]
            accuracies.append(round(accuracy, 4))
        accuracy_df = pd.DataFrame({'threshold': thresholds, 'accuracy': accuracies})
        return accuracy_df
    
    def create_cases(self, sim_df, dis_df):
        '''
        Create cases for each threshold value.
        Args:
            sim_df (DataFrame): The DataFrame to process.
        Return:
            sim_df (DataFrame): The DataFrame with the 'thres-x' columns added.
        '''
        sim_thresholds, dis_thresholds = [], []
        sim_thresholds.extend([(0.01 * i + 6*0.1) for i in range(20)])
        sim_thresholds.extend([(0.005 * i + 8*0.1) for i in range(20)])
        sim_thresholds = [round(thres, 2) for thres in sim_thresholds]

        dis_thresholds.extend([(i*0.1 + 3) for i in range(2*10+1)])
        for thres in sim_thresholds:
            sim_df[f'thres-{thres}'] = np.where((sim_df['score'] > thres) & (sim_df['status'] == 1), 1, 0)
        for thres in dis_thresholds:
            dis_df[f'thres-{thres}'] = np.where((dis_df['score'] < thres) & (dis_df['status'] == 1), 1, 0)
        return sim_df, dis_df, sim_thresholds, dis_thresholds

    def plot_score(self, accuracy_sim, accuracy_dis):
        '''
        Plot the accuracy scores for different threshold values.
        Args:
            accuracy_df (DataFrame): DataFrame containing threshold values and corresponding accuracies.
        '''
        fig, axs = plt.subplots(1, 2, figsize=(15, 6))  # Create 1 row, 2 columns subplot structure

        # Plot for similarity scores
        sns.lineplot(data=accuracy_sim, x='threshold', y='accuracy', ax=axs[0], color='green')
        axs[0].set_title('Accuracy vs. Threshold (Similarity)')
        axs[0].set_xlabel('Threshold')
        axs[0].set_ylabel('Accuracy')
        axs[0].grid()

        # Plot for dissimilarity scores
        sns.lineplot(data=accuracy_dis, x='threshold', y='accuracy', ax=axs[1], color='red')
        axs[1].set_title('Accuracy vs. Threshold (Distance)')
        axs[1].set_xlabel('Threshold')
        axs[1].set_ylabel('Accuracy')
        axs[1].grid()

        plt.show()


if __name__ == "__main__":
    findThres = FindThreshold(data_name="lfw-deepfunneled", score_path="test_scores.txt")
    findThres.write_scores(num_classes=1, model_dict=model_dict, option="avg_before", isTesting=True, useLoader=True)
    # sim_df, dis_df = findThres.read_scores_to_dataframe()
    # sim_df, dis_df, sim_thresholds, dis_thresholds = findThres.create_cases(sim_df, dis_df)
    # accuracy_sim = findThres.count_thres_one(sim_df, sim_thresholds)
    # accuracy_dis = findThres.count_thres_one(dis_df, dis_thresholds)
    # findThres.plot_score(accuracy_sim, accuracy_dis)