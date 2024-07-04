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

def encodeWithOneModel(image_path, model_name, device):
    model = model_dict[model_name]["model"]
    if model_name == "resnet":
        model = model.to(device)
        img = cv2.imread(image_path)
        face = torch.Tensor(img).to(device)
        face = face.unsqueeze(0).permute(0, 3, 1, 2)
        embeddings = model(face).detach()
    else:
        embeddings = [model(image_path, model_name=model_name, enforce_detection=False)[0]["embedding"]]
    return embeddings

def saveFeatures(lfw_path, model_name, data):
    parent = os.path.join(FEATURES_PATH, lfw_path)
    if not os.path.exists(parent):
        os.makedirs(parent)
    data_path = os.path.join(parent, f"{model_name}.pt")
    torch.save(data, data_path)
    print(f"Saved {model_name} features to {data_path}")

def findCuda():
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

def encode(lfw_path, codeList, model_name, device):
    features = []
    for code in tqdm(codeList, desc=f"Encoding {lfw_path} with {model_name}"):
        person_path = os.path.join(DATABASE_PATH, lfw_path, code)
        feature = [encodeWithOneModel(os.path.join(person_path, image), model_name, device) for image in os.listdir(person_path)]
        features.append(feature)
    data = [features, codeList]
    print(f"Encoded {lfw_path} with {len(codeList)} people.")
    return data

def getEmbedData(lfw_path, codeList, model_dict, device):
    for model_name in model_dict:
        data = encode(lfw_path=lfw_path, 
                      codeList=codeList, 
                      model_name=model_name, 
                      device=device)
        
        saveFeatures(lfw_path=lfw_path, 
                     model_name=model_name, 
                     data=data)

def getStudentCodesFromClassCode(class_code, data):
    students = []
    for i in range(len(data)):
        if class_code in data[i]["Class Code"]:
            students.append(data[i]["Student Code"])
    return students

def calculateScore(pred, true_feature):
    def calScore(y_pred, y_true):
        try:
            y_pred = y_pred.squeeze()
            y_true = y_true.squeeze()
        except:
            pass
        y_pred = torch.Tensor(y_pred).squeeze()
        y_true = torch.Tensor(y_true).squeeze()
        return torch.dot(y_pred, y_true) / (torch.norm(y_pred) * torch.norm(y_true))
    scores = [calScore(pred, true_feature[i]) for i in range(len(true_feature))]
    return sum(scores) / len(scores)

@lru_cache(maxsize=128)
def getIndices(existing_students, id_list):
    indices = []
    for student in existing_students:
        for i, _id in enumerate(id_list):
            if student == _id:
                indices.append(i)
    return indices

@lru_cache(maxsize=128)
def load_available_data(data_path, student_codes):
    data = torch.load(data_path)
    true_feature_list = data[0]
    id_list = data[1]
    true_feature_list = [true_feature_list[i] for i in getIndices(tuple(student_codes), tuple(id_list))]
    id_list = [id_list[i] for i in getIndices(tuple(student_codes), tuple(id_list))]
    return true_feature_list, id_list

def getScoresPerInput(image_path, model_dict, data_name, student_codes, device):
    score_list_for_each_person = []
    for model_name in model_dict:
        pred_feat = encodeWithOneModel(image_path, model_name, device)
        true_feature_list, trueCodeList = load_available_data(os.path.join(FEATURES_PATH, data_name, f"{model_name}.pt"), tuple(student_codes))
        score_list_for_each_person_each_model = []

        for true_feature in (true_feature_list):
            score = calculateScore(pred_feat, true_feature)
            score_list_for_each_person_each_model.append(score)
        score_list_for_each_person.append(score_list_for_each_person_each_model)
    return score_list_for_each_person, trueCodeList

def getFinalScoresPerInput(score_list_for_each_person, model_dict):
    total_score_list_for_each_image = []
    number_of_person = len(score_list_for_each_person[0])
    number_of_model = len(score_list_for_each_person)

    for i in range(number_of_person):
        total_score_for_each_person = 0
        for j in range(number_of_model):
            total_score_for_each_person += score_list_for_each_person[j][i] * model_dict[list(model_dict.keys())[j]]["gamma"]
        total_score_list_for_each_image.append(total_score_for_each_person)
    return total_score_list_for_each_image

def getPredCode(image_path, model_dict, data_name, student_codes, device):
    score_list_for_each_person, trueCodeList = getScoresPerInput(image_path, model_dict, data_name, student_codes, device)
    total_score_list_for_each_image = getFinalScoresPerInput(score_list_for_each_person, model_dict)
    max_idx = total_score_list_for_each_image.index(max(total_score_list_for_each_image))
    return trueCodeList[max_idx], max(total_score_list_for_each_image)

if __name__ == "__main__":
    data_name = "lfw-deepfunneled"
    data = read_metadata(os.path.join(META_PATH, f"{data_name}.json"))
    device = findCuda()
    
    codeList = [data[i]["Student Code"] for i in range(len(data))]
    getEmbedData(data_name, codeList, model_dict, device)