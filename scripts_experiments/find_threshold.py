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

def encodeWithOneModel(image_path, model_name, device):
    model = model_dict[model_name]["model"]
    if model_name == "resnet":
        model = model.to(device)
        img = cv2.imread(image_path)
        # face = mtcnn(img).to(device)
        # if face is not None:
        face = torch.Tensor(img).to(device)
        face = face.unsqueeze(0).permute(0, 3, 1, 2)
        embeddings = model(face).detach()
        # else:
            # embeddings = None
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

def matchClassCode(class_code, data):
    students = []
    for i in range(len(data)):
        if class_code in data[i]["Class Code"]:
            students.append(data[i]["Student Code"])
    return students

# Feature of a student from one model
def getTrueFeature(code, data_path):
    data = torch.load(data_path)
    features, codeList = data
    index = codeList.index(code)
    return features[index]

# Features of students in a class from one model
def getTrueFeatures(model_name, student_codes, data_name):
    data_path = os.path.join(FEATURES_PATH, data_name, f"{model_name}.pt")
    features = [getTrueFeature(code, data_path) for code in student_codes]
    return features

def calculateScore(pred, true_feature):
    def calScore(y_pred, y_true):
        return 0
    scores = [calScore(pred, true_feature[i]) for i in range(len(true_feature))]
    return sum(scores) / len(scores)

def getPredCode(image_path, model_dict, data_name, student_codes, device):
    pred_scores = []
    class_scores = []
    for model_name in model_dict:
        gamma = model_dict[model_name]["gamma"]
        pred = encodeWithOneModel(image_path, model_name, device)
        true_features = getTrueFeatures(model_name, student_codes, data_name)
        scores = []
        for true_feature in true_features:
            score = calculateScore(pred, true_feature)
            scores.append(score)
        class_scores.append([scores, gamma])
    
    for i in range(len(class_scores[0])):
        score = 0
        for j in range(len(class_scores)):
            score += class_scores[j][0][i] * class_scores[j][1]
        pred_scores.append(score)
    
    max_index = pred_scores.index(max(pred_scores))
    return student_codes[max_index], pred_scores[max_index]
    
if __name__ == "__main__":
    data_name = "lfw-deepfunneled"
    data = read_metadata(os.path.join(META_PATH, f"{data_name}.json"))
    device = findCuda()
    
    codeList = [data[i]["Student Code"] for i in range(len(data))]
    getEmbedData(data_name, codeList, model_dict, device)

    # class_code = "SE" + str(1900 + 1)
    # num_correct = 0
    # student_codes = matchClassCode(class_code, data)
    
    # for true_code in codeList:
    #     student_path = os.path.join(DATABASE_PATH, data_name, true_code)
    #     for image in os.listdir(student_path):
    #         image_path = os.path.join(student_path, image)
    #         pred_code, score = getPredCode(image_path, model_dict, data_name, student_codes, device)
    #         if pred_code == true_code:
    #             num_correct += 1
    # accuracy = num_correct / len(codeList)
    # print(f"Accuracy: {accuracy:.4f} -- Class: {class_code}")