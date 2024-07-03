import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

from scripts.config import DATABASE_PATH, META_PATH, FEATURES_PATH
from model.models import model_dict, mtcnn
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

if __name__ == "__main__":
    lfw_path = "lfw-deepfunneled"
    data = read_metadata(os.path.join(META_PATH, f"{lfw_path}.json"))
    device = findCuda()
    
    codeList = [data[i]["Student Code"] for i in range(len(data))]
    
    for model_name in model_dict:
        data = encode(lfw_path=lfw_path, 
                      codeList=codeList, 
                      model_name=model_name, 
                      device=device)
        
        saveFeatures(lfw_path=lfw_path, 
                     model_name=model_name, 
                     data=data)