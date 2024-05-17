import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from deepface import DeepFace
from models import model_dict, mtcnn

from threading import Thread
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import argparse
import shutil
import os
from config import *
from helper import *

# === ArgParse ===
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--isNew', action='store_true', help='Whether to process new data')
args = parser.parse_args()
isNew = args.isNew

# === Configuration ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Load available embeddings ===
def load_available_data(data_path, isNew):
    if isNew:
        old_data = torch.load(data_path)
        old_embedding_list, old_name_list = old_data
    else:
        old_embedding_list = []
        old_name_list = []
    return old_embedding_list, old_name_list


@timing
def get_embed_data(isNew, model_name):
    dataset_path = STAGE_PATH if isNew else DATASET_PATH
    embedding_list = []
    name_list = []
    print(f"Embedding data for {model_name}...")
    if model_name == "resnet":
        dataset = datasets.ImageFolder(dataset_path)
        idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}
        model = model_dict[model_name]["model"]

        def collate_func(batch):
            faces = []
            indices = []
            threads = []

            def process_image(img, idx):
                face, prob = mtcnn(img, return_prob=True)
                if face is not None and prob >= 0.9:
                    faces.append(face)
                    indices.append(idx)

            for img, idx in batch:
                thread = Thread(target=process_image, args=(img, idx))
                threads.append(thread)
                thread.start()

            for thread in threads:
                thread.join()
            return faces, indices        

        loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=collate_func)
        embeddings_dict = {}

        for batch_faces, batch_indices in loader:
            faces = [face.to(device) for face in batch_faces]
            with torch.no_grad():
                faces = [face.unsqueeze(0).to(device) for face in batch_faces]
                if model_name == "resnet":
                    embeddings = model(torch.cat(faces)).cpu()
                    
            for idx, emb in zip(batch_indices, embeddings):
                if idx not in embeddings_dict:
                    embeddings_dict[idx] = [emb]
                else:
                    embeddings_dict[idx].append(emb)


        for idx, embeddings_list in embeddings_dict.items():
            avg_embedding = torch.stack(embeddings_list).mean(dim=0)
            embedding_list.append(avg_embedding)
            name_list.append(idx_to_class[idx])

        return embedding_list, name_list

    else:
        embeddings = {}
        executor = ThreadPoolExecutor(max_workers=WORKERS)
        for person_folder in os.listdir(dataset_path):
            person_embeddings = []
            person_images_folder = os.path.join(dataset_path, person_folder)
            if os.path.isdir(person_images_folder):
                image_paths = [os.path.join(person_images_folder, image_file) for image_file in os.listdir(person_images_folder)]
                futures = [executor.submit(encode_image, image_path, model_name) for image_path in image_paths]
                results = [future.result() for future in futures]
                for result in results:
                    if result is not None:
                        person_embeddings.append(result)
                if person_embeddings:
                    person_embeddings = np.array(person_embeddings)
                    average_embedding = np.mean(person_embeddings, axis=0)
                    embeddings[person_folder] = average_embedding
        return list(embeddings.values()), list(embeddings.keys())

def encode_image(image_path, model_name):
    try:
        embedding = DeepFace.represent(image_path, model_name=model_name, enforce_detection=False)[0]["embedding"]
        return embedding
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def save(model_name, data_path, isNew=isNew):
    old_embedding_list, old_name_list = load_available_data(data_path, isNew)
    embedding_list, name_list = get_embed_data(isNew, model_name)
    if isNew:
        old_embedding_list.extend(embedding_list)
        old_name_list.extend(name_list)
        embedding_list, name_list = old_embedding_list, old_name_list
    data = [embedding_list, name_list]
    print(f"Save {data_path}.")
    torch.save(data, data_path)

def moving():
    f"""
    Move folders from {STAGE_PATH} to {DATASET_PATH} after embedding new students in {STAGE_PATH}
    """
    new_folders = os.listdir(STAGE_PATH)
    new_folders_path = [os.path.join(STAGE_PATH, folder) for folder in new_folders]
    destination_path = DATASET_PATH
    for source_path in new_folders_path:
        shutil.move(source_path, destination_path)
        print(f"Moved {source_path} => {destination_path}")


for model_name in model_dict:
    save(model_name, model_dict[model_name]["data_path"])

if isNew:
    moving()