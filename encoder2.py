from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
from torchvision import datasets
from torch.utils.data import DataLoader
from helper import timing
from config import DATASET_PATH, DATA_PATH, STAGE_PATH
import argparse
import shutil
import os

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--isNew', action='store_true', help='Whether to process new data')
args = parser.parse_args()

isNew = args.isNew

mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained="vggface2").eval()


if isNew:
    old_data = torch.load(DATA_PATH)
    old_embedding_list, old_name_list = old_data
else:
    old_embedding_list = []
    old_name_list = []

@timing
def get_embed_data(isNew=True):
    dataset_path = STAGE_PATH if isNew else DATASET_PATH
    dataset = datasets.ImageFolder(dataset_path)
    idx_to_class = {i: c for c, i in dataset.class_to_idx.items()}

    def collate_func(x):
        return x[0]

    loader = DataLoader(dataset, collate_fn=collate_func)
    embeddings_dict = {}
    for img, idx in loader:
        face, prob = mtcnn(img, return_prob=True)
        if face is not None and prob >= 0.9:
            embeddings = resnet(face.unsqueeze(0)).detach()
            if idx not in embeddings_dict:
                embeddings_dict[idx] = [embeddings]
            else:
                embeddings_dict[idx].append(embeddings)

    embedding_list = []
    name_list = []
    for idx, embeddings_list in embeddings_dict.items():
        avg_embedding = torch.stack(embeddings_list).mean(dim=0)
        embedding_list.append(avg_embedding)
        name_list.append(idx_to_class[idx])

    return embedding_list, name_list
    
embedding_list, name_list = get_embed_data(isNew)
if isNew:
    old_embedding_list.extend(embedding_list)
    old_name_list.extend(name_list)
    embedding_list, name_list = old_embedding_list, old_name_list

data = [embedding_list, name_list]
print(f"New name_list: {name_list}")

@timing
def save_data():
    print(f"Save {DATA_PATH}.")
    torch.save(data, DATA_PATH)

@timing
def moving():
    f"""
    Move folders from {STAGE_PATH} to {DATASET_PATH} after embed new students in {STAGE_PATH}
    """

    new_folders = os.listdir(STAGE_PATH)
    new_folders_path = [STAGE_PATH + "/" + folder for folder in new_folders]
    destination_path = DATASET_PATH
    for source_path in new_folders_path:
        shutil.move(source_path, destination_path)
        print(f"Moved {source_path} => {destination_path}")

save_data()
if isNew:
    moving()