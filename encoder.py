import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import datasets
from torch.utils.data import DataLoader
from threading import Thread
from helper import timing
import argparse
import shutil
import os
from config import DATASET_PATH, VGG_DATA_PATH, STAGE_PATH, RESNET_DATA_PATH, VGG_WEIGHTS_PATH

# === ArgParse ===
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--isNew', action='store_true', help='Whether to process new data')
args = parser.parse_args()
isNew = args.isNew

# === Configuration ===
device = "cuda" if torch.cuda.is_available() else "cpu"

# === Initialize models ===
mtcnn = MTCNN()
resnet = InceptionResnetV1(pretrained="vggface2").eval().to(device)


# Models dict
model_dict = {
    "resnet": {
        "model": resnet,
        "data_path": RESNET_DATA_PATH
    },
    "vgg": {
        "model": None,
        "data_path": VGG_DATA_PATH
    }
}

# === Load available embeddings ===
@timing
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

    loader = DataLoader(dataset, batch_size=32, collate_fn=collate_func)
    embeddings_dict = {}

    for batch_faces, batch_indices in loader:
        # Move faces to device before processing
        faces = [face.to(device) for face in batch_faces]
        with torch.no_grad():
            faces = [face.unsqueeze(0).to(device) for face in batch_faces]
            if model_name == "vgg":
                features = torch.cat(faces)
                embeddings = model(features)
                embeddings = embeddings.view(embeddings.size(0), -1).cpu()
            elif model == resnet:
                embeddings = resnet(torch.cat(faces)).cpu()
                
        for idx, emb in zip(batch_indices, embeddings):
            if idx not in embeddings_dict:
                embeddings_dict[idx] = [emb]
            else:
                embeddings_dict[idx].append(emb)

    embedding_list = []
    name_list = []
    for idx, embeddings_list in embeddings_dict.items():
        avg_embedding = torch.stack(embeddings_list).mean(dim=0)
        embedding_list.append(avg_embedding)
        name_list.append(idx_to_class[idx])

    return embedding_list, name_list

@timing
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

@timing
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

save('resnet', RESNET_DATA_PATH)
save('vgg', VGG_DATA_PATH)
if isNew:
    moving()