import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

from scripts.helper import timing
from scripts.compare_faces import ImageInfo
from scripts.config import BENCHMARK_INFO_PATH, BENCHMARK_PATH
import json
from tqdm import tqdm
import time
from facenet_pytorch import MTCNN, InceptionResnetV1



with open(BENCHMARK_INFO_PATH, 'r') as f:
    data = json.load(f)

def match_name(name):
    for student in data:
        if student["Name"] == name:
            return student["ID"]
    return "Unknown"

image_info = ImageInfo("SE1900", json_path=BENCHMARK_INFO_PATH, dataset_path=BENCHMARK_PATH)


def calculate_accuracy(option, method):
    correct = 0
    mid_time = 0
    start_time = time.time()
    dir = "mtcnn-extracted" if method=="mtcnn" else "dlib-extracted"
    for image in tqdm(os.listdir(dir)):
        image_path = os.path.join(dir, image)
        name = image.split(".")[0]
        true_id = match_name(name)
        _id, _ = image_info.face_match(image_path, isBenchmark=True, option=option)
        if true_id == _id:
            correct += 1
    end_time = time.time()
    mid_time = end_time - start_time
    return correct, mid_time

options = ["avg_before", "avg_after"]
methods = ["mtcnn", "dlib"]
for option in options:
    for method in methods:
        correct, total_time = calculate_accuracy(option, method)
        num_images = len(os.listdir("test-experiments"))
        time_per_image = total_time / num_images
        accuracy = (correct / num_images) * 100
        print(f"Method: {method} -- Algorithm: {option} -- Accuracy: {accuracy:.4f}% -- Time per image: {time_per_image:.4f}s")