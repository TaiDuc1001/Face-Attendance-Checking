import sys
import os

script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)
import json
import shutil
from scripts.compare_faces import ImageInfo
from config import *
from tqdm import tqdm
import subprocess
test_dir = "test-experiments"

benchmark_info = BENCHMARK_INFO_PATH
benchmark_path = BENCHMARK_PATH
image_info = ImageInfo("SE1900", benchmark_info, benchmark_path)
with open(benchmark_info, "r") as f:
    data = json.load(f)

def match_name(name):
    for student in data:
        if student["Name"] == name:
            return student["ID"]
    return "Unknown"

options = ["avg_before", "avg_after"]
alphas = [0, 1]
methods = ["dlib", "mtcnn"]

def run_detection(image_path, method):
    cmd = ["python", "scripts/face_detector.py", "-f", image_path, "-m", method]
    subprocess.run(cmd)

for option in options:
    for alpha in alphas:
        for method in methods:
            num_true = 0
            for image in (os.listdir(test_dir)):
                image_path = os.path.join(test_dir, image)
                run_detection(image_path, method)
                name = image.split(".")[0]
                true_id = match_name(name)
                for image in os.listdir("extracted-faces"):
                    image_path = os.path.join("extracted-faces", image)
                    pred_id, _ = image_info.face_match(image_path, isBenchmark=True, option=option, alpha=alpha)
                    print(f"True ID: {true_id}, Predicted ID: {pred_id}, {true_id == pred_id}")
                    num_true += true_id == pred_id
            print(f"{option} - Alpha: {alpha} - Method: {method} - Accuracy: {num_true / len(os.listdir(test_dir))}")
            break
        break
    break