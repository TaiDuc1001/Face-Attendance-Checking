import random
import os

# Folder containing subfolders with images
lfw_folder = "lfw-deepfunneled"

# New folder to store 5000 random subfolders
benchmark_folder = "benchmark"

# Get all subfolders within lfw-funneled directory
subfolders = [f.name for f in os.scandir(lfw_folder) if f.is_dir()]

# Select 5000 random subfolders
random_subfolders = random.sample(subfolders, 5000)

# Create the benchmark folder if it doesn't exist
if not os.path.exists(benchmark_folder):
    os.makedirs(benchmark_folder)

# Move each randomly selected subfolder to the benchmark folder
for folder in random_subfolders:
    src = os.path.join(lfw_folder, folder)
    dst = os.path.join(benchmark_folder, folder)
    os.rename(src, dst)

print(f"Successfully moved {len(random_subfolders)} subfolders to {benchmark_folder}")
