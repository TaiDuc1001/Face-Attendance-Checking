import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

from scripts.config import DATABASE_PATH, META_PATH
import random
import json
import subprocess

def download_lfw():
    subprocess.run(["wget", "http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"])
    if not os.path.exists(DATABASE_PATH):
        os.makedirs(DATABASE_PATH)
    subprocess.run(["mv", "lfw-deepfunneled.tgz", DATABASE_PATH])
    subprocess.run(["tar", "-xvzf", os.path.join(DATABASE_PATH, "lfw-deepfunneled.tgz"), "-C", DATABASE_PATH])

def isName(path):
    return True if "0" not in os.listdir(path)[0] else False

def gen_id(ids):
    id = random.choice(["SE", "SS"]) + str(random.randint(16, 20)*10000 + random.randint(0, 9999))
    if id in ids:
        print(f"{id} already exists. Generating new ID...")
        return gen_id(ids)
    return id

def get_name(name, data):
    for i in range(len(data)):
        if name == data[i]["Name"]:
            return data[i]["Student Code"]
    print(f"Name {name} not found in data")
    return None

def name_to_code(lfw_path, data):
    for name in os.listdir(lfw_path):
        code = get_name(name, data)
        new_path = os.path.join(lfw_path, code)
        old_path = os.path.join(lfw_path, name)
        os.rename(old_path, new_path)
    print("Name to code conversion done")

def read_metadata(path):
    with open(path, "r") as f:
        return json.load(f)

def write_metadata(people_name, data_name):
    ids = []
    data = []

    for i, name in enumerate(people_name):
        info = {
            "Student Code": gen_id(ids),
            "Name": name,
            "Class Code": [f"SE{1900 + i//40}"]
        }
        data.append(info)
        ids.append(info["Student Code"])

    new_json = os.path.join(META_PATH, f"{data_name}.json")
    with open(new_json, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Metadata written to {data_name}.json")

if __name__ == "__main__":
    data_name = "lfw-deepfunneled"
    lfw_path = os.path.join(DATABASE_PATH, data_name)

    if not os.path.exists(lfw_path):
        download_lfw()

    people_name = os.listdir(lfw_path)
    num_people = len(people_name)
    print(f"Number of people: {num_people}, isName: {isName(lfw_path)}")
    
    if isName(lfw_path):
        write_metadata(people_name=people_name, 
                       data_name=data_name)
        data = read_metadata(os.path.join(META_PATH, f"{data_name}.json"))
        name_to_code(lfw_path=lfw_path, 
                     data=data)