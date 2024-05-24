import sys
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(script_dir, '..')
sys.path.append(root_dir)

import json
from scripts.config import BENCHMARK_INFO_PATH
with open(BENCHMARK_INFO_PATH, "r") as f:
  data = json.load(f)

print(data[0]["Class Code"])
for i in range(len(data)):
    data[i]["Class Code"] = [f"SE{1900 + i//40}"]
    print(data[i]["Class Code"])

with open(BENCHMARK_INFO_PATH, "w") as f:
  json.dump(data, f, indent=4)


