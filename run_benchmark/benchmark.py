import subprocess
import os

# Clone project
subprocess.call(["git", "clone", "https://github.com/TaiDuc1001/Face-Attendance-Checking"])
print("Cloned repository successfully.")

# cd to Face-Attendance-Checking
os.chdir("/content/Face-Attendance-Checking")
print("Changed directory to Face-Attendance-Checking.")

# Download benchmark.zip
subprocess.call(["gdown", "1MMdMTO4BydZT2oEHxm7D7gm5DPsD8s7Q"])
print("Downloaded benchmark.zip!")

# Download benchmark.json
subprocess.call(["gdown", "1iq2l27WcT37bqQ9xlbkzEuY178lkdNbv"])
print("Downloaded benchmark.json!")

# unzip benchmark.zip
subprocess.call(["unzip", "-q", "benchmark.zip"])
print("Unzipped benchmark.zip!")

# Remove benchmark.zip
subprocess.call(["rm", "benchmark.zip"])
print("Removed benchmark.zip")

# Rename from benchmark/ to database/
subprocess.call(["mv", "benchmark", "database"])
print("Changed from benchmark/ => database/")

# Rename from benchmark.json to info.json
subprocess.call(["mv", "benchmark.json", "info.json"])
print("Changed from benchmark.json => info.json")

os.mkdir("extracted-faces")
print("Created extracted-faces/")

os.mkdir("features")
print("Created features/")

subprocess.call(["pip", "install", "deepface"])
print("Installed deepface library!")
subprocess.call(["pip", "install", "facenet_pytorch"])
print("Installed facenet_pytorch library!")