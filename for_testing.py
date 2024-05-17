import subprocess
import os

# Clone project
subprocess.call(["git", "clone", "https://github.com/TaiDuc1001/Face-Attendance-Checking"])
print("Cloned repository successfully.")

# cd to Face-Attendance-Checking
os.chdir("/content/Face-Attendance-Checking")
print("Changed directory to Face-Attendance-Checking.")

# Download my_friends.zip
subprocess.call(["gdown", "1jvix9YP5b5VlxrChSGs9Wgn-veuar-CW"])
print("Downloaded my_friends.zip!")

# Download test.zip
subprocess.call(["gdown", "15oWzWcdRzI3EFboVhjiUWbzj42f5PL2T"])
print("Downloaded test.zip!")

# Download my_friends.json
subprocess.call(["gdown", "1XRR2PqCPTzFJpudG52Pl8Qj5yK56Dp02"])
print("Downloaded my_friends.json!")

# unzip my_friends.zip
subprocess.call(["unzip", "-q", "my_friends.zip"])
print("Unzipped my_friends.zip!")

# unzip test.zip
subprocess.call(["unzip", "-q", "test.zip"])
print("Unzipped test.zip!")

# Remove my_friends.zip
subprocess.call(["rm", "my_friends.zip"])
print("Removed my_friends.zip")

# Remove test.zip
subprocess.call(["rm", "test.zip"])
print("Removed test.zip")

# Rename from my_friends/ to database/
subprocess.call(["mv", "my_friends", "database"])
print("Changed from my_friends/ => database/")

# Rename from my_friends.json to info.json
subprocess.call(["mv", "my_friends.json", "info.json"])
print("Changed from my_friends.json => info.json")

os.mkdir("extracted-faces")
print("Created extracted-faces/")

os.mkdir("features")
print("Created features/")

subprocess.call(["pip", "install", "deepface"])
print("Installed deepface library!")
subprocess.call(["pip", "install", "facenet_pytorch"])
print("Installed facenet_pytorch library!")