import os

def rename_files(start_index=1, directory=".", printout=False):
    for root, _, files in os.walk(directory):
        for i, filename in enumerate(files):
            new_name = f"{start_index + i}.jpg"
            old_path = os.path.join(root, filename)
            new_path = os.path.join(root, new_name)
            # Print the renaming operation for confirmation (optional)
            if printout:
                print(f"'{old_path}' => '{new_path}'")
            # Uncomment the following line to actually rename the files
            os.rename(old_path, new_path)

start_index = 50
directory = "aita-test-experiments"
rename_files(start_index, directory, printout=False)

start_index = 1
rename_files(start_index, directory, printout=True)


