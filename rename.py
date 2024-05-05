import os

def rename_files(start_index=1, directory=".", printout=False):
    """
    Renames files in a directory and its subdirectories with sequential indices.

    Args:
        start_index: The starting index for renaming (default: 1).
        directory: The directory to start searching from (default: current directory).
    """
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
directory = "storage"
rename_files(start_index, directory, printout=False)

start_index = 1
directory = "storage"
rename_files(start_index, directory, printout=True)


