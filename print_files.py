import os

# Define the root directory
root_dir = "C:/Users/mason/programming/catanatron"

# Define the paths of the pertinent directories
pertinent_dirs = [
    "catanatron_core/catanatron",
    "catanatron_core/catanatron/models",
    "catanatron_experimental/catanatron_experimental",
    "catanatron_experimental/catanatron_experimental/cli",
    "catanatron_experimental/catanatron_experimental/machine_learning",
    "catanatron_gym/catanatron_gym",
    "catanatron_server/catanatron_server",
    "tests",
    "ui/src"
]

# Paths to the output files
directory_structure_file_path = "directory_structure.txt"
files_content_file_path = "files_content.txt"

# Function to check if a file is a text file
def is_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as check_file:
            check_file.read()
        return True
    except:
        return False

# Function to print the directory structure and save file contents
def print_directory_structure(dir, indent=""):
    with open(directory_structure_file_path, "a") as ds_file, open(files_content_file_path, "a", encoding="utf-8") as fc_file:
        for root, dirs, files in os.walk(dir):
            # Exclude node_modules directories
            dirs[:] = [d for d in dirs if d != 'node_modules']
            for name in files:
                relative_path = os.path.relpath(os.path.join(root, name), root_dir).replace("\\", "/")
                is_in_pertinent_dir = any(relative_path.startswith(pd) for pd in pertinent_dirs)
                if is_in_pertinent_dir and not relative_path.endswith(".pyc") and not relative_path.endswith(".svg"):
                    indent_level = len(relative_path.split("/")) - 1
                    indentation = " " * indent_level
                    ds_file.write(f"{indentation}{relative_path}\n")
                    if is_text_file(os.path.join(root_dir, relative_path)):
                        save_file_contents(fc_file, relative_path)

# Function to save file contents to a single output file
def save_file_contents(fc_file, file_path):
    full_path = os.path.join(root_dir, file_path)
    if os.path.exists(full_path):
        fc_file.write(f"\n--- {file_path} ---\n")
        try:
            with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                fc_file.write(f.read())
        except Exception as e:
            fc_file.write(f"\n--- Error reading file: {e} ---\n")
    else:
        fc_file.write(f"\n--- {file_path} not found ---\n")

# Clear the output files if they exist
if os.path.exists(directory_structure_file_path):
    os.remove(directory_structure_file_path)
if os.path.exists(files_content_file_path):
    os.remove(files_content_file_path)

# Ensure the output files are created
open(directory_structure_file_path, "w").close()
open(files_content_file_path, "w").close()

# Call the function to print the directory structure and save file contents
print_directory_structure(root_dir)

print(f"Directory structure has been saved to {directory_structure_file_path}")
print(f"File contents have been saved to {files_content_file_path}")
