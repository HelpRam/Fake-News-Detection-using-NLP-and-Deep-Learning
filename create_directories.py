import os

# Folder paths
folders = [
    "data/wel_fake",
    "data/fakenewsnet",
    "scripts/preprocess",
    "scripts/train",
    "models",
    "outputs"
]

# File paths to create
files = [
    "scripts/preprocess/preprocess_welfake.py",
    "scripts/preprocess/preprocess_fakenewsnet.py",
    "scripts/train/train_baselines.py",
    "scripts/train/train_hybrid.py",
    "scripts/train/utils.py",
    "main.py",
    "requirements.txt"
]

# Create folders
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Create empty files
for file_path in files:
    with open(file_path, "a"):
        pass

print(" Full project structure with necessary files created.")
