# Directory layout for the Prenatal Anemia Screening project

# Create required subfolders within the workspace
import os

workspace = r"C:/Users/gshan/OneDrive/Documents/PrenatalAnemia"
subfolders = [
    "src",
    "data",
    "results",
    "logs",
    "artifacts",
]
for folder in subfolders:
    path = os.path.join(workspace, folder)
    os.makedirs(path, exist_ok=True)
    print(f"Created: {path}")
