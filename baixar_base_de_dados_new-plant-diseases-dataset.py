import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("vipoooool/new-plant-diseases-dataset")

print("Path to dataset files:", path)

# Copiando dataset para dentro do projeto
shutil.copytree(path, os.path.join(os.getcwd(), 'Dataset', 'new_plant_diseases)dataset'))