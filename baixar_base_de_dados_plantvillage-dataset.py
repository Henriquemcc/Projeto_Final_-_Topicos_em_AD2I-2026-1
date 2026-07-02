import kagglehub
import shutil
import os

# Download latest version
path = kagglehub.dataset_download("abdallahalidev/plantvillage-dataset")

print("Path to dataset files:", path)

# Copiando dataset para dentro do projeto
shutil.copytree(os.path.join(path, 'plantvillage dataset'), os.path.join(os.getcwd(), 'Dataset', 'plantvillage_dataset'))