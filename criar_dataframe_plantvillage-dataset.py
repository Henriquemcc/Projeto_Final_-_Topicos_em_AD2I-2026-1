import pandas
import pathlib
import os

from numpy.f2py.rules import aux_rules

# Definindo caminho base para a pasta do dataset
dataset_path = pathlib.Path('Dataset/plantvillage_dataset')

# Criando uma lista para guardar os dados
dados = []

# Criando uma lista de extensões de imagens
formatos_imagem = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.PNG']

# Lista de arquivos
arquivos = []

# Adicionando arquivos á lista de arquivos
for formato in formatos_imagem:
    arquivos.extend(dataset_path.rglob(formato))

# Iterando pelos arquivos para extraír o caminho e os rótulos
for filepath in arquivos:
    # Ignorando arquivos ocultos do sistema
    if filepath.name.startswith('.'):
        continue

    # Estrutura esperada: plantvillage_dataset / versão / classe / arquivo.jpg
    versao = filepath.parts[-3] # Ex: color, grayscale ou segmented
    classe_completa = filepath.parts[-2] # Ex: Tomato__Early_blight

    # Separando a planta da doença
    if '___' in classe_completa:
        planta, doenca = classe_completa.split('___', 1)
    else:
        planta, doenca = classe_completa, 'Desconhecido'

    # Adicionando informações na lista
    dados.append({
        'file_path': str(filepath),
        'version': versao,
        'class': classe_completa,
        'plant': planta,
        'disease': doenca
    })

# Criando o dataframe
data_frame = pandas.DataFrame(dados)

# Criando diretório processed
os.makedirs('Dataset/processed', exist_ok=True)

# Salvando em arquivo CSV
data_frame.to_csv('Dataset/processed/plantvillage_data.csv')