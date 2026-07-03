# Realiza o treinamento do modelo EfficientNetV2S com o dataset Plantvillage

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras import models, layers
import os
import json
import joblib


# Abrindo dataset
df = pd.read_csv('Dataset/processed/plantvillage_data.csv')


# Extraindo caminhos das imagens e rótulos
caminhos_imagens = df['file_path'].values
rotulos = df['class'].values


# Convertendo rótulos de texto para número
lable_encoder = LabelEncoder()
rotulos_inteiros = lable_encoder.fit_transform(rotulos)
num_classes = len(lable_encoder.classes_)

print('Total de imagens encontradas: {}'.format(len(caminhos_imagens)))
print('Número de classes: {}'.format(num_classes))

# Dividindo em conjunto de treino e conjunto de teste
x_treino, x_val, y_treino, y_val = train_test_split(
    caminhos_imagens,
    rotulos_inteiros,
    test_size=0.2,
    random_state=1024,
    stratify=rotulos_inteiros
)

# Definindo constantes
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# Definindo função para pré-processamento das imagens
def preprocessar_imagem_efficientnetv2s(caminho, rotulo):
    # Lendo imagem do disco
    img = tf.io.read_file(caminho)

    # Decodificando o JPEG (assume que seja .jpg)
    img = tf.image.decode_jpeg(img, channels=3)

    # Redimensionando imagem para o 224x224
    img = tf.image.resize(img, IMG_SIZE)

    # Convertendo os pixels para o tipo float32
    img = tf.cast(img, tf.float32)

    # Aplica o pré-processamento específico da EfficientNetV2S
    img = tf.keras.applications.efficientnet_v2.preprocess_input(img)

    return img, rotulo

# Criando o dataset do tensorflow
ds_treino = tf.data.Dataset.from_tensor_slices((x_treino, y_treino))
ds_treino = ds_treino.map(preprocessar_imagem_efficientnetv2s, num_parallel_calls=AUTOTUNE)
ds_treino = ds_treino.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocessar_imagem_efficientnetv2s, num_parallel_calls=AUTOTUNE)
ds_val = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Construindo modelo EfficientNetV2S
base_model = EfficientNetV2S(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# Congelando as camadas base para não serem atualizadas no primeiro momento
base_model.treinable = False

# Construindo o modelo sequencial adicionando a base e as novas camadas de classificação
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

# Compilando o modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Adicionando callback para parar o treino mais cedo caso o modelo pare de melhorar
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# Criando o callback para salvar o melhor modelo
checkpoint_callback = ModelCheckpoint(
    filepath='modelo_efficientnetv2s_plantvillage.keras', # Nome do arquivo a ser salvo
    monitor='val_loss',                      # Monitora a perda da validação
    save_best_only=True,                     # Salva apenas se melhorar (sobrescreve o anterior)
    save_weights_only=False,                 # Salva a arquitetura + pesos + otimizador
    mode='min',                              # 'min' porque queremos que a perda diminua
    verbose=1                                # Mostra no console quando salvar
)

# Treinando o modelo
historico = model.fit(
    ds_treino,
    validation_data=ds_val,
    epochs=20,
    callbacks=[early_stopping, checkpoint_callback]
)

# Exportando modelo
model.save('modelo_efficientnetv2s_plantvillage.h5')

# Exportando histórico
with open('historico_treinamento_efficientnetv2s_plantvillage.json', 'w') as f:
    json.dump(historico.history, f)

# Exportando o Label Encoder
joblib.dump(lable_encoder, 'labelencoder_efficientnetv2s_plantvillage.pkl')