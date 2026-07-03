# Realiza o treinamento do modelo GoogleNet com o dataset Plantvillage

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Dense, Flatten, Input, concatenate)
from tensorflow.keras.models import Model
import os
import json
import joblib

# Abrindo dataset
df = pd.read_csv('Dataset/processed/plantvillage_data.csv')

# Extraíndo caminhos das imagens e rótulos
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
def processar_imagem_googlenet(caminho, rotulo):
    # Lendo a imagem do disco
    img = tf.io.read_file(caminho)

    # Decodificando o JPEG (assume que seja .jpg)
    img = tf.io.decode_jpeg(img, channels=3)

    # Redimensionando imagem para 224x224
    img = tf.image.resize(img, IMG_SIZE)

    # Normalizando os pixels para o intervalo entre 0 e 1
    img = img / 255.0

    return img, (rotulo, rotulo, rotulo)

# Criando dataset do tensorflow
ds_treino = tf.data.Dataset.from_tensor_slices((x_treino, y_treino))
ds_treino = ds_treino.map(processar_imagem_googlenet, num_parallel_calls=AUTOTUNE)
ds_treino = ds_treino.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(processar_imagem_googlenet, num_parallel_calls=AUTOTUNE)
ds_val = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Construindo modelo GoogleNet
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):
    # Ramo 1: Convolução 1x1
    path1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', name=f'{name}_1x1')(x)

    # Ramo 2: Convolução 1x1 -> Convolução 3x3
    path2 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', name=f'{name}_3x3_reduce')(x)
    path2 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', name=f'{name}_3x3')(path2)

    # Ramo 3: Convolução 1x1 -> Convolução 5x5
    path3 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', name=f'{name}_5x5_reduce')(x)
    path3 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', name=f'{name}_5x5')(path3)

    # Ramo 4: MaxPool 3x3 -> Convolução 1x1
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name=f'{name}_pool')(x)
    path4 = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', name=f'{name}_pool_proj')(path4)

    # Concatenação dos ramos na dimensão dos canais
    return concatenate([path1, path2, path3, path4], axis=-1, name=name)

def auxiliary_classifier(x, name, num_classes):
    # Redução espacial
    x = AveragePooling2D((5, 5), strides=(3, 3), name=f'{name}_avgpool')(x)
    x = Conv2D(128, (1, 1), padding='same', activation='relu', name=f'{name}_conv')(x)
    x = Flatten(name=f'{name}_flatten')(x)
    x = Dense(1024, activation='relu', name=f'{name}_fc')(x)
    x = Dropout(0.7, name=f'{name}_dropout')(x)
    x = Dense(num_classes, activation='softmax', name=f'{name}_output')(x)
    
    return x

def build_googlenet(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    # Estágio 1
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1_7x7')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool1_3x3')(x)

    # Estágio 2
    x = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv2_1x1')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv2_3x3')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool2_3x3')(x)

    # Estágio 3 (Inception 3a e 3b)
    x = inception_module(x, 64, 96, 128, 16, 32, 32, name='inception_3a')
    x = inception_module(x, 128, 128, 192, 32, 96, 64, name='inception_3b')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool3_3x3')(x)

    # Estágio 4 (Inception 4a a 4e) + Classificador Auxiliar 1
    x = inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_4a')
    aux1 = auxiliary_classifier(x, name='aux1', num_classes=num_classes)

    x = inception_module(x, 160, 112, 224, 24, 64, 64, name='inception_4b')
    x = inception_module(x, 128, 128, 256, 24, 64, 64, name='inception_4c')
    x = inception_module(x, 112, 144, 288, 32, 64, 64, name='inception_4d')
    aux2 = auxiliary_classifier(x, name='aux2', num_classes=num_classes)

    x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_4e')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool4_3x3')(x)

    # Estágio 5 (Inception 5a e 5b)
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_5a')
    x = inception_module(x, 384, 192, 384, 48, 128, 128, name='inception_5b')

    # Classificador principal
    x = AveragePooling2D((7, 7), strides=(1, 1), name='global_avg_pool')(x)
    x = Flatten(name='final_flatten')(x)
    # x = Dropout(0.4, name='final_dropout')(x)
    main_output = Dense(num_classes, activation='softmax', name='main_output')(x)

    # O modelo retorna 3 saídas (1 principal e 2 auxiliares)
    model = Model(inputs=input_layer, outputs=[main_output, aux1, aux2], name='GoogleNet')
    return model


# Construindo modelo
model = build_googlenet(input_shape=(224, 224, 3), num_classes=num_classes)

# Compilando modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss = {
        'main_output': 'sparse_categorical_crossentropy',
        'aux1_output': 'sparse_categorical_crossentropy',
        'aux2_output': 'sparse_categorical_crossentropy'
    },
    loss_weights= {
        'main_output': 1.0,
        'aux1_output': 0.3,
        'aux2_output': 0.3
    },
    metrics= {
        'main_output': 'accuracy',
        'aux1_output': 'accuracy',
        'aux2_output': 'accuracy'
    }
)

# Adicionando callback para parar o treino mais cedo caso o modelo pare de melhorar
early_stopping = EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# Criando o callback para salvar o melhor modelo
checkpoint_callback = ModelCheckpoint(
    filepath='modelo_googlenet_plantvillage.keras', # Nome do arquivo a ser salvo
    monitor='val_loss',                      # Monitora a perda da validação
    save_best_only=True,                     # Salva apenas se melhorar (sobrescreve o anterior)
    save_weights_only=False,                 # Salva a arquitetura + pesos + otimizador
    mode='min',                              # 'min' porque queremos que a perda diminua
    verbose=1                                # Mostra no console quando salvar
)

# Obtendo fotmato das imagens e dos rótulos
for imagens, rotulos in ds_treino.take(1):
    print("Formato das imagens no lote:", imagens.shape)
    print("Formato dos rótulos no lote:", rotulos[0].shape)
    break

# Treinando modelo
historico = model.fit(
    ds_treino,
    validation_data=ds_val,
    epochs=20,
    callbacks=[early_stopping, checkpoint_callback],
)

# Exportando modelo
model.save('modelo_googlenet_plantvillage.h5')

# Exportando histórico
with open('historico_treinamento_googlenet_plantvillage.json', 'w') as f:
    json.dump(historico.history, f)

# Exportando o Label Encoder
joblib.dump(lable_encoder, 'labelencoder_googlenet_plantvillage.pkl')