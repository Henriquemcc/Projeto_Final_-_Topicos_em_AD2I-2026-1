# Realiza o treinamento do modelo ResNet50 com o dataset Plantvillage

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

df = pd.read_csv('Dataset/processed/plantvillage_data.csv')

# Extraíndo caminhos das imagens e rótulos
caminhos_imagens = df['file_path'].values
rotulos = df['class'].values

# Convertendo rótulos em texto
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
def processar_imagem_resnt50(caminho, rotulo):
    # Lendo a imagem do disco
    img = tf.io.read_file(caminho)

    # Decodificando o JPEG (assume que seja .jpg)
    img = tf.image.decode_jpeg(img, channels=3)

    # Redimensionando imagem para 224x224
    img = tf.image.resize(img, IMG_SIZE)

    # Aplicando o pré-procesamento específico da ResNet50 (normalização adequada)
    img = tf.keras.applications.resnet50.preprocess_input(img)

    return img, rotulo


# Criando dataset do tensorflow
ds_treino = tf.data.Dataset.from_tensor_slices((x_treino, y_treino))
ds_treino = ds_treino.map(processar_imagem_resnt50, num_parallel_calls=AUTOTUNE)
ds_treino = ds_treino.shuffle(buffer_size=1000).batch(BATCH_SIZE).prefetch(AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(processar_imagem_resnt50, num_parallel_calls=AUTOTUNE)
df_val = ds_val.batch(BATCH_SIZE).prefetch(AUTOTUNE)

# Construíndo modelo ResNet50
base_model = tf.keras.applications.ResNet50(
    weights = 'imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Correção 1: 'treinable' alterado para 'trainable'
base_model.trainable = False 

# Construindo modelo final
# Correção 2: '244' alterado para '224'
inputs = tf.keras.Input(shape=(224, 224, 3)) 

x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)

# Camada de saída com neurônios igual ao número de classes
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Correção 3: 'input' alterado para 'inputs'
model = tf.keras.Model(inputs, outputs) 

# Compilando o modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Adicionando callback para parar o treino mais cedo caso o modelo pare de melhorar
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True
)

# Obtendo fotmato das imagens e dos rótulos
for imagens, rotulos in ds_treino.take(1):
    print("Formato das imagens no lote:", imagens.shape)
    print("Formato dos rótulos no lote:", rotulos.shape)
    break

# Treinando modelo
history = model.fit(
    ds_treino,
    validation_data=ds_val,
    epochs=20,
    callbacks=[early_stopping]
)

# Exportando modelo
model.save('modelo_resnet50_plantvillage.h5')