import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras import models, layers

...

# Construindo modelo EfficientNetV2S
base_model = EfficientNetV2S(
    include_top=False,
    weights='imagenet',
    input_shape=(224, 224, 3)
)

# Congelando as camadas base para nao serem atualizadas no primeiro momento
base_model.treinable = False

# Construindo o modelo sequencial adicionando a base e as novas camadas de classificaçao
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation='softmax')
])

...
