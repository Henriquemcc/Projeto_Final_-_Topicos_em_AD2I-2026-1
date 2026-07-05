import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

...

# Construindo modelo ResNet50
base_model = tf.keras.applications.ResNet50(
    weights = 'imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# Congelando as camadas base para nao serem atualizadas no primeiro momento
base_model.trainable = False 

# Construindo modelo final
inputs = tf.keras.Input(shape=(224, 224, 3)) 

x = base_model(inputs, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)

# Camada de saida com neurônios igual ao número de classes
outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)

# Construindo modelo
model = tf.keras.Model(inputs, outputs)

...
