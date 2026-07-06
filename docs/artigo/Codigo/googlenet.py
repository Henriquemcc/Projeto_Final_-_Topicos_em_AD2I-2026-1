import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Dense, Flatten, Input, concatenate)
from tensorflow.keras.models import Model

...

# Construindo modelo GoogleNet
def inception_module(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool_proj, name=None):
    # Ramo 1: Convolucao 1x1
    path1 = Conv2D(filters_1x1, (1, 1), padding='same', activation='relu', name=f'{name}_1x1')(x)

    # Ramo 2: Convolucao 1x1 -> Convolucao 3x3
    path2 = Conv2D(filters_3x3_reduce, (1, 1), padding='same', activation='relu', name=f'{name}_3x3_reduce')(x)
    path2 = Conv2D(filters_3x3, (3, 3), padding='same', activation='relu', name=f'{name}_3x3')(path2)

    # Ramo 3: Convolucao 1x1 -> Convolucao 5x5
    path3 = Conv2D(filters_5x5_reduce, (1, 1), padding='same', activation='relu', name=f'{name}_5x5_reduce')(x)
    path3 = Conv2D(filters_5x5, (5, 5), padding='same', activation='relu', name=f'{name}_5x5')(path3)

    # Ramo 4: MaxPool 3x3 -> Convolucao 1x1
    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name=f'{name}_pool')(x)
    path4 = Conv2D(filters_pool_proj, (1, 1), padding='same', activation='relu', name=f'{name}_pool_proj')(path4)

    # Concatenacao dos ramos na dimensao dos canais
    return concatenate([path1, path2, path3, path4], axis=-1, name=name)

def auxiliary_classifier(x, name, num_classes):
    # Reducao espacial
    x = AveragePooling2D((5, 5), strides=(3, 3), name=f'{name}_avgpool')(x)
    x = Conv2D(128, (1, 1), padding='same', activation='relu', name=f'{name}_conv')(x)
    x = Flatten(name=f'{name}_flatten')(x)
    x = Dense(1024, activation='relu', name=f'{name}_fc')(x)
    x = Dropout(0.7, name=f'{name}_dropout')(x)
    x = Dense(num_classes, activation='softmax', name=f'{name}_output')(x)
    
    return x

def build_googlenet(input_shape, num_classes):
    input_layer = Input(shape=input_shape)

    # Estagio 1
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='same', activation='relu', name='conv1_7x7')(input_layer)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool1_3x3')(x)

    # Estagio 2
    x = Conv2D(64, (1, 1), padding='same', activation='relu', name='conv2_1x1')(x)
    x = Conv2D(192, (3, 3), padding='same', activation='relu', name='conv2_3x3')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool2_3x3')(x)

    # Estagio 3 (Inception 3a e 3b)
    x = inception_module(x, 64, 96, 128, 16, 32, 32, name='inception_3a')
    x = inception_module(x, 128, 128, 192, 32, 96, 64, name='inception_3b')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool3_3x3')(x)

    # Estagio 4 (Inception 4a a 4e) + Classificador Auxiliar 1
    x = inception_module(x, 192, 96, 208, 16, 48, 64, name='inception_4a')
    aux1 = auxiliary_classifier(x, name='aux1', num_classes=num_classes)

    x = inception_module(x, 160, 112, 224, 24, 64, 64, name='inception_4b')
    x = inception_module(x, 128, 128, 256, 24, 64, 64, name='inception_4c')
    x = inception_module(x, 112, 144, 288, 32, 64, 64, name='inception_4d')
    aux2 = auxiliary_classifier(x, name='aux2', num_classes=num_classes)

    x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_4e')
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='maxpool4_3x3')(x)

    # Estagio 5 (Inception 5a e 5b)
    x = inception_module(x, 256, 160, 320, 32, 128, 128, name='inception_5a')
    x = inception_module(x, 384, 192, 384, 48, 128, 128, name='inception_5b')

    # Classificador principal
    x = AveragePooling2D((7, 7), strides=(1, 1), name='global_avg_pool')(x)
    x = Flatten(name='final_flatten')(x)
    main_output = Dense(num_classes, activation='softmax', name='main_output')(x)

    # O modelo retorna 3 saidas (1 principal e 2 auxiliares)
    model = Model(inputs=input_layer, outputs=[main_output, aux1, aux2], name='GoogleNet')
    return model


# Construindo modelo
model = build_googlenet(input_shape=(224, 224, 3), num_classes=num_classes)

...
