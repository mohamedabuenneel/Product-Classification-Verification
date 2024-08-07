from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.layers import Input
import numpy as np
import cv2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.applications import EfficientNetB0
from keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
import warnings
from keras.regularizers import l2
from keras.layers import GlobalAveragePooling2D
from keras.callbacks import EarlyStopping
import tensorflow as tf
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Model
from keras.layers import Input
import numpy as np
import cv2
from keras.models import Sequential
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.applications import EfficientNetB0
from keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint
import warnings
from keras.regularizers import l2
from keras.layers import GlobalAveragePooling2D
import keras, os
import tensorflow as tf
from keras.models import Model
import matplotlib.pyplot as plt
from keras.models import Sequential
from tensorflow.keras.layers import concatenate
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import datasets, layers, models, losses
from keras.layers import Input, Dense, Conv2D, MaxPool2D, Flatten, MaxPooling2D, GlobalAveragePooling2D, \
    AveragePooling2D, Dropout, Activation, BatchNormalization
from keras.models import load_model
from tensorflow.keras.applications.vgg16 import VGG16
import numpy as np
import cv2
import os
from tensorflow.keras.applications import ResNet50

x_train = np.load('../Data/preprocessed data/train_images.npy')
y_train = np.load('../Data/preprocessed data/train_labels.npy')

base_model = ResNet50(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

for layer in base_model.layers:
    layer.trainable = False

x = layers.Flatten()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(5, activation='Softmax')(x)
model = tf.keras.models.Model(base_model.input, x)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

model.fit(x_train, y_train, epochs=3)

model.save('../Data/Saved Models/resnet.h5')