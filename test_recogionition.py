import matplotlib.pyplot as plt
import numpy as np
import os
import random
import glob
import tensorflow as tf
import cv2
import time
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.applications import Xception
from tensorflow.keras import layers, metrics
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import plot_model
from keras.models import load_model
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

path = '../Data/Product Recoginition/'


def test_validate(path):
    encoder = load_model('../Data/Saved Models/SiameseInception(1).h5')
    maxi = 1000
    label = 0
    true = 0
    false = 0
    for h in range(1, len(os.listdir(path + 'Validation Data/')) + 1):
        anchor = cv2.imread(f'{path}Validation Data/{h + 40}/web{5}.png')
        maxi = 1000
        label = 0
        for i in range(1, len(os.listdir(path + 'Validation Data/')) + 1):
            distance = 0
            for j in range(1, 3):
                anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
                anchor = cv2.resize(anchor, (224, 224))

                image2 = cv2.imread(f'{path}Validation Data/{i + 40}/web{j}.png')
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                image2 = cv2.resize(image2, (224, 224))

                im1 = preprocess_input(anchor)
                im2 = preprocess_input(image2)

                embedding1 = encoder.predict(np.array([im1]))
                embedding2 = encoder.predict(np.array([im2]))
                distance += np.sum(np.square(embedding1 - embedding2), axis=-1)
                prediction = np.where(distance <= 0.7, 1, 0)
            if (maxi > distance):
                maxi = distance
                label = i + 40
        if (label == h + 40):
            true += 1
        else:
            false += 1
        print("True", true)
        print("False", false)
        print(label)
        print(maxi)


def test_train(path):
    encoder = load_model('/kaggle/working/SiameseInception.h5')
    maxi = 1000
    label = 0
    true = 0
    false = 0
    for h in range(1, len(os.listdir(path + 'Training Data/'))):
        anchor = cv2.imread(f'{path}Training Data/{h}/web{4}.png')
        maxi = 1000
        label = 0
        for i in range(1, len(os.listdir(path + 'Training Data/'))):
            distance = 0
            for j in range(1, 4):
                anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
                anchor = cv2.resize(anchor, (224, 224))

                image2 = cv2.imread(f'{path}Training Data/{i}/web{j}.png')
                image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
                image2 = cv2.resize(image2, (224, 224))

                im1 = preprocess_input(anchor)
                im2 = preprocess_input(image2)

                embedding1 = encoder.predict(np.array([im1]))
                embedding2 = encoder.predict(np.array([im2]))
                distance += np.sum(np.square(embedding1 - embedding2), axis=-1)
                prediction = np.where(distance <= 0.7, 1, 0)
            if (maxi > distance):
                maxi = distance
                label = i
        if (label == h):
            true += 1
        else:
            false += 1
        print("True", true)
        print("False", false)
        print(label)
        print(maxi)


def test_script(path):
    encoder = load_model('../Data/Saved Models/SiameseInception(1).h5')
    maxi = 1000
    label = 0
    true = 0
    false = 0
    anchor = cv2.imread('../Data/Product Recoginition/Test Data/anchor.jpeg')
    maxi = 1000
    label = 0
    for i in range(1, len(os.listdir(path + 'Test Data/'))):
        distance = 0
        anchor = cv2.cvtColor(anchor, cv2.COLOR_BGR2RGB)
        anchor = cv2.resize(anchor, (224, 224))

        image2 = cv2.imread(f'../Data/Product Recoginition/Test Data/{i}.jpeg')
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = cv2.resize(image2, (224, 224))

        im1 = preprocess_input(anchor)
        im2 = preprocess_input(image2)

        embedding1 = encoder.predict(np.array([im1]))
        embedding2 = encoder.predict(np.array([im2]))
        distance += np.sum(np.square(embedding1 - embedding2), axis=-1)
        prediction = np.where(distance <= 0.7, 1, 0)
        if (maxi > distance):
            maxi = distance
            label = i
    print("True", true)
    print("False", false)
    print(label)
    print(maxi)


test_script(path)
