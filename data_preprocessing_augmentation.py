import os
import cv2
import pandas as pd
import numpy as np
from random import shuffle
import random


def resize_img(img):
    img = cv2.resize(img, (224, 224))
    return img


def create_label(image_class):
    hot_encoded_list = []
    for i in range(20):
        if i + 1 == int(image_class):
            hot_encoded_list.append(1)
        else:
            hot_encoded_list.append(0)
    return hot_encoded_list


def random_zoom_and_crop(img):
    zoom_factor = random.uniform(0.8, 1.2)

    new_width = int(img.shape[1] * zoom_factor)
    new_height = int(img.shape[0] * zoom_factor)

    start_x = random.randint(0, max(0, new_width - img.shape[1]))
    start_y = random.randint(0, max(0, new_height - img.shape[0]))

    zoomed_img = img[start_y:(start_y + img.shape[0]), start_x:(start_x + img.shape[1])]
    zoomed_img = cv2.resize(zoomed_img, (img.shape[1], img.shape[0]))

    return zoomed_img


def random_brightness_contrast(img):
    alpha = random.uniform(0.8, 1.2)
    beta = random.uniform(-20, 20)

    adjusted_img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    return adjusted_img


data_classification = []
data_recognition = []
path = '../Data'


# augmentation_probability_flip = 0.2
# augmentation_probability_zoom = 0.3
def preprocessing_and_augmentation(path, classification=True, recognition=False):
    if classification:
        path += '/Product Classification/Train'

        for class_folder in os.listdir(path):
            for image in os.listdir(path + '/' + class_folder):
                img = cv2.imread(path + '/' + class_folder + '/' + image, cv2.IMREAD_COLOR)
                img = resize_img(img)
                data_classification.append((img, create_label(class_folder)))

                data_classification.append((cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE), create_label(class_folder)))
                data_classification.append((cv2.flip(img, -1), create_label(class_folder)))
                data_classification.append((cv2.flip(img, 0), create_label(class_folder)))
                data_classification.append((cv2.flip(img, 1), create_label(class_folder)))

                zoomed_img = random_zoom_and_crop(img)
                data_classification.append((zoomed_img, create_label(class_folder)))

                bright_contrast_img = random_brightness_contrast(img)
                data_classification.append((bright_contrast_img, create_label(class_folder)))


preprocessing_and_augmentation(path)

# Convert the list of tuples into separate arrays
images, labels = zip(*data_classification)
images = np.array(images)
labels = np.array(labels)

# Shuffle the data
indices = np.arange(len(images))
np.random.shuffle(indices)
images = images[indices]
labels = labels[indices]

# Save the arrays
np.save('../Data/preprocessed data/train_images.npy', images)
np.save('../Data/preprocessed data/train_labels.npy', labels)

print('done')

x_train = np.load('../Data/preprocessed data/train_images.npy')
y_train = np.load('../Data/preprocessed data/train_labels.npy')

print(len(x_train))
print(len(y_train))
