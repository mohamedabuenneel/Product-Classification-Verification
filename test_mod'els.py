import os
import cv2
import pandas as pd
import tensorflow as tf
import numpy as np
import csv

output = pd.DataFrame(columns=['image_id', 'label'])

path = '../Data/Product Classification/Validation'


def load_test_data():
    test_data = []
    for i in os.listdir(path):
        # if i != '6':
        for j in os.listdir(path + '/' + i):
            img = cv2.imread(path + '/' + i + '/' + j, cv2.IMREAD_COLOR)
            img = cv2.resize(img, (224, 224))
            test_data.append((img, i))
            # print(test_data[-1][1])
    return test_data


test_data = load_test_data()
x_test, y_test = zip(*test_data)
x_test = np.array(x_test)
y_test = np.array(y_test)
# for i in range(30):
#     print(classes[i])
#     cv2.imshow('', images[i])
#     cv2.waitKey(0)
model = tf.keras.models.load_model('../Data/Saved Models/efficient_net_model2.h5')
# print(model.predict(x_test[1].reshape(1, 224, 224, 3)))
# print('**********************************************************')
# print(y_test[1])

predictions = []
for i in range(len(x_test)):
    predictions.append(model.predict(x_test[i].reshape(1, 224, 224, 3)))

# Convert each prediction to class index
predicted_class_indices_list = []

for model_prediction in predictions:
    predicted_class_index = np.argmax(model_prediction) + 1
    predicted_class_indices_list.append(predicted_class_index)

print(f"Predicted Class Indices: {predicted_class_indices_list}")
print(f"Predicted Class Indices: {y_test}")

counter = 0
for i in range(len(predicted_class_indices_list)):
    if predicted_class_indices_list[i] == int(y_test[i]):
        counter += 1

print('**************************************************************')
acc = counter / len(y_test)
print('Accuracy:', acc * 100)
print(len(y_test) - counter)
print('**************************************************************')

# print('----------------------------------------------------------------------')
#
# img = cv2.imread('../Data/Product Classification/Validation/5/web7.png', cv2.IMREAD_COLOR)
# img = cv2.resize(img, (224, 224))
# pr = model.predict(img.reshape(1, 224, 224, 3))
# print(np.argmax(pr) + 1)
# print(pr)
# print('//////////////////////////')
# img = cv2.imread('D:/Downloads/2.jpg', cv2.IMREAD_COLOR)
# img = cv2.resize(img, (224, 224))
# pr = model.predict(img.reshape(1, 224, 224, 3))
# print(np.argmax(pr) + 1)
# print(pr)

img = cv2.imread('D:/Downloads/1.jpeg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))
pr = model.predict(img.reshape(1, 224, 224, 3))
print(np.argmax(pr) + 1)
print('1', pr)

img = cv2.imread('D:/Downloads/2.jpeg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))
pr = model.predict(img.reshape(1, 224, 224, 3))
print(np.argmax(pr) + 1)
print('2', pr)
img = cv2.imread('D:/Downloads/3.jpeg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))
pr = model.predict(img.reshape(1, 224, 224, 3))
print(np.argmax(pr) + 1)
print('3', pr)
img = cv2.imread('D:/Downloads/4.jpeg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))
pr = model.predict(img.reshape(1, 224, 224, 3))
print(np.argmax(pr) + 1)
print('4', pr)
img = cv2.imread('D:/Downloads/5.jpeg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))
pr = model.predict(img.reshape(1, 224, 224, 3))
print(np.argmax(pr) + 1)
print('5', pr)
img = cv2.imread('D:/Downloads/6.jpeg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))
pr = model.predict(img.reshape(1, 224, 224, 3))
print(np.argmax(pr) + 1)
print('6', pr)
img = cv2.imread('D:/Downloads/7.jpeg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))
pr = model.predict(img.reshape(1, 224, 224, 3))
print(np.argmax(pr) + 1)
print('7', pr)
img = cv2.imread('D:/Downloads/8.jpeg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))
pr = model.predict(img.reshape(1, 224, 224, 3))
print(np.argmax(pr) + 1)
print('8', pr)
img = cv2.imread('D:/Downloads/9.jpeg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))
pr = model.predict(img.reshape(1, 224, 224, 3))
print(np.argmax(pr) + 1)
print('9', pr)
img = cv2.imread('D:/Downloads/10.jpeg', cv2.IMREAD_COLOR)
img = cv2.resize(img, (224, 224))
pr = model.predict(img.reshape(1, 224, 224, 3))
print(np.argmax(pr) + 1)
print('10', pr)
