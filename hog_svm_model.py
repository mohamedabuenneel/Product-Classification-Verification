import numpy as np
import cv2
import matplotlib as mat
import cv2
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
##########################
import os


def getFiles(path, type):
    imlist = {}
    count = 0
    labels_list = []
    for each in os.listdir(path):
        # print(" #### Reading image category ", each, " ##### ")
        imlist[each] = []
        for imagefile in os.listdir(path + '/' + each + '/' + type):
            # print("Reading file ", imagefile)
            im = cv2.imread(path + '/' + each + '/' + type + '/' + imagefile)
            resized_image = cv2.resize(im, (180, 180))
            imlist[each].append(resized_image)
            labels_list.append(each)
            count += 1

    return [imlist, count, labels_list]


def extract_hog_features(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate HOG features
    hog = cv2.HOGDescriptor()
    features = hog.compute(gray)

    return features.flatten()


def create_bow_representation(features, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(features)
    return kmeans


path = "C:/Users/AboEnneel/Desktop/Data/Product Classification"
train = 'Train'
test = 'Validation'

imgsOfTrain, numbersOfTrain, labelOfTrain = getFiles(path, train)
imgsOfTest, numbersOfTest, labelOfTest = getFiles(path, test)

hog_features1 = [[]]
for label in imgsOfTrain:
    for img in imgsOfTrain[label]:
        hog_features1.append(extract_hog_features(img))
X_train = hog_features1
X_train.pop(0)
Y_train = np.array(labelOfTrain)

#####
hog_features2 = [[]]
for label in imgsOfTest:
    for img in imgsOfTest[label]:
        hog_features2.append(extract_hog_features(img))
X_test = hog_features2
X_test.pop(0)
Y_test = np.array(labelOfTest)
# print(len(X_train))
# print(len(Y_train))


# k = 100
# kmeans_model = create_bow_representation(X_train, k)
# bow_representation_train = kmeans_model.predict(X_train)
# bow_representation_test = kmeans_model.predict(X_test)
#
#
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)

# Train the SVM classifier
svm_classifier = SVC(kernel='linear', C=1.0)
svm_classifier.fit(X_train_scaled, Y_train)

# Make predictions on the testing data
pred_of_training = svm_classifier.predict(X_train_scaled)
predictions = svm_classifier.predict(X_test_scaled)
# print(len(predictions))
# print(len(Y_test))

# Evaluate the model
acc = accuracy_score(Y_train, pred_of_training)
accuracy = accuracy_score(Y_test, predictions)

print(f"Accuracy: {acc * 100:.2f}%")
print(f"Accuracy: {accuracy * 100:.2f}%")
