import os
import cv2
import csv
from keras import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from sklearn import svm
from sklearn.linear_model import Perceptron
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder): #get every file from the directory
        img = cv2.imread(os.path.join(folder,filename)) #read the image
        if img is not None:
            images.append(img)
    return images #return a list of images

list = load_images_from_folder("data/train")
train_images = np.array(list)

f1 = open('C:/Users/Andreea/Dropbox/My PC (DESKTOP-7M48CSA)/Desktop/FACULTATE NOU/SEM2/IA/kaggle/data/train.txt', 'r')
list = [int(l.split(",")[1][0:1]) for l in f1.readlines()]
train_labels = np.array(list)

list = load_images_from_folder("data/validation")
validation_images = np.array(list)

f1 = open('C:/Users/Andreea/Dropbox/My PC (DESKTOP-7M48CSA)/Desktop/FACULTATE NOU/SEM2/IA/kaggle/data/validation.txt', 'r')
list = [int(l.split(",")[1][0:1]) for l in f1.readlines()]
validation_labels = np.array(list)

list = load_images_from_folder("data/test")
test_images = np.array(list)

f1 = open('C:/Users/Andreea/Dropbox/My PC (DESKTOP-7M48CSA)/Desktop/FACULTATE NOU/SEM2/IA/kaggle/data/test.txt', 'r')
id_test = f1.read().splitlines() #read image names

cnn = models.Sequential([
    layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.25),

    layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.35),

    layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    layers.Dropout(0.5),

    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.25),
    layers.Dense(9, activation='softmax')
])

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.fit(train_images/255, train_labels, epochs=100, batch_size=64, validation_data=(validation_images/255, validation_labels))
print(cnn.evaluate(validation_images/255, validation_labels)) #get the accuracy of the classifier for the validation data
y_pred = cnn.predict(test_images/255) #predict the labels for test images

with open('nn2_out.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["id", "label"])
    for i in range(len(y_pred)):
        writer.writerow([str(id_test[i]), y_pred[i].argmax()]) #write the name of the image and the predicted label