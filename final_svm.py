import os
import cv2
import numpy as np
from sklearn import svm, metrics
import matplotlib.pyplot as plt


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder): #get every file from the directory
        img = cv2.imread(os.path.join(folder, filename)) #read the image
        img = img.flatten() #vectorize the image
        if img is not None:
            images.append(img)
    return images #return a list of vectorized images


list = load_images_from_folder("data/train")
train_images = np.array(list) #read train images

f1 = open('C:/Users/Andreea/Dropbox/My PC (DESKTOP-7M48CSA)/Desktop/proiect_nou/data/train.txt', 'r')
list = [int(l.split(",")[1][0:1]) for l in f1.readlines()] #read train labels from txt file
train_labels = np.array(list)

list = load_images_from_folder("data/validation")
validation_images = np.array(list) #read validation images

f1 = open('C:/Users/Andreea/Dropbox/My PC (DESKTOP-7M48CSA)/Desktop/proiect_nou/data/validation.txt','r')
list = [int(l.split(",")[1][0:1]) for l in f1.readlines()] #read validation labels from txt file
validation_labels = np.array(list)

list = load_images_from_folder("data/test")
test_images = np.array(list) #read test images

clf = svm.SVC(C=50, gamma=0.015, verbose=True)
clf.fit(train_images, train_labels)
print(clf.score(validation_images, validation_labels))
#y_pred = clf.predict(test_images)
v_pred = clf.predict(validation_images)

confusion_matrix = metrics.confusion_matrix(validation_labels, v_pred, labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
print(confusion_matrix)
#true positive		false positive
#false negative		true negative