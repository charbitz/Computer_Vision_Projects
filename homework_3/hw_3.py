import os
import cv2 as cv
import numpy as np
import json

train_folders1 = ["caltech/imagedb/145.motorbikes-101"]
train_folders2 = ["caltech/imagedb/178.school-bus"]
train_folders3 = ["caltech/imagedb/224.touring-bike"]
train_folders4 = ["caltech/imagedb/251.airplanes-101"]
train_folders5 = ["caltech/imagedb/252.car-side-101"]

train_folders = [train_folders1, train_folders2, train_folders3, train_folders4, train_folders5]

sift = cv.xfeatures2d_SIFT.create()

def extract_local_features(path):
    img = cv.imread(path)
    kp = sift.detect(img)
    desc = sift.compute(img, kp)
    desc = desc[1]
    return desc

def create_vocabulary(train_folders):
    print('Extracting features...')
    train_descs = np.zeros((0, 128))
    for train_folder in train_folders:
        for folder in train_folder:
            files = os.listdir(folder)
            for file in files:
                path = os.path.join(folder, file)
                desc = extract_local_features(path)
                if desc is None:
                    continue
                train_descs = np.concatenate((train_descs, desc), axis=0)

    # Creating vocabulary
    print('Creating vocabulary...')
    term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
    loss, assignments, vocabulary = cv.kmeans(train_descs.astype(np.float32), 50, None, term_crit, 1, 0)
    np.save('vocabulary.npy', vocabulary)
    return vocabulary

def load_vocabulary():
    vocabulary = np.load('vocabulary.npy')
    return vocabulary

# Creating vocabulary :
# vocabulary = create_vocabulary(train_folders)

# Loading the created vocabulary :
vocabulary = load_vocabulary()
