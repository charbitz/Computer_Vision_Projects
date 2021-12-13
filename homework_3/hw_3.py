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

    # Creating vocabulary with K-Means algorithm :
    print('Creating vocabulary...')
    term_crit = (cv.TERM_CRITERIA_EPS, 30, 0.1)
    loss, assignments, vocabulary = cv.kmeans(train_descs.astype(np.float32), 50, None, term_crit, 1, 0)
    # Creating vocabulary.npy file :
    np.save('vocabulary.npy', vocabulary)
    return vocabulary

def load_vocabulary():
    vocabulary = np.load('vocabulary.npy')
    return vocabulary

def encode_bovw_descriptor(desc, vocabulary):
    bovw_desc = np.zeros((1, vocabulary.shape[0]))
    for d in range(desc.shape[0]):
        distances = np.sum((desc[d, :] - vocabulary) ** 2, axis=1)
        mini = np.argmin(distances)
        bovw_desc[0, mini] += 1
    if np.sum(bovw_desc) > 0:
        bovw_desc = bovw_desc / np.sum(bovw_desc)
    return bovw_desc

def create_index(train_folders, vocabulary):
    print('Creating index...')
    img_paths = []
    bovw_descs = np.zeros((0, vocabulary.shape[0]))
    for train_folder in train_folders:
        for folder in train_folder:
            files = os.listdir(folder)
            for file in files:
                path = os.path.join(folder, file)
                desc = extract_local_features(path)
                if desc is None:
                    continue
                bovw_desc = encode_bovw_descriptor(desc, vocabulary)

                img_paths.append(path)
                bovw_descs = np.concatenate((bovw_descs, bovw_desc), axis=0)

    # Creating index.npy and index_paths.txt files :
    np.save('index.npy', bovw_descs)
    with open('index_paths.txt', mode='w+') as file:
        json.dump(img_paths, file)
    return img_paths, bovw_descs

def load_index():
    bovw_descs = np.load('index.npy')
    with open('index_paths.txt', mode='r') as file:
        img_paths = json.load(file)
    return img_paths, bow_descs

# Creating vocabulary :
# vocabulary = create_vocabulary(train_folders)

# Loading the created vocabulary :
vocabulary = load_vocabulary()

# Creating index :
# img_paths, bow_descs = create_index(train_folders, vocabulary)

# Loading the created index :
img_paths, bow_descs = load_index()