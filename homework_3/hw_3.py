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

test_folders1 = ["caltech/imagedb_test/145.motorbikes-101"]
test_folders2 = ["caltech/imagedb_test/178.school-bus"]
test_folders3 = ["caltech/imagedb_test/224.touring-bike"]
test_folders4 = ["caltech/imagedb_test/251.airplanes-101"]
test_folders5 = ["caltech/imagedb_test/252.car-side-101"]

test_folders = [test_folders1, test_folders2, test_folders3, test_folders4, test_folders5]

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

def create_index(train_folders, test_folders, vocabulary):
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

    img_paths_test = []
    for test_folder in test_folders:
        for t_folder in test_folder:
            t_files = os.listdir(t_folder)
            for t_file in t_files:
                t_path = os.path.join(t_folder, t_file)
                img_paths_test.append(t_path)

    # Creating index_paths_test.txt file :
    with open('index_paths_test.txt', mode='w+') as t_file:
        json.dump(img_paths_test, t_file)

    return img_paths, img_paths_test, bovw_descs

def load_index():
    bovw_descs = np.load('index.npy')

    # Loading training image paths (img_paths) :
    with open('index_paths.txt', mode='r') as file:
        img_paths = json.load(file)

    # Loading testing image paths (img_paths_test) :
    with open('index_paths_test.txt', mode='r') as t_file:
        img_paths_test = json.load(t_file)

    return img_paths, img_paths_test, bovw_descs

def knn_classifier(q_bovw_desc, bovw_descs, img_paths, k):

    # # Deleting the query's word from the bag of words:
    # for row in range(bovw_descs.shape[0]):
    #     if (bovw_descs[row, :] == q_bovw_desc).all():
    #         bovw_descs_del = np.delete(bovw_descs, row, 0)
    #         print("delete the row:", row)                       # may stay for some prints

    eucl_dist = []
    # Computing euclidean distance from the query word to all the other words:
    for row in range(bovw_descs.shape[0]):
        eucl_dist_2 = np.linalg.norm(bovw_descs[row, :] - q_bovw_desc)
        eucl_dist.append(eucl_dist_2)

    neighbours = [0, 0, 0, 0, 0]

    # Computing the k nearest neighbours :
    for times in range(k):
        # Finding the number of the image with minimum distance:
        mini = np.argmin(eucl_dist)

        # Finding the class with the minimum distance:
        if "motorbike" in img_paths[mini]:
            neighbours[0] += 1
        elif "school-bus" in img_paths[mini]:
            neighbours[1] += 1
        elif "bike" in img_paths[mini]:
            neighbours[2] += 1
        elif "airplane" in img_paths[mini]:
            neighbours[3] += 1
        elif "car" in img_paths[mini]:
            neighbours[4] += 1

        # Setting this minimum distance to infinite, in order not to be appeared again:
        eucl_dist[mini] = np.inf

    # Finding the class of the query image :
    max_class = np.argmax(neighbours)

    class_pred = ""
    if max_class == 0:
        class_pred = "motorbike"
    elif max_class == 1:
        class_pred = "school-bus"
    elif max_class == 2:
        class_pred = "bike"
    elif max_class == 3:
        class_pred = "airplane"
    elif max_class == 4:
        class_pred = "car"

    return neighbours, class_pred

# Creating vocabulary :
# vocabulary = create_vocabulary(train_folders)

# Loading the created vocabulary :
vocabulary = load_vocabulary()

# Creating index :
# img_paths, img_paths_test, bow_descs = create_index(train_folders, test_folders, vocabulary)

# Loading the created index :
img_paths, img_paths_test, bovw_descs = load_index()


correct_predictions = 0
class_exp = ""

# Testing of kNN classifier :
for test_image_path in img_paths_test:

    # Measuring the expected(known) class of the image:
    if "motorbike" in test_image_path:
        class_exp = "motorbike"
    elif "school-bus" in test_image_path:
        class_exp = "school-bus"
    elif "bike" in test_image_path:
        class_exp = "bike"
    elif "airplane" in test_image_path:
        class_exp = "airplane"
    elif "car" in test_image_path:
        class_exp = "car"

    # test_image = cv.imread(test_image_path)
    #
    # cv.namedWindow('test_image', cv.WINDOW_NORMAL)
    # cv.imshow('test_image', test_image)
    # cv.waitKey(0)

    q_desc = extract_local_features(test_image_path)

    q_bovw_desc = encode_bovw_descriptor(q_desc, vocabulary)

    k = 5
    neighbours, class_pred = knn_classifier(q_bovw_desc, bovw_descs, img_paths, k)

    print("This belongs to the class:", class_pred, "   with neighbours:", neighbours)

    if (class_pred == class_exp):
        correct_predictions += 1

print("correct predictions:", correct_predictions)

knn_global_acc = correct_predictions/len(img_paths_test)
print("The knn accuracy ath the global test set is:", knn_global_acc)
