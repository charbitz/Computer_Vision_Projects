import os
import cv2 as cv
import numpy as np
import json
import matplotlib.pyplot as plt

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

def create_vocabulary(train_folders, num_of_words):
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
    loss, assignments, vocabulary = cv.kmeans(train_descs.astype(np.float32), num_of_words, None, term_crit, 1, 0)
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

def svm_one_versus_all_training(img_paths, svm_kernel_type):
    # SVM for motorbike training :
    print('Training SVM for motorbike ...')
    svm_motorbike = cv.ml.SVM_create()
    svm_motorbike.setType(cv.ml.SVM_C_SVC)
    svm_motorbike.setKernel(svm_kernel_type)
    svm_motorbike.setTermCriteria((cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_EPS, 200, 1.e-06))

    labels_motorbike = []
    for i in img_paths:
        labels_motorbike.append('motorbike' in i)
    labels_motorbike = np.array(labels_motorbike, np.int32)

    svm_motorbike.trainAuto(bovw_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels_motorbike)
    svm_motorbike.save('svm_motorbike')
    # ###############################################################################################

    # SVM for school-bus training :
    print('Training SVM for school-bus ...')
    svm_schoolbus = cv.ml.SVM_create()
    svm_schoolbus.setType(cv.ml.SVM_C_SVC)
    svm_schoolbus.setKernel(svm_kernel_type)
    svm_schoolbus.setTermCriteria((cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_EPS, 200, 1.e-06))

    labels_schoolbus = []
    for i in img_paths:
        labels_schoolbus.append('school-bus' in i)
    labels_schoolbus = np.array(labels_schoolbus, np.int32)

    svm_schoolbus.trainAuto(bovw_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels_schoolbus)
    svm_schoolbus.save('svm_schoolbus')
    # ###############################################################################################

    # SVM for touring-bike training :
    print('Training SVM for touring-bike ...')
    svm_bike = cv.ml.SVM_create()
    svm_bike.setType(cv.ml.SVM_C_SVC)
    svm_bike.setKernel(svm_kernel_type)
    svm_bike.setTermCriteria((cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_EPS, 200, 1.e-06))

    labels_bike = []
    for i in img_paths:
        labels_bike.append('touring-bike' in i)
    labels_bike = np.array(labels_bike, np.int32)

    svm_bike.trainAuto(bovw_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels_bike)
    svm_bike.save('svm_bike')
    # ###############################################################################################

    # SVM for airplane training :
    print('Training SVM for airplane ...')
    svm_airplane = cv.ml.SVM_create()
    svm_airplane.setType(cv.ml.SVM_C_SVC)
    svm_airplane.setKernel(svm_kernel_type)
    svm_airplane.setTermCriteria((cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_EPS, 200, 1.e-06))

    labels_airplane = []
    for i in img_paths:
        labels_airplane.append('airplane' in i)
    labels_airplane = np.array(labels_airplane, np.int32)

    svm_airplane.trainAuto(bovw_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels_airplane)
    svm_airplane.save('svm_airplane')
    # ###############################################################################################

    # SVM for car training :
    print('Training SVM for car ...')
    svm_car = cv.ml.SVM_create()
    svm_car.setType(cv.ml.SVM_C_SVC)
    svm_car.setKernel(svm_kernel_type)
    svm_car.setTermCriteria((cv.TERM_CRITERIA_COUNT + cv.TERM_CRITERIA_EPS, 200, 1.e-06))

    labels_car = []
    for i in img_paths:
        labels_car.append('car' in i)
    labels_car = np.array(labels_car, np.int32)

    svm_car.trainAuto(bovw_descs.astype(np.float32), cv.ml.ROW_SAMPLE, labels_car)
    svm_car.save('svm_car')

    return svm_motorbike, svm_schoolbus, svm_bike, svm_airplane, svm_car


def svm_one_versus_all_testing(img_paths_test, svm_motorbike, svm_schoolbus, svm_bike, svm_airplane, svm_car ):
    motorbikes = 0
    schoolbuses = 0
    bikes = 0
    airplanes = 0
    cars = 0

    svm_class_pred_list = []
    svm_class_exp_list = []

    for image_path in img_paths_test:

        print("image path :", image_path)

        svm_class_exp = ""
        svm_class_pred = ""

        # Extracting the expected class of the test image path :
        svm_class_exp = extract_image_path_class_exp(image_path)
        svm_class_exp_list.append(svm_class_exp)

        test_desc = extract_local_features(image_path)
        test_bovw_desc = encode_bovw_descriptor(test_desc, vocabulary)

        print("SVM classifier testing for motorbikes:")
        response_motorbikes = svm_motorbike.predict(test_bovw_desc.astype(np.float32),
                                                    flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        if response_motorbikes[1] < 0:
            print('It is a motorbike')
        else:
            print('It is sth else')

        print("SVM classifier testing for school-bus:")
        response_schoolbus = svm_schoolbus.predict(test_bovw_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        if response_schoolbus[1] < 0:
            print('It is a school-bus')
        else:
            print('It is sth else')

        print("SVM classifier testing for touring-bike:")
        response_bike = svm_bike.predict(test_bovw_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        if response_bike[1] < 0:
            print('It is a touring-bike')
        else:
            print('It is sth else')

        print("SVM classifier testing for aiplane:")
        response_airplane = svm_airplane.predict(test_bovw_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        if response_airplane[1] < 0:
            print('It is a aiplane')
        else:
            print('It is sth else')

        print("SVM classifier testing for car:")
        response_car = svm_car.predict(test_bovw_desc.astype(np.float32), flags=cv.ml.STAT_MODEL_RAW_OUTPUT)
        if response_car[1] < 0:
            print('It is a car')
        else:
            print('It is sth else')

        min_dist = min(response_motorbikes[1], response_schoolbus[1], response_bike[1], response_airplane[1],
                       response_car[1])

        if min_dist == response_motorbikes[1]:
            svm_class_pred = "motorbike"
            motorbikes += 1
        elif min_dist == response_schoolbus[1]:
            svm_class_pred = "school-bus"
            schoolbuses += 1
        elif min_dist == response_bike[1]:
            svm_class_pred = "bike"
            bikes += 1
        elif min_dist == response_airplane[1]:
            svm_class_pred = "airplane"
            airplanes += 1
        elif min_dist == response_car[1]:
            svm_class_pred = "car"
            cars += 1

        svm_class_pred_list.append(svm_class_pred)
        print("The final classification is ", svm_class_pred)

        print(" ")

        svm_class_pred = ""

    return svm_class_pred_list, svm_class_exp_list


def extract_image_path_class_exp(test_image_path):
    class_exp = ""
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
    return class_exp


def compute_accuracy(class_pred_list, class_exp_list):
    global_sum = 0
    motorbike_sum = 0
    schoolbus_sum = 0
    bike_sum = 0
    airplane_sum = 0
    car_sum = 0

    paths_motorbike = class_exp_list[0:10]
    paths_schoolbus = class_exp_list[10:19]
    paths_bike = class_exp_list[19:30]
    paths_airplane = class_exp_list[30:41]
    paths_car = class_exp_list[41:52]

    for i in range(0, len(class_exp_list)):
        if class_pred_list[i] == class_exp_list[i]:
            global_sum += 1

    global_acc = global_sum / len(class_exp_list)

    for i in range(0, len(paths_motorbike)):
        if class_pred_list[i] == class_exp_list[i]:
            motorbike_sum += 1

    motorbike_acc = motorbike_sum / len(paths_motorbike)

    for i in range(0, len(paths_schoolbus)):
        if class_pred_list[i+10] == class_exp_list[i+10]:
            schoolbus_sum += 1

    schoolbus_acc = schoolbus_sum / len(paths_schoolbus)

    for i in range(0, len(paths_bike)):
        if class_pred_list[i+19] == class_exp_list[i+19]:
            bike_sum += 1

    bike_acc = bike_sum / len(paths_bike)

    for i in range(0, len(paths_airplane)):
        if class_pred_list[i+30] == class_exp_list[i+30]:
            airplane_sum += 1

    airplane_acc = airplane_sum / len(paths_airplane)

    for i in range(0, len(paths_car)):
        if class_pred_list[i+41] == class_exp_list[i+41]:
            car_sum += 1

    car_acc = car_sum / len(paths_car)

    return motorbike_acc, schoolbus_acc, bike_acc, airplane_acc, car_acc, global_acc

num_of_words_list = [50, 100, 150]

for num_of_words in num_of_words_list:

    # Creating vocabulary :
    vocabulary = create_vocabulary(train_folders, num_of_words)

    # Loading the created vocabulary :
    # vocabulary = load_vocabulary()

    # Creating index :
    img_paths, img_paths_test, bovw_descs = create_index(train_folders, test_folders, vocabulary)

    # Loading the created index :
    # img_paths, img_paths_test, bovw_descs = load_index()

    k_vals = [5, 20, 40, 60, 80, 100]

    knn_motorbike_list = []
    knn_schoolbus_list = []
    knn_bike_list = []
    knn_airplane_list = []
    knn_car_list = []

    knn_global_acc_list = []

    for k_value in k_vals:
        class_exp = ""
        class_pred = ""

        class_pred_list = []
        class_exp_list = []

        # Testing of kNN classifier on image_db_test :
        for test_image_path in img_paths_test:

            # Measuring the test image's expected(known) class :
            class_exp = extract_image_path_class_exp(test_image_path)
            class_exp_list.append(class_exp)

            # Extracting the test image's local features :
            q_desc = extract_local_features(test_image_path)

            # Extracting the test image's features based on the BOVW model :
            q_bovw_desc = encode_bovw_descriptor(q_desc, vocabulary)

            # # Parameter ok knn: number of neighbours :
            # k = 300
            neighbours, class_pred = knn_classifier(q_bovw_desc, bovw_descs, img_paths, k_value)

            class_pred_list.append(class_pred)

            print("This belongs to the class:", class_pred, "   with neighbours:", neighbours)

        # Computing the accuracy of the knn classifier :
        motorbike_acc, schoolbus_acc, bike_acc, airplane_acc, car_acc, global_acc = compute_accuracy(class_pred_list, class_exp_list)

        knn_results = [motorbike_acc, schoolbus_acc, bike_acc, airplane_acc, car_acc, global_acc]

        knn_motorbike_list.append(motorbike_acc)
        knn_schoolbus_list.append(schoolbus_acc)
        knn_bike_list.append(bike_acc)
        knn_airplane_list.append(airplane_acc)
        knn_car_list.append(car_acc)

        knn_global_acc_list.append(global_acc)

        print("knn results with k =", k_value, " are:", knn_results)

    # Plotting a diagram to indicate the dependency of the accuracy from the parameter k :
    plt.plot(k_vals,knn_motorbike_list, 'r-o',label='Motorbike accuracy.')
    plt.plot(k_vals,knn_schoolbus_list, 'g-o',label='School-bus accuracy.')
    plt.plot(k_vals,knn_bike_list,      'c-o',label='Bike accuracy.')
    plt.plot(k_vals,knn_airplane_list,  'm-o',label='Airplane accuracy.')
    plt.plot(k_vals,knn_car_list,       'b-o',label='Car accuracy.')

    plt.plot(k_vals,knn_global_acc_list,'k-o',label='Global accuracy.')
    plt.title(label=('KNN Classifier with Number of words:', num_of_words))
    plt.xlabel('Parameter k')
    plt.ylabel('Types of Accuracy')
    plt.legend()
    plt.show()

    # Creating a list with the types of the svm kernel and a list with the tags of the corresponding types for the accuracy diagramm :
    svm_kernel_type_list = [cv.ml.SVM_RBF, cv.ml.SVM_LINEAR, cv.ml.SVM_CHI2]
    svm_kernel_type_list_tags = ["RBF", "LINEAR", "CHI2"]

    svm_motorbike_list = []
    svm_schoolbus_list = []
    svm_bike_list = []
    svm_airplane_list = []
    svm_car_list = []

    svm_global_acc_list = []

    for type in svm_kernel_type_list:

        # SVM training :
        svm_motorbike, svm_schoolbus, svm_bike, svm_airplane, svm_car = svm_one_versus_all_training(img_paths, type)

        # SVM testing :
        svm_class_pred_list, svm_class_exp_list = svm_one_versus_all_testing(img_paths_test, svm_motorbike, svm_schoolbus, svm_bike, svm_airplane, svm_car)

        # Computing the accuracy of the SVM classifier :
        svm_motorbike_acc, svm_schoolbus_acc, svm_bike_acc, svm_airplane_acc, svm_car_acc, svm_global_acc = compute_accuracy(svm_class_pred_list, svm_class_exp_list)
        results_svm = [svm_motorbike_acc, svm_schoolbus_acc, svm_bike_acc, svm_airplane_acc, svm_car_acc, svm_global_acc]
        print("Results of the classifier svm are:", results_svm)

        svm_motorbike_list.append(svm_motorbike_acc)
        svm_schoolbus_list.append(svm_schoolbus_acc)
        svm_bike_list.append(svm_bike_acc)
        svm_airplane_list.append(svm_airplane_acc)
        svm_car_list.append(svm_car_acc)

        svm_global_acc_list.append(svm_global_acc)

    # Plotting a diagram to indicate the dependency of the accuracy from the parameter SVM kernel type :
    plt.plot(svm_kernel_type_list_tags, svm_motorbike_list, 'r-o', label='Motorbike accuracy.')
    plt.plot(svm_kernel_type_list_tags, svm_schoolbus_list, 'g-o', label='School-bus accuracy.')
    plt.plot(svm_kernel_type_list_tags, svm_bike_list, 'c-o', label='Bike accuracy.')
    plt.plot(svm_kernel_type_list_tags, svm_airplane_list, 'm-o', label='Airplane accuracy.')
    plt.plot(svm_kernel_type_list_tags, svm_car_list, 'b-o', label='Car accuracy.')

    plt.plot(svm_kernel_type_list_tags, svm_global_acc_list, 'k-o', label='Global accuracy.')
    plt.title(label=('SVM Classifier with Number of words:', num_of_words, "and type of kernel", str(type)))
    plt.ylabel('Types of Accuracy')
    plt.legend()
    plt.show()
