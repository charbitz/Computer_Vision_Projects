import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics as st
import csv

image_path = 'dataset/5_noise.png'

noise = 1 if "noise" in image_path else 0

print("The path of the input image is:", image_path)
print("So the noise is:", noise)

image = cv2.imread(image_path)

print("The shape of the input image is:", image.shape)

cv2.namedWindow('image')
image_r = cv2.resize(image, (650, 800))
cv2.imshow('image', image_r)
cv2.waitKey(0)

image_gr = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

cv2.namedWindow('image_gr')
image_gr_r = cv2.resize(image_gr, (650, 800))
cv2.imshow('image_gr', image_gr_r)
cv2.waitKey(0)

# This is the image, in which for "original" data images (noise = 0), we are going to implement thresholding :
image_to_bin = image_gr

if noise == 1 :
    # Median filtering with opencv :
    image_filt_cv = cv2.medianBlur(image_gr, ksize=3)

    cv2.namedWindow('image_filt_cv')
    image_filt_cv_r = cv2.resize(image_filt_cv, (650, 800))
    cv2.imshow('image_filt_cv', image_filt_cv_r)
    cv2.waitKey(0)

    image_to_bin = image_filt_cv
    # Median filtering with opencv :

    # Median filtering without using an OpenCV function :
    # Firstly we should use zero padding :
    # image_gr_pad = np.pad(image_gr, ((1, 1), (1, 1)), 'constant')
    #
    # print("The shape of the input image in grayscale with zero padding is:", image_gr_pad.shape)
    #
    # image_filt_alg = image_gr_pad.copy()
    #
    # for i in range(1, image_gr_pad.shape[0] - 1):
    #     for j in range(1, image_gr_pad.shape[1] - 1):
    #         neighborhood = [image_gr_pad[i-1][j-1], image_gr_pad[i-1][j], image_gr_pad[i-1][j+1],
    #                         image_gr_pad[i][j-1], image_gr_pad[i][j], image_gr_pad[i][j+1],
    #                         image_gr_pad[i+1][j-1], image_gr_pad[i+1][j], image_gr_pad[i+1][j+1]]
    #         image_filt_alg[i][j] = st.median(neighborhood)
    #
    # # Post processing the image to delete the zero-padded rows and columns:
    # image_filt_alg = np.delete(image_filt_alg, 0, 0)
    # image_filt_alg = np.delete(image_filt_alg, 0, 1)
    # image_filt_alg = np.delete(image_filt_alg, -1, 0)
    # image_filt_alg = np.delete(image_filt_alg, -1, 1)
    #
    # cv2.namedWindow('image_filt_alg')
    # image_filt_alg_r = cv2.resize(image_filt_alg, (650, 800))
    # cv2.imshow('image_filt_alg', image_filt_alg_r)
    # cv2.waitKey(0)
    #
    # # This is the image, in which for "noise" data images (noise = 1), we are going to implement thresholding :
    # image_to_bin = image_filt_alg
    # Median filtering without using an OpenCV function :

# Computing the integral image of the data image :
my_image = image_to_bin
my_image_int = cv2.integral(my_image)

# Printing the dimensions of the integral image to check:
print("The shape of the integral image is :", my_image_int.shape)

# Converting to a binary image with the cv2.threshold() :
thresh_otsu, image_bin = cv2.threshold(src=image_to_bin, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)         #  first tests of OTSU METHOD

cv2.namedWindow('image_bin')
image_bin_r = cv2.resize(image_bin, (650, 800))
cv2.imshow('image_bin', image_bin_r)
cv2.waitKey(0)

# Using some Morphological Operations in order to extract bigger regions of objects. Target is the words :
# Dilation kernels for word detection for each image type :
if "2" in image_path:
    strel_dil_words = cv2.getStructuringElement(cv2.MORPH_CROSS, (8,3))
    print("GET IMAGE 2")
elif "3" in image_path:
    strel_dil_words = cv2.getStructuringElement(cv2.MORPH_CROSS, (14, 8))
    print("GET IMAGE 3")
else :
    strel_dil_words = cv2.getStructuringElement(cv2.MORPH_CROSS, (10, 10))
    print("GET ANOTHER IMAGE ")

# strel_dil_words = cv2.getStructuringElement(cv2.MORPH_RECT, (10,3))     # need to be bigger in order NOT to find more words
image_dil_words = cv2.morphologyEx(image_bin, cv2.MORPH_DILATE, strel_dil_words)

cv2.namedWindow('image_dil_words')
image_dil_words_r = cv2.resize(image_dil_words, (650, 800))
cv2.imshow('image_dil_words', image_dil_words_r)
cv2.waitKey(0)

# Using some Morphological Operations before finding the contours, in order to extract bigger regions of objects. Target is the regions of texts :

# Dilation :

strel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
image_dil = cv2.morphologyEx(image_bin, cv2.MORPH_DILATE, strel_dil)

cv2.namedWindow('image_dil')
image_dil_r = cv2.resize(image_dil, (650, 800))
cv2.imshow('image_dil', image_dil_r)
cv2.waitKey(0)

# Erosion :

strel_eros = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))                                          # we need a rectangular kernel in order not to deform the desirable for detection regions
image_eros = cv2.morphologyEx(image_dil, cv2.MORPH_ERODE, strel_eros)

cv2.namedWindow('image_eros')
image_eros_r = cv2.resize(image_eros, (650, 800))
cv2.imshow('image_eros', image_eros_r)
cv2.waitKey(0)

# Closing :

strel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 60))
image_close = cv2.morphologyEx(image_eros, cv2.MORPH_CLOSE, strel_close)

cv2.namedWindow('image_close')
image_close_r = cv2.resize(image_close, (650, 800))
cv2.imshow('image_close', image_close_r)
cv2.waitKey(0)

# Dilation on the y axis :

strel_dil_y = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 20))
image_dil_y = cv2.morphologyEx(image_close, cv2.MORPH_DILATE, strel_dil_y)

cv2.namedWindow('image_dil_y')
image_dil_y_r = cv2.resize(image_dil_y, (650, 800))
cv2.imshow('image_dil_y', image_dil_y_r)
cv2.waitKey(0)

# Computing Contours:
image_bin_contours = cv2.findContours(image=image_dil_y, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

# Making a copy of the initial image:
image_to_write = image.copy()

image_contours = image_bin_contours[0] if len(image_bin_contours) == 2 else image_bin_contours[1]         # THIS MAY BE SIMPLE AS : image_contours = image_bin_contours[1]

# This is done for the right visual requirements:
image_contours.reverse()

counter = 0

# Try keep data at a csv file :
rule = "if_else_line_152---median_cv2"  # UPDATE WHEN CHANGING !!!

header = ['filename', 'rule', 'region', 'pxl_area', 'bb_area', 'words', 'mean_gr_val']

data_list = []

# Computing the cv2.boundingRect() and cv2.rectangle() :
for cntr in image_contours:
    counter += 1
    x, y, w, h = cv2.boundingRect(cntr)
    cv2.rectangle(img = image_to_write, pt1=(x, y), pt2=(x + w, y + h), color=(0, 0, 255), thickness=2)
    cv2.putText(img=image_to_write, text=str(counter), org=(x, y + 50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=1, bottomLeftOrigin=0)

    print("@@@@@@@@@@@@@@@@@@@@@@@@")
    print("contour: ", counter)
    print("x,y,w,h:", x, y, w, h)

    #   Computing the text-pixel area of a region (question 2a) :

    # Cropping the image :
    image_crop = []
    image_crop = image_bin[y:y+h, x:x+w].copy()

    # This imshow() will be helpful for the report
    cv2.namedWindow('image_crop_'+str(counter))
    cv2.imshow('image_crop_'+str(counter), image_crop)
    cv2.waitKey(0)

    print("---- Region ", counter, ": ----")
    print("Area (px): ", np.sum(image_crop == 255))

    # Computing the bounding box-pixel area of a region :
    bound_box_pxls = w*h

    print("Bounding Box Area(px): ", bound_box_pxls)

    # Computing the number of words in a region :
    # We'll use the cv2.connectedComponents() :
    # Cropping the image :
    image_crop_words = image_dil_words[y:y + h, x:x + w].copy()

    # This imshow() will be helpful for the report :
    cv2.namedWindow('image_crop_words_' + str(counter))
    cv2.imshow('image_crop_words_' + str(counter), image_crop_words)
    cv2.waitKey(0)

    comp_labels, image_con_comp = cv2.connectedComponents(image_crop_words, connectivity = 4)

    # This '-1' stands for the subtraction of the background as a label :
    print("There are ", comp_labels - 1, "words.")

    # Computing the mean grayscale value of the pixels inside the bounding box area of a region :

    sum_gr = my_image_int[y + h][x + w] + my_image_int[y][x] - my_image_int[y + h][x] - my_image_int[y][x + w]

    mean_gr = sum_gr / bound_box_pxls

    print("Mean gray-level value in bounding box: ", mean_gr)

#   Try keep data at a csv file :
    filename = image_path[8:-4]
    data = [filename, rule, counter, 0, bound_box_pxls, comp_labels - 1, mean_gr]

    data_list.append(data)

with open('results/measurements.csv', 'a', encoding='UTF8', newline='') as f:               # 'w' for first time writing to the csv file, 'a' for editing the csv file
    writer = csv.writer(f)

    # write the header
    # writer.writerow(header)                                                       # uncommented for first time writing to the csv file, commented for editing the csv file

    # write multiple rows
    writer.writerows(data_list)

cv2.namedWindow('image_to_write')
image_to_write_r = cv2.resize(image_to_write, (650, 800))
cv2.imshow('image_to_write', image_to_write_r)
cv2.waitKey(0)

# Saving the image with the text regions bounding boxes :
write_path = "results/" + image_path[8:-4] + "_res.png"
cv2.imwrite(write_path, image_to_write)
