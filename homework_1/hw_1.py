import cv2
import numpy as np
import matplotlib.pyplot as plt
import statistics as st

image_path = 'dataset/1_noise.png'

noise_images = ('dataset/1_noise.png', 'dataset/2_noise.png', 'dataset/3_noise.png', 'dataset/4_noise.png', 'dataset/5_noise.png')

noise = 1 if (image_path in noise_images) else 0

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
    # Median filtering :
    image_filt_cv = cv2.medianBlur(image_gr, ksize=3)

    # Median filtering without using an OpenCV method :
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

    cv2.namedWindow('image_filt_cv')
    image_filt_cv_r = cv2.resize(image_filt_cv, (650, 800))
    cv2.imshow('image_filt_cv', image_filt_cv_r)
    cv2.waitKey(0)

    # cv2.namedWindow('image_filt_alg')
    # image_filt_alg_r = cv2.resize(image_filt_alg, (650, 800))
    # cv2.imshow('image_filt_alg', image_filt_alg_r)
    # cv2.waitKey(0)

    # Computing the histogram of the filtered image:
    # plt.hist(image_filt_cv.ravel(), 256, [0, 256])
    # plt.title("Histogram of nois_1_after_median")
    # plt.show()

    # This is the image, in which for "noise" data images (noise = 1), we are going to implement thresholding :
    image_to_bin = image_filt_cv
    # print(image_filtered.shape)

# Computing the integral image of the data image :
my_image = image_to_bin
my_image_int = cv2.integral(my_image)

# Printing the dimensions of the integral image to check:
print("The shape of the integral image is :", my_image_int.shape)

# Because the dimensions of the integral image are +1 row and +1 col (the first ones are the additionals), we have to delete them :
my_image_int_del = np.delete(my_image_int, 0, 0)
my_image_int_del = np.delete(my_image_int_del, 0, 1)

print("The shape of the integral image after processing is :", my_image_int_del.shape)

# Converting to a binary image with the cv2.threshold() :
thresh_otsu, image_bin = cv2.threshold(src=image_to_bin, thresh=0, maxval=255, type=cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)         #  first tests of OTSU METHOD

cv2.namedWindow('image_bin')
image_bin_r = cv2.resize(image_bin, (650, 800))
cv2.imshow('image_bin', image_bin_r)
cv2.waitKey(0)

# Using some Morphological Operations in order to extract bigger regions of objects. Target is the words :
strel_dil_words = cv2.getStructuringElement(cv2.MORPH_RECT, (15,8))
image_dil_words = cv2.morphologyEx(image_bin, cv2.MORPH_DILATE, strel_dil_words)

cv2.namedWindow('image_dil_words')
image_dil_words_r = cv2.resize(image_dil_words, (650, 800))
cv2.imshow('image_dil_words', image_dil_words_r)
cv2.waitKey(0)

# Using some Morphological Operations before finding the contours, in order to extract bigger regions of objects. Target is the regions of texts :
# Starting with dilation, in order to get the text lines together as one region :
strel_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (45, 45))
image_dil = cv2.morphologyEx(image_bin, cv2.MORPH_DILATE, strel_dil)

cv2.namedWindow('image_dil')
image_dil_r = cv2.resize(image_dil, (650, 800))
cv2.imshow('image_dil', image_dil_r)
cv2.waitKey(0)

# Then we apply erosion to set apart the desirable regions :
strel_eros = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 50))                                          # we need a rectangular kernel in order not to deform the desirable for detection regions
print("strel MORPH_RECT")
print(strel_eros)

image_eros = cv2.morphologyEx(image_dil, cv2.MORPH_ERODE, strel_eros)

cv2.namedWindow('image_eros')
image_eros_r = cv2.resize(image_eros, (650, 800))
cv2.imshow('image_eros', image_eros_r)
cv2.waitKey(0)

# Then we apply closing to produce less false regions (regions with black pixels) :
strel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 60))
image_close = cv2.morphologyEx(image_eros, cv2.MORPH_CLOSE, strel_close)

cv2.namedWindow('image_close')
image_close_r = cv2.resize(image_close, (650, 800))
cv2.imshow('image_close', image_close_r)
cv2.waitKey(0)

# Computing Contours:
image_bin_contours = cv2.findContours(image=image_close, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

# Making a copy of the initial image:
image_copy = image.copy()

image_contours = image_bin_contours[0] if len(image_bin_contours) == 2 else image_bin_contours[1]         # THIS MAY BE SIMPLE AS : image_contours = image_bin_contours[1]

# This is done for the right visual requirements:
image_contours.reverse()

counter = 0

# Computing the cv2.boundingRect() and cv2.rectangle() :
for cntr in image_contours:
    counter += 1
    x, y, w, h = cv2.boundingRect(cntr)
    cv2.rectangle(img = image_copy, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=2)
    cv2.putText(img=image_copy, text=str(counter), org=(x,y+50), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=2, color=(0, 0, 255), thickness=4, lineType=1, bottomLeftOrigin=0)

    print("@@@@@@@@@@@@@@@@@@@@@@@@")
    print("contour: ", counter)
    print("x,y,w,h:", x, y, w, h)

    #   Computing the text-pixel area of a region (question 2a) :
    text_pxls = 0

    for i in range(y, y+h+1):
        for j in range(x, x+w+1):
            if (image_bin[i][j] == 255):
               text_pxls += 1

    print("---- Region ", counter, ": ----")
    print("Area (px): ", text_pxls)

    # Computing the bounding box-pixel area of a region :
    bound_box_pxls = w*h

    print("Bounding Box Area(px): ", bound_box_pxls)

    # Computing the number of words in a region :
    # We'll use the cv2.connectedComponents() :
    comp_labels, image_con_comp = cv2.connectedComponents(image_dil_words[y:y+h][x:x+w], connectivity = 8)

    # This '-1' stands for the subtraction of the background as a label :
    print("There are ", comp_labels - 1, "words.")

    # Computing the mean grayscale value of the pixels inside the bounding box area of a region :
    sum_gr = my_image_int_del[y + h][x + w] + my_image_int_del[y][x] - my_image_int_del[y + h][x] - my_image_int_del[y][x + w]
    mean_gr = sum_gr / bound_box_pxls
    print("Mean gray-level value in bounding box: ", mean_gr)

cv2.namedWindow('image_copy')
image_copy_r = cv2.resize(image_copy, (650, 800))
cv2.imshow('image_copy', image_copy_r)
cv2.waitKey(0)









# Boundary extraction of binary image :                                                                 # NOT SURE IF THAT WILL HELP !!!
kernel = np.ones((3,3), np.uint8)
image_bin_bounds = cv2.morphologyEx(image_bin, cv2.MORPH_GRADIENT, kernel)

cv2.namedWindow('image_bin_bounds')
image_bin_bounds_r = cv2.resize(image_bin_bounds, (650, 800))
cv2.imshow('image_bin_bounds', image_bin_bounds_r)
cv2.waitKey(0)







# Taking the connected components of the binary image. It'll be helpful later :

image_con_comp_labels, image_bin_con_comp = cv2.connectedComponents(image_bin)

cv2.namedWindow('image_bin_con_comp')
image_bin_con_comp_r = cv2.resize(image_bin_con_comp.astype('uint8'), (650, 800))                   # worked with " .astype('uint8') " (THIS DETAIL IS ONLY FOR RESIZING)
cv2.imshow('image_bin_con_comp', image_bin_con_comp_r)
cv2.waitKey(0)

print("The number of labels found is :",image_con_comp_labels)

# Normalizing the pixel values of the image with the connected components :                             # NOT SURE IF THAT WILL HELP !!!

image_con_comp_norm = cv2.normalize(image_bin_con_comp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

cv2.namedWindow('or_1_con_comp')
image_con_comp_norm_r = cv2.resize(image_con_comp_norm.astype('uint8'), (650, 800))                     # worked with " .astype('uint8') " (THIS DETAIL IS ONLY FOR RESIZING)
cv2.imshow('or_1_con_comp', image_con_comp_norm_r)
cv2.waitKey(0)
