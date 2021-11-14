import cv2
import numpy as np
import matplotlib.pyplot as plt

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

if noise == 1 :
    # Median filtering :                                                                                # This is going to be implemented without OpenCV methods !!!
    filtered_image_gr = cv2.medianBlur(image_gr, ksize=3)

    cv2.namedWindow('filtered_image_gr')
    filtered_image_gr_r = cv2.resize(filtered_image_gr, (650, 800))
    cv2.imshow('filtered_image_gr', filtered_image_gr_r)
    cv2.waitKey(0)

    # Computing the histogram of the filtered image:
    plt.hist(filtered_image_gr.ravel(), 256, [0, 256])
    plt.title("Histogram of nois_1_after_median")
    plt.show()

    image_gr = filtered_image_gr

# Converting to a binary image with the cv2.threshold() :
retval, image_bin = cv2.threshold(image_gr, thresh=215, maxval=255, type=cv2.THRESH_BINARY_INV)         #  first tests of the function

cv2.namedWindow('image_bin')
image_bin_r = cv2.resize(image_bin, (650, 800))
cv2.imshow('image_bin', image_bin_r)
cv2.waitKey(0)


# Computing the cv2.findContours() :
image_bin_contours = cv2.findContours(image=image_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

# Making a copy of the initial image:
image_copy = image.copy()


image_contours = image_bin_contours[0] if len(image_bin_contours) == 2 else image_bin_contours[1]         # THIS MAY BE SIMPLE AS : image_contours = image_bin_contours[1]


counter = 0

# Computing the cv2.boundingRect() and cv2.rectangle() :
for cntr in image_contours:
    counter += 1
    x, y, w, h = cv2.boundingRect(cntr)
    cv2.rectangle(img = image_copy, pt1=(x, y), pt2=(x+w, y+h), color=(0, 0, 255), thickness=2)
    if counter < 11 :                                                                                     # THIS IS FOR PRINTING SOME RESULTS, DELETE AT THE END !!!
        print("new_cntr :", cntr)
        print("x,y,w,h:", x, y, w, h)

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







# Computing an integral image. It'll be helpful later:

# HERE my_image = image_gr e.g. LATER PUT my_image = {the desirable image } :
my_image = image_gr
my_image_int = cv2.integral(my_image)

# Printing the dimensions of the integral image to check:
print("The shape of the integral image is :", my_image_int.shape)

# Because the dimensions of the integral image are +1 row and +1 col (the first ones are the additionals), we have to delete them:
# It was checked from the print of the shape and from the debugger (ndarray view as array).

my_image_int_del = np.delete(my_image_int, 0, 0)
my_image_int_del = np.delete(my_image_int_del, 0, 1)

print("The shape of the integral image after processing is :", my_image_int_del.shape)