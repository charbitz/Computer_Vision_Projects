import cv2
import numpy as np
import matplotlib.pyplot as plt

orig_1_path = 'dataset/1_original.png'
nois_1_path = 'dataset/1_noise.png'

orig_1 = cv2.imread(orig_1_path, cv2.IMREAD_GRAYSCALE)
nois_1 = cv2.imread(nois_1_path, cv2.IMREAD_GRAYSCALE)

print(orig_1.shape)
print(nois_1.shape)

cv2.namedWindow('1_original')
orig_1_r = cv2.resize(orig_1, (650, 800))
cv2.imshow('1_original', orig_1_r)
cv2.waitKey(0)

cv2.namedWindow('1_noise')
nois_1_r = cv2.resize(nois_1, (650, 800))
cv2.imshow('1_noise', nois_1_r)
cv2.waitKey(0)

# Median filtering:             This is going to be implemented without OpenCV methods !!!

nois_1_after_median = cv2.medianBlur(nois_1, ksize=3)

cv2.namedWindow('nois_1_after_median')
nois_1_after_median_r = cv2.resize(nois_1_after_median, (650, 800))
cv2.imshow('nois_1_after_median', nois_1_after_median_r)
cv2.waitKey(0)

# Computing the histogram of the filtered image:

plt.hist(nois_1_after_median.ravel(),256,[0,256])
plt.title("Histogram of nois_1_after_median")
plt.show()

# Converting the image 'dataset/1_noise' to a Binary image:

retval_no, filt_med_1_bin = cv2.threshold(nois_1_after_median, thresh=215, maxval=255, type=cv2.THRESH_BINARY_INV)           #  first tests of the function

cv2.namedWindow('filt_med_1_bin')
filt_med_1_bin_r = cv2.resize(filt_med_1_bin, (650, 800))
cv2.imshow('filt_med_1_bin', filt_med_1_bin_r)
cv2.waitKey(0)

# Converting the image 'dataset/1_original' to a Binary image:

retval_or, orig_1_bin = cv2.threshold(orig_1, thresh=215, maxval=255, type=cv2.THRESH_BINARY_INV)                            #  first tests of the function

cv2.namedWindow('orig_1_bin')
orig_1_bin_r = cv2.resize(orig_1_bin, (650, 800))
cv2.imshow('orig_1_bin', orig_1_bin_r)
cv2.waitKey(0)

# Boundary extraction of 'dataset/1_original' binary image :

kernel = np.ones((3,3),np.uint8)
orig_1_bin_bound = cv2.morphologyEx(orig_1_bin, cv2.MORPH_GRADIENT, kernel)

cv2.namedWindow('orig_1_bin_bound')
orig_1_bin_bound_r = cv2.resize(orig_1_bin_bound, (650, 800))
cv2.imshow('orig_1_bin_bound', orig_1_bin_bound)

cv2.waitKey(0)

# Boundary extraction of 'dataset/1_noise' binary image :

nois_1_bin_bound = cv2.morphologyEx(filt_med_1_bin, cv2.MORPH_GRADIENT, kernel)

cv2.namedWindow('nois_1_bin_bound')
nois_1_bin_bound_r = cv2.resize(nois_1_bin_bound, (650, 800))
cv2.imshow('nois_1_bin_bound', nois_1_bin_bound)
cv2.waitKey(0)











# Taking the connected components of the 'dataset/1_original' binary image. It'll be helpful later :

or_1_con_comp_labels, or_1_con_comp = cv2.connectedComponents(orig_1_bin)

cv2.namedWindow('or_1_con_comp')
or_1_con_comp_r = cv2.resize(or_1_con_comp.astype('uint8'), (650, 800))                 ##### worked with " .astype('uint8') " (THIS DETAIL IS ONLY FOR RESIZING)
cv2.imshow('or_1_con_comp', or_1_con_comp_r)
cv2.waitKey(0)

print(or_1_con_comp_labels)

# Normalizing the pixel values of the 'dataset/1_original' image with the connected components  DONT KNOW IF IT WILL BE HELPFUL :

or_1_con_comp_norm = cv2.normalize(or_1_con_comp, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

cv2.namedWindow('or_1_con_comp')
or_1_con_comp_norm_r = cv2.resize(or_1_con_comp_norm.astype('uint8'), (650, 800))                 ##### worked with " .astype('uint8') " (THIS DETAIL IS ONLY FOR RESIZING)
cv2.imshow('or_1_con_comp', or_1_con_comp_norm_r)
cv2.waitKey(0)



# Computing an integral image. It'll be helpful later:

# HERE my_image = orig_1 e.g. LATER PUT my_image = {the desirable image } :
my_image = orig_1
my_image_int = cv2.integral(my_image)

# Printing the dimensions of the integral image to check:
print("integral image of orig_1 and shape:", my_image_int.shape)

# Because the dimensions of the integral image are +1 row and +1 col (the first ones are the additionals), we have to delete them:
# It was checked from the print of the shape and from the debugger (ndarray view as array).

my_image_int_del = np.delete(my_image_int, 0, 0)
my_image_int_del = np.delete(my_image_int_del, 0, 1)

print("integral image of orig_1 and shape after deleting:", my_image_int_del.shape)