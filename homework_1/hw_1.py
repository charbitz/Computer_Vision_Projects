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

# Median filtering:

nois_1_after_median = cv2.medianBlur(nois_1, ksize=3)

cv2.namedWindow('nois_1_after_median')
nois_1_after_median_r = cv2.resize(nois_1_after_median, (650, 800))
cv2.imshow('nois_1_after_median', nois_1_after_median_r)
cv2.waitKey(0)

# Computing the histogram of the filtered image:

plt.hist(nois_1_after_median.ravel(),256,[0,256])
plt.title("Histogram of nois_1_after_median")
plt.show()

# Converting to Binary image:

retval, filt_med_1_bin = cv2.threshold(nois_1_after_median, thresh=215, maxval = 255, type=cv2.THRESH_BINARY)           #  first tests of the function

cv2.namedWindow('filt_med_1_bin')
filt_med_1_bin_r = cv2.resize(filt_med_1_bin, (650, 800))
cv2.imshow('filt_med_1_bin', filt_med_1_bin_r)
cv2.waitKey(0)

# Computing an integral image. It'll be helpfull later:

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