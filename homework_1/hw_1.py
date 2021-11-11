import cv2
import numpy as np

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

# Linear filtering:

# kernel = np.array([[-8,0,8], [-2,0,2], [-8,0,8]])
kernel = 1/9*np.ones((3,3))

nois_1_filt = cv2.filter2D(nois_1, cv2.CV_8UC1, kernel)

cv2.namedWindow('1_noise_filtered')
nois_1_filt_r = cv2.resize(nois_1_filt, (650, 800))
cv2.imshow('1_noise_filtered', nois_1_filt_r)
cv2.waitKey(0)

nois_1_after_median = cv2.medianBlur(nois_1, ksize=3)

cv2.namedWindow('nois_1_after_median')
nois_1_after_median_r = cv2.resize(nois_1_after_median, (650, 800))
cv2.imshow('nois_1_after_median', nois_1_after_median_r)
cv2.waitKey(0)

# Non linear filtering:

strel_cr = cv2.getStructuringElement(cv2.MORPH_CROSS, (3,3))
print("strel MORPH_CROSS")
print(strel_cr)

nois_1_eros = cv2.morphologyEx(nois_1, cv2.MORPH_ERODE, strel_cr)

cv2.namedWindow('nois_1_eros')
nois_1_eros_r = cv2.resize(nois_1_eros, (650, 800))
cv2.imshow('nois_1_eros', nois_1_eros_r)
cv2.waitKey(0)

# Converting to Binary image:
# filt_med_1_bin = cv2.threshold(nois_1_after_median,0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
#
# cv2.namedWindow('filt_med_1_bin')
# cv2.imshow('filt_med_1_bin', filt_med_1_bin)
# cv2.waitKey(0)