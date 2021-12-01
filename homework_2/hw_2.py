import numpy as np
import cv2

# image :
# image_path = "notre-dame-1.jpg"
image_path = 'dataset/yard-house/yard-house-05.png'
# image_path = 'dataset/my-images/png/image_1.png'
image = cv2.imread(image_path)

cv2.namedWindow('image')
cv2.imshow('image', image)
cv2.waitKey(0)

# image_gr = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
#
# cv2.namedWindow('image_gr')
# cv2.imshow('image_gr', image_gr)
# cv2.waitKey(0)

# sift for images :
sift = cv2.xfeatures2d_SIFT.create(2000)

# sift keypoints and descriptors of image :
kp1 = sift.detect(image)                 # tosa keypoints, octave
desc1 = sift.compute(image, kp1)         # desc1[1]: grammes keiypoints , sthles einai ta 128  characteristics toy kaue keypoint

# image with keypoints :
# image_w_kp = cv2.drawKeypoints(image, kp1, image)
#
# cv2.namedWindow('image')
# cv2.imshow('image', image_w_kp)
# cv2.waitKey(0)

# image_2 :
image_path2 = 'dataset/yard-house/yard-house-04.png'
# image_path2 = "notre-dame-2.jpg"
image_2 = cv2.imread(image_path2)

cv2.namedWindow('image_2')
cv2.imshow('image_2', image_2)
cv2.waitKey(0)

# sift keypoints and descriptors of image_2 :
kp2 = sift.detect(image_2)
desc2 = sift.compute(image_2, kp2)

# image_2 with keypoints :
# image_2_w_kp = cv2.drawKeypoints(image_2, kp2, image_2)
#
# cv2.namedWindow('image_2')
# cv2.imshow('image_2', image_2_w_kp)
# cv2.waitKey(0)

def match1(d1, d2):
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    matches = []
    for i in range(n1):
        fv = d1[i, :]
        diff = d2 - fv
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        i2 = np.argmin(distances)
        mindist2 = distances[i2]

        matches.append(cv2.DMatch(i, i2, mindist2))

    return matches

# match_1 matches :
matches_1 = match1(desc1[1], desc2[1])

image_match_1 = cv2.drawMatches(image, desc1[0], image_2, desc2[0], matches_1, None)
cv2.namedWindow('match1')
cv2.imshow('match1', image_match_1)
cv2.waitKey(0)

def match2(d1, d2):
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    matches = []
    for i in range(n1):
        fv = d1[i, :]
        diff = d2 - fv
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        i2 = np.argmin(distances)
        mindist2 = distances[i2]

        distances[i2] = np.inf

        i3 = np.argmin(distances)
        mindist3 = distances[i3]

        if mindist2 / mindist3 < 0.5:
            matches.append(cv2.DMatch(i, i2, mindist2))

    return matches

# match_2 matches :
matches_2 = match2(desc1[1], desc2[1])

image_match_2 = cv2.drawMatches(image, desc1[0], image_2, desc2[0], matches_2, None)
cv2.namedWindow('match2')
cv2.imshow('match2', image_match_2)
cv2.waitKey(0)

# brute force matches :
bf = cv2.BFMatcher(cv2.NORM_L1, crossCheck=True)              # συνηθως βαζω την NORM_L1 επιεδη ειναι πιο γρήγορη , το crosscheck κανει απο την μια προσ την αλλη και απο την αλλη προς την μια
bf_matches = bf.match(desc1[1], desc2[1])

image_match_bf = cv2.drawMatches(image, desc1[0], image_2, desc2[0], bf_matches, None)
cv2.namedWindow('bf_match')
cv2.imshow('bf_match', image_match_bf)
cv2.waitKey(0)

# implementation for matches_2 :
img_pt1 = []
img_pt2 = []
for x in matches_2:
    img_pt1.append(kp1[x.queryIdx].pt)
    img_pt2.append(kp2[x.trainIdx].pt)
img_pt1 = np.array(img_pt1)
img_pt2 = np.array(img_pt2)

# homography :
M, mask = cv2.findHomography(img_pt2, img_pt1, cv2.RANSAC)

# merged image :
merged_image = cv2.warpPerspective(image_2, M, (image.shape[1]+1000, image.shape[0]+1000))

cv2.namedWindow('merged_image')
cv2.imshow('merged_image', merged_image)
cv2.waitKey(0)

merged_image[0: image.shape[0], 0: image.shape[1]] = image

cv2.namedWindow('merged_image_2')
cv2.imshow('merged_image_2', merged_image)
cv2.waitKey(0)

