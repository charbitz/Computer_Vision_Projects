import numpy as np
import cv2

def panorama(image_2, image_1):

    # sift for images :
    sift = cv2.xfeatures2d_SIFT.create(2000)

    # surf = cv2.xfeatures2d_SURF.create(400)
    # surf = cv2.xfeatures2d_SURF.create(hessianThreshold=400,nOctaves=None,nOctaveLayers=None,extended=1,upright=None)
    #
    # kp1, desc1 = surf.detectAndCompute(image_1, None)
    # kp2, desc2 = surf.detectAndCompute(image_2, None)

    # sift keypoints and descriptors of image :
    kp1 = sift.detect(image_1)
    desc1 = sift.compute(image_1, kp1)

    # sift keypoints and descriptors of image_2 :
    kp2 = sift.detect(image_2)
    desc2 = sift.compute(image_2, kp2)

    # match_2 matches :
    matches = matching(desc1[1], desc2[1])

    image_w_matches = cv2.drawMatches(image_1, desc1[0], image_2, desc2[0], matches, None)
    cv2.namedWindow('image_w_matches for ' + image_path[8:-4])
    image_w_matches_r = cv2.resize(image_w_matches,  (800, 650))
    cv2.imshow('image_w_matches for ' + image_path[8:-4], image_w_matches_r)
    cv2.waitKey(0)
    # cv2.destroyWindow('image_w_matches')

    # implementation for matches_2 :
    img_pt1 = []
    img_pt2 = []
    for x in matches:
        img_pt1.append(kp1[x.queryIdx].pt)
        img_pt2.append(kp2[x.trainIdx].pt)
    img_pt1 = np.array(img_pt1)
    img_pt2 = np.array(img_pt2)

    # homography :
    M = 0
    mask = 0
    M, mask = cv2.findHomography(img_pt2, img_pt1, cv2.RANSAC)

    # merged image :
    merged_image = []
    merged_image = cv2.warpPerspective(image_2, M, (image_1.shape[1] + 1000, image_1.shape[0] + 1000))

    cv2.namedWindow('merged_image for ' + image_path[8:-4])
    merged_image_r = cv2.resize(merged_image,  (800, 650))
    cv2.imshow('merged_image for ' + image_path[8:-4], merged_image_r)
    cv2.waitKey(0)
    # cv2.destroyWindow('merged_image')

    merged_image[0: image_1.shape[0], 0: image_1.shape[1]] = image_1

    cv2.namedWindow('merged_image_2 for ' + image_path[8:-4])
    merged_image_r = cv2.resize(merged_image,  (800, 650))
    cv2.imshow('merged_image_2 for ' + image_path[8:-4], merged_image_r)
    cv2.waitKey(0)
    # cv2.destroyWindow('merged_image_2')

    return merged_image

def matching(d1, d2):
    n1 = d1.shape[0]
    n2 = d2.shape[0]

    matches_1 = []
    for i in range(n1):
        row_d1 = d1[i, :]
        diff = d2 - row_d1
        diff = np.abs(diff)
        distances = np.sum(diff, axis=1)

        min_dist_pos_1 = np.argmin(distances)
        min_dist_1 = distances[min_dist_pos_1]

        distances[min_dist_pos_1] = np.inf

        sec_min_dist_pos_1 = np.argmin(distances)
        sec_min_dist_1 = distances[sec_min_dist_pos_1]

        if min_dist_1 / sec_min_dist_1 < 0.5:
            matches_1.append(cv2.DMatch(i, min_dist_pos_1, min_dist_1))

    matches_2 = []
    for j in range(n2):
        row_d2 = d2[j, :]
        diff2 = d1 - row_d2
        diff2 = np.abs(diff2)
        distances2 = np.sum(diff2, axis=1)

        min_dist_pos_2 = np.argmin(distances2)
        min_dist_2 = distances2[min_dist_pos_2]

        distances2[min_dist_pos_2] = np.inf

        sec_min_dist_pos_2 = np.argmin(distances2)
        sec_min_dist_2 = distances2[sec_min_dist_pos_2]

        if min_dist_2 / sec_min_dist_2 < 0.5:
            matches_2.append(cv2.DMatch(j, min_dist_pos_2, min_dist_2))

    # # Initialize the final lists of matching:
    fin_matches = []

    # Find the list of matches with the min length :
    if len(matches_1) < len(matches_2):
        min_len_matches = matches_1.copy()
        max_len_matches = matches_2.copy()
    else:
        min_len_matches = matches_2.copy()
        max_len_matches = matches_1.copy()
    #
    # # Cross checking:
    # # for i in min_len_matches:
    # #     if min_len_matches[i].queryIdx == max_len_matches[]
    #
    # for i in range(len(max_len_matches)):
    #     for j in range(len(min_len_matches)):
    #         if min_len_matches[j].queryIdx == max_len_matches[i].trainIdx and min_len_matches[j].trainIdx == max_len_matches[i].queryIdx:
    #             fin_matches.append(cv2.DMatch(min_len_matches[j].queryIdx, min_len_matches[j].trainIdx, min_len_matches[j].distance))



    # return fin_matches
    # return fin_matches
    return matches_1

def cropping(image):

    image_gr = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # cv2.namedWindow('image_gr')
    # image_gr_r = cv2.resize(image_gr, (800, 650))
    # cv2.imshow('image_gr', image_gr_r)
    # cv2.waitKey(0)

    thresh, image_bin = cv2.threshold(src=image_gr, thresh=0, maxval=255, type=cv2.THRESH_BINARY)
    # cv2.namedWindow('image_bin')
    # image_bin_r = cv2.resize(image_bin,  (800, 650))
    # cv2.imshow('image_bin', image_bin_r)
    # cv2.waitKey(0)

    image_bin_contours = cv2.findContours(image=image_bin, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

    cnt = image_bin_contours[0]
    x, y, w, h = cv2.boundingRect(cnt)

    crop = image[y:y + h, x:x + w]

    cv2.namedWindow('crop for ' + image_path[8:-4])
    crop_r = cv2.resize(crop,  (800, 650))
    cv2.imshow('crop for ' + image_path[8:-4], crop_r)
    cv2.waitKey(0)

    return crop

# def panorama_surf(image_1, image_2):
#
#     surf = cv2.SURF(400)
#     surf.hessianThreshold = 50000
#
#     kp1, desc1 = surf.detectAndCompute(image_1, None)
#     kp2, desc2 = surf.detectAndCompute(image_2, None)
#
#     image_1w = cv2.drawKeypoints(image_1, kp1, None, (255, 0, 0), 4)
#
#     cv2.namedWindow('image_1w for' + image_path[8:-4])
#     image_1w_r = cv2.resize(image_1w, (800, 650))
#     cv2.imshow('image_1w for ' + image_path[8:-4], image_1w_r)
#     cv2.waitKey(0)
#
#     image_2w = cv2.drawKeypoints(image_2, kp2, None, (255, 0, 0), 4)
#
#     cv2.namedWindow('image_2w for' + image_path[8:-4])
#     image_2w_r = cv2.resize(image_2w, (800, 650))
#     cv2.imshow('image_2w for ' + image_path[8:-4], image_2w_r)
#     cv2.waitKey(0)


first_path = 'dataset/yard-house/yard-house-01.png'
first_image = cv2.imread(first_path)
first_element = first_image

images = ['dataset/yard-house/yard-house-02.png', 'dataset/yard-house/yard-house-03.png',
          'dataset/yard-house/yard-house-04.png','dataset/yard-house/yard-house-05.png']


# first_path = 'dataset/my-house/scene2/png/image_1.png'
# first_image = cv2.imread(first_path)
# first_element = first_image

# images = ['dataset/my-house/scene2/png/image_2.png', 'dataset/my-house/scene2/png/image_3.png',
#           'dataset/my-house/scene2/png/image_4.png']

for image_path in images:
    image = cv2.imread(image_path)

    cv2.namedWindow('image for ' + image_path[8:-4])
    image_r = cv2.resize(image,  (800, 650))
    cv2.imshow('image for ' + image_path[8:-4], image_r)
    cv2.waitKey(0)

    final_image = panorama(first_element, image)

    cv2.namedWindow('final_image for ' + image_path[8:-4])
    final_image_r = cv2.resize(final_image, (800, 650))
    cv2.imshow('final_image for ' + image_path[8:-4], final_image_r)
    cv2.waitKey(0)

    image_cropped = cropping(final_image)

    cv2.namedWindow('image_cropped for ' + image_path[8:-4])
    image_cropped_r = cv2.resize(image_cropped,  (800, 650))
    cv2.imshow('image_cropped for ' + image_path[8:-4], image_cropped_r)
    cv2.waitKey(0)

    first_element = image_cropped
    print("first_element changed !")