import cv2
import numpy as np
import glob
original = cv2.imread("img/ibm-1.png")

# Sift and Flann
sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)

# Load all the images
all_images_to_compare = []
titles = []
for f in glob.iglob("img/*"):
    image = cv2.imread(f)
    titles.append(f)
    all_images_to_compare.append(image)

for image_to_compare, title in zip(all_images_to_compare, titles):
    # Check for similarities between the 2 images
    kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

    matches = flann.knnMatch(desc_1, desc_2, k=2)
    good_points = []
    for m, n in matches:
        if m.distance < 0.6*n.distance:
            good_points.append(m)
    number_keypoints = 0
    if len(kp_1) <= len(kp_2):
        number_keypoints = len(kp_1)
    else:
        number_keypoints = len(kp_2)

    print("Title: " + title)
    percentage_similarity = len(good_points) / float(number_keypoints) * 100
    print("Similarity: " + str(int(percentage_similarity)) + "\n")
