import cv2
import numpy as np

original = cv2.imread("img/ibm-1.png")
image_to_compare = cv2.imread("img/ibm-1.png")

# 2) Check for similarities between the 2 images
sift = cv2.xfeatures2d.SIFT_create()
kp_1, desc_1 = sift.detectAndCompute(original, None)
kp_2, desc_2 = sift.detectAndCompute(image_to_compare, None)

index_params = dict(algorithm=0, trees=5)
search_params = dict()
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc_1, desc_2, k=2)

good_points = []
ratio = 0.6
for m, n in matches:
    print(m.distance,n.distance)
    if m.distance < ratio*n.distance:
        good_points.append(m)
print(len(good_points))
result = cv2.drawMatches(original, kp_1, image_to_compare, kp_2, good_points, None)

# Define how similar they are
number_keypoints = 0
if len(kp_1) <= len(kp_2):
    number_keypoints = len(kp_1)
else:
    number_keypoints = len(kp_2)
print("Keypoints 1ST Image: " + str(len(kp_1)))
print("Keypoints 2ND Image: " + str(len(kp_2)))
print("GOOD Matches:", len(good_points))
match = (len(good_points) / float(number_keypoints))*100
print("How good it's the match: ", match, "%")

# cv2.imshow("result", result)
# cv2.imshow("Original", original)
# cv2.imshow("Duplicate", image_to_compare)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
