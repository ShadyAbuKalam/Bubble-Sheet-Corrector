import numpy as np
import cv2
import math

# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}


def display_normal(title, image):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
    cv2.waitKey(0)


def optimize_image_rotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 150)

    # find circles in the edge map, then find two almost on the same line
    dp_values = [2,3,4]
    for dp in dp_values:
        circles = cv2.HoughCircles(edged, cv2.HOUGH_GRADIENT, dp, 700, minRadius=35, maxRadius=45)
        if circles is None:
            continue
        else:
            circles = circles[0]
        #
        # for c in circles:
        #     cv2.circle(image,(c[0],c[1]),c[2],(255,255,0),3)
        # temp = image.copy()
        # for c in circles:
        #     cv2.circle(temp, (c[0], c[1]), c[2], (0, 255, 255), 3)
        #     cv2.circle(temp, (c[0], c[1]), 1, (0, 0, 255), 3)

        first = None
        second = None
        for i in range(len(circles)):
            for j in range(len(circles)):
                if i == j:  # we want different circles
                    continue

                if abs(circles[i][1] - circles[j][1]) < 50:
                    if circles[i][0] < circles[j][0]:
                        first = circles[i]
                        second = circles[j]
                    else:
                        first = circles[j]
                        second = circles[i]

                    break

        if first is None or second is None:
            continue
        rotation_center = (first[0], first[1])
        x_diff = second[0] - first[0]
        y_diff = second[1] - first[1]
        rotation_angle = math.atan2(y_diff, x_diff)
        rotation_angle = math.degrees(rotation_angle)
        rotation_matrix = cv2.getRotationMatrix2D(rotation_center, rotation_angle, 1)

        image = cv2.warpAffine(image, rotation_matrix, gray.T.shape)
        return image
    return None

def crop_answer_section(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    circle_with_cross_template = cv2.imread("circle.png")
    circle_with_cross_template = cv2.cvtColor(circle_with_cross_template, cv2.COLOR_BGR2GRAY)
    result = cv2.matchTemplate(gray, circle_with_cross_template, cv2.TM_CCORR_NORMED)

    objects_matched = []
    while len(objects_matched) < 2:

        minV, maxV, minLoc, maxLoc = cv2.minMaxLoc(result)

        for i in range(-3, 3):
            for j in range(-3, 3):
                result[maxLoc[1] + i, maxLoc[0] + j] = 0
        maxLoc = list(maxLoc)
        maxLoc[0] = maxLoc[0] + circle_with_cross_template.shape[1] // 2
        maxLoc[1] = maxLoc[1] + circle_with_cross_template.shape[0] // 2
        maxLoc = tuple(maxLoc)
        objects_matched.append(maxLoc)
    p1, p2 = sorted(objects_matched)
    image = image[p1[1]:p2[1], p1[0]:p2[0]]
    return image


import os,pathlib
failed_images = 0
tried_files=0
for f in pathlib.Path('test').iterdir():
    tried_files+=1
    try:
        image = cv2.imread(f.as_posix())
        image = optimize_image_rotation(image)
        image = crop_answer_section(image)
        #display_normal("Cropped version of image",image)

    except:
        print("File {0} failed".format(f.as_posix()))
        failed_images+=1

print("{0} files failed to crop".format(failed_images))