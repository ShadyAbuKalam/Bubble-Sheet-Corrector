import numpy as np
import cv2
import math
import pathlib
import csv

# define the answer key which maps the question number
# to the correct answer
ANSWER_KEY = {1: 2, 2: 3, 3: 1, 4: 1, 5: 4, 6: 1, 7: 3, 8: 3, 9: 1, 10: 3, 11: 1, 12: 2, 13: 3, 14: 3, 15: 2,
              16: 1, 17: 4, 18: 2, 19: 3, 20: 2, 21: 4, 22: 3, 23: 4, 24: 2, 25: 4, 26: 3, 27: 4, 28: 4, 29: 2, 30: 3,
              31: 2, 32: 2, 33: 4, 34: 3, 35: 2, 36: 3, 37: 2, 38: 3, 39: 3, 40: 1, 41: 2, 42: 2, 43: 3, 44: 3, 45: 2}


def mark(answers):
    total_mark = 0
    for k, v in ANSWER_KEY.items():
        if answers[k] == v:
            total_mark += 1
    return total_mark


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0

    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True

    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1

    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))

    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def display_normal(title, image):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
    cv2.waitKey(0)


def optimize_image_rotation(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 150)

    # find circles in the edge map, then find two almost on the same line
    dp_values = [2, 3, 4]
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

    # Pick 4 matches, then take the one to the most left and the one to the most right
    objects_matched = []
    while len(objects_matched) < 4:

        minV, maxV, minLoc, maxLoc = cv2.minMaxLoc(result)

        for i in range(-3, 3):
            for j in range(-3, 3):
                result[maxLoc[1] + i, maxLoc[0] + j] = 0
        maxLoc = list(maxLoc)
        maxLoc[0] = maxLoc[0] + circle_with_cross_template.shape[1] // 2
        maxLoc[1] = maxLoc[1] + circle_with_cross_template.shape[0] // 2
        maxLoc = tuple(maxLoc)

        objects_matched.append(maxLoc)

        objects_matched = sorted(objects_matched)
        p1 = objects_matched[0]
        p2 = objects_matched[-1]
    image = image[p1[1]:p2[1], p1[0]:p2[0]]
    return image


def have_nearby_contours(contours, needle):
    (n_x, n_y, n_w, n_h) = cv2.boundingRect(needle)

    for c in contours:
        (x, y, w, h) = cv2.boundingRect(c)
        if abs(n_x - x) < 10 and abs(n_y - y) < 10:
            return True
    return False


def find_question_contours(image, blur=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if blur:
        blurred_gray = cv2.GaussianBlur(gray, (3, 3), 1)
        edged = cv2.Canny(blurred_gray, 75, 150)
    else:
        edged = cv2.Canny(gray, 75, 150)
    # display_normal("Edge",edged)

    # todo : make it more generic to remove all lines
    # remove annoying vertical lines longer than 15 pixels
    linek = np.zeros((15, 11), dtype=np.uint8)
    linek[..., 5] = 1
    vlines = cv2.morphologyEx(edged, cv2.MORPH_OPEN, linek)
    edged = edged - vlines
    # display_normal("vLines",edged)

    # Dilate circles trying to complete them
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    edged = cv2.dilate(edged, kernel)

    # find contours in the thresholded image, then initialize
    # the list of contours that correspond to questions
    cnts = cv2.findContours(edged, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[1]
    questionCnts = []

    # loop over the contours
    for c in cnts:
        # compute the bounding box of the contour, then use the
        # bounding box to derive the aspect ratio
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        if y > image.shape[0] - 20:
            continue
        if x < 10:
            continue
        if 200 <= x <= 345:
            continue
        if 520 <= x <= 670:
            continue
        if len(c) < 5:
            continue
        if w < 20 or h < 20:
            continue
        if have_nearby_contours(questionCnts, c):
            continue

        # in order to label the contour as a question, region
        # should be sufficiently wide, sufficiently tall, and
        # have an aspect ratio approximately equal to 1
        questionCnts.append(c)
    cv2.drawContours(image, questionCnts, -1, (255, 255, 0), 2)
    # display_normal("With contours",image)
    return questionCnts


# todo: If we have less than 180 choice, then cluster them each related 4 together and neglect the remaining
def get_answers(image):
    questionCnts = find_question_contours(image.copy())
    if len(questionCnts) != 45 * 4:
        questionCnts = find_question_contours(image.copy(), True)

    if len(questionCnts) != 45 * 4:
        raise Exception("Didn't found all possible answers, only found {0}".format(len(questionCnts)))

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    answers = {}
    questionCnts, bounding_boxes = sort_contours(questionCnts, "top-to-bottom")
    # each question has 4 possible answers, to loop over the
    # question in batches of 4
    for (q, i) in enumerate(np.arange(0, len(questionCnts), 12)):
        # sort the contours for the current question from
        # left to right, then initialize the index of the
        # bubbled answer
        cnts = sort_contours(questionCnts[i:i + 12])[0]
        for i in np.arange(0, 12, 4):
            question = q + (i // 4) * 15 + 1
            single_q_cnts = cnts[i:i + 4]
            ratios = []
            # loop over the sorted contours
            for (j, c) in enumerate(single_q_cnts):
                # construct a mask that reveals only the current
                # "bubble" for the question
                mask = np.zeros(thresh.shape, dtype="uint8")
                cv2.drawContours(mask, [c], -1, 255, -1)
                # apply the mask to the thresholded image, then
                # count the number of non-zero pixels in the
                # bubble area
                mask = cv2.bitwise_and(thresh, thresh, mask=mask)

                area = cv2.contourArea(c)
                total = cv2.countNonZero(mask)
                ratio = total / area

                ratios.append(ratio)


            r1,r2 = sorted(ratios,reverse=True)[0:2]
            if (abs(r2-r1)) <0.25:

                answers[question]=0
            else:
                answers[question] = ratios.index(max(ratios))+1




    return answers


failed_images = 0
tried_files = 0
grades = {}
for f in pathlib.Path('test').iterdir():
    tried_files += 1
    try:
        image = cv2.imread(f.as_posix())
        image = optimize_image_rotation(image)
        image = crop_answer_section(image)
        answers = get_answers(image)
        grade = mark(answers)
        print ("#{0} : {1}   :   {2}".format(tried_files,f.name,grade))
        grades[f.name] = grade
    except Exception as e:
        grades[f.name] = 0
        print("File {0} failed with error message {1}".format(f.as_posix(), e))
        failed_images += 1

with open('submission.csv', 'w') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(["FileName", "Mark"])

    for key, value in grades.items():
        writer.writerow([key, value])

print("failed to correct {0} files ".format(failed_images))
