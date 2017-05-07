import cv2
import math
import Constants


def display_normal(title, img):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, img)
    cv2.waitKey(0)


def rotate_image(image_to_rotate):
    blur = cv2.GaussianBlur(image_to_rotate, (5, 5), 0)
    edge = cv2.Canny(blur, 75, 150)

    # find circles in the edge map, then find two almost on the same line
    dp_values = [2, 3, 4]
    for dp in dp_values:
        circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, dp, 700, minRadius=35, maxRadius=45)
        if circles is None:
            continue
        else:
            circles = circles[0]
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
        return cv2.warpAffine(image_to_rotate, rotation_matrix, image_to_rotate.T.shape)
    return None


def crop_image(image_to_crop):
    template_circle = cv2.imread('circle.png')
    template_circle = cv2.cvtColor(template_circle, cv2.COLOR_BGR2GRAY)
    # Apply template Matching
    matched_circles = []
    res = cv2.matchTemplate(image_to_crop, template_circle, cv2.TM_CCOEFF_NORMED)
    while len(matched_circles) < 2:
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        matched_circles.append(max_loc)
        for i in range(max_loc[1] - template_circle.shape[0], max_loc[1] + template_circle.shape[0]):
            for j in range(max_loc[0] - template_circle.shape[1], max_loc[0] + template_circle.shape[1]):
                res[i, j] = 0

    if matched_circles[0][0] < matched_circles[1][0]:
        top_left_cropped = (matched_circles[0][0] + template_circle.shape[1],
                            matched_circles[0][1] + template_circle.shape[0])
        bottom_right_cropped = matched_circles[1]
    else:
        top_left_cropped = (matched_circles[1][0] + template_circle.shape[1],
                            matched_circles[1][1] + template_circle.shape[0])
        bottom_right_cropped = matched_circles[0]

    cropped = image_to_crop[top_left_cropped[1]:bottom_right_cropped[1], top_left_cropped[0]:bottom_right_cropped[0]]
    return cropped


def get_questions_list(all_questions):
    questions = []
    x_left = 6
    x_right = x_left + 153
    y_top = 10
    y_bottom = 40
    for i in range(15):
        q = all_questions[y_top:y_bottom, x_left:x_right]
        questions.append(q)
        # display_normal("Q. {0}".format(i + 1), q)
        y_top += 41
        y_bottom += 41
    x_left += 330
    x_right = x_left + 153
    y_top = 10
    y_bottom = 40
    for i in range(15):
        q = all_questions[y_top:y_bottom, x_left:x_right]
        questions.append(q)
        # display_normal("Q. {0}".format(i + 16), q)
        y_top += 41
        y_bottom += 41
    x_left += 330
    x_right = x_left + 153
    y_top = 10
    y_bottom = 40
    for i in range(15):
        q = all_questions[y_top:y_bottom, x_left:x_right]
        questions.append(q)
        # display_normal("Q. {0}".format(i + 31), q)
        y_top += 41
        y_bottom += 41
    return questions


def number_of_dark_pixels(image_to_check):
    dark_pixels = 0
    for row in range(image_to_check.shape[0]):
        for col in range(image_to_check.shape[1]):
            if image_to_check[row, col] == 0:
                dark_pixels += 1
    return dark_pixels


def mark_question(question_no, question_img):
    correct_answer = Constants.ANSWER_KEY[question_no]
    # display_normal("Q.{0}".format(question_no), question_img)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    question_img = cv2.morphologyEx(question_img, cv2.MORPH_CLOSE, kernel)
    # display_normal("Q.{0} after closing".format(question_no), question_img)
    rows, cols = question_img.shape
    choice1 = question_img[:, 0:cols / 4]
    choice2 = question_img[:, cols / 4:cols / 2]
    choice3 = question_img[:, cols / 2:(cols * 3) / 4]
    choice4 = question_img[:, (cols * 3) / 4:cols]
    choices = [choice1, choice2, choice3, choice4]
    chosen = [0, 0, 0, 0]
    darkness = [number_of_dark_pixels(c) for c in choices]
    avg_darkness = sum(darkness) / 4
    for c in range(4):
        if darkness[c] > avg_darkness:
            chosen[c] = 1
    if not sum(chosen) == 1:
        return 0
    chosen_answer = chosen.index(1) + 1
    if chosen_answer == correct_answer:
        return 1
    else:
        return 0


def get_grade(image_to_grade):
    questions = get_questions_list(image_to_grade)
    total_mark = 0
    q_no = 1
    for question in questions:
        q_mark = mark_question(q_no, question)
        total_mark += q_mark
        q_no += 1
        # print q_no - 1, ":", q_mark
    return total_mark
