import pathlib
import csv
from Constants import *
from FirstApproach import *
from SecondApproach import *

operation = SAMPLE
approach = SECOND_APPROACH

if operation == SAMPLE:
    if approach == FIRST_APPROACH:
        image = cv2.imread('sample.png')
        image = optimize_image_rotation(image)
        image = crop_answer_section(image)
        answers = get_answers(image)
        grade = mark(answers)
        print grade
    elif approach == SECOND_APPROACH:
        original_image = cv2.imread('sample.png')
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(original_image, (5, 5), 0)
        rotated_image = rotate_image(blurred)
        cropped_image = crop_image(rotated_image)
        ret, threshold_image = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        grade = get_grade(threshold_image)
        display_normal("O", original_image)
        display_normal("B", blurred)
        display_normal("R", rotated_image)
        display_normal("C", cropped_image)
        display_normal("T", threshold_image)
        print grade
elif operation == TRAIN or operation == TEST:
    failed_images = 0
    tried_files = 0
    grades = {}
    if operation == TRAIN:
        images_path = 'train'
    else:
        images_path = 'test'
    for f in pathlib.Path(images_path).iterdir():
        tried_files += 1
        try:
            grade = 0
            if approach == FIRST_APPROACH:
                image = cv2.imread(f.as_posix())
                image = optimize_image_rotation(image)
                image = crop_answer_section(image)
                answers = get_answers(image)
                grade = mark(answers)
            elif approach == SECOND_APPROACH:
                original_image = cv2.imread(f.as_posix())
                original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(original_image, (5, 5), 0)
                rotated_image = rotate_image(blurred)
                cropped_image = crop_image(rotated_image)
                ret, threshold_image = cv2.threshold(cropped_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                grade = get_grade(threshold_image)
            print ("#{0} : {1}   :   {2}".format(tried_files, f.name, grade))
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
