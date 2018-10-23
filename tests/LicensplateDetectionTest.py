import math
import os
import time
from collections import namedtuple

import cv2
import numpy as np
from shapely.geometry import Polygon

from LicensePlateDetection import LicensePlateDetection


def load_image(path):
    full_path = os.path.abspath("tests/testImages/" + path)
    input_file = cv2.imread(full_path)
    return input_file


def get_plate(input_file):
    license_plate_detection = LicensePlateDetection(input_file)
    potential_plates = license_plate_detection.detect_license_plate()
    if potential_plates is not None:
        plate = [[p[0], p[1]] for p in potential_plates]
    else:
        plate = None
    return plate


def calculate_iou(expected_poly, plate_poly):
    intersection_area = expected_poly.intersection(plate_poly).area
    union_area = plate_poly.area + expected_poly.area - intersection_area
    iou = intersection_area / union_area
    return iou


def show_result(expected_points, input_file, plate, name, iou):
    cv2.drawContours(input_file, [np.array(expected_points)], -1, (0, 255, 0), 1)
    if plate is not None: cv2.drawContours(input_file, [np.array(plate)], -1, (0, 0, 255), 1)
    cv2.imshow(name + " IOU: " + str(iou), input_file)
    cv2.waitKey()


def get_test_data():
    data = []
    TestData = namedtuple('TestData', ['image_path', 'expected_plate_pos'])
    data.append(TestData("angle1.png", [[208, 160], [208, 146], [275, 142], [275, 155]]))
    data.append(TestData("angle2.png", [[223, 157], [222, 171], [292, 167], [291, 153]]))
    data.append(TestData("standard1.png", [[160, 204], [159, 188], [237, 182], [238, 196]]))
    data.append(TestData("standard2.png", [[341, 183], [341, 198], [416, 195], [416, 180]]))
    data.append(TestData("standard3.png", [[169, 201], [169, 218], [250, 209], [249, 193]]))
    data.append(TestData("standard4.png", [[194, 214], [196, 232], [280, 222], [278, 206]]))
    data.append(TestData("bottom_edge1.png", [[240, 172], [241, 188], [313, 185], [313, 170]]))
    data.append(TestData("small1.png", [[124, 121], [124, 131], [172, 128], [171, 118]]))
    data.append(TestData("big1.png", [[472, 341], [474, 371], [619, 355], [618, 326]]))
    data.append(TestData("big2.png", [[88, 350], [267, 356], [270, 394], [90, 386]]))
    data.append(TestData("difficult_front1.png", [[94, 364], [277, 371], [279, 409], [96, 401]]))
    data.append(TestData("difficult_light1.png", [[49, 355], [224, 360], [227, 396], [51, 390]]))
    data.append(TestData("lot_of_environment.png", [[432, 290], [433, 313], [549, 301], [548, 277]]))
    return data


def test_images():
    total_iou = 0
    for test_data in get_test_data():
        iou = 0
        input_file = load_image(test_data.image_path)
        plate = get_plate(input_file)
        if plate is not None:
            plate_poly = Polygon(plate)
            expected_poly = Polygon(test_data.expected_plate_pos)
            iou = calculate_iou(expected_poly, plate_poly)
            total_iou += iou
            print(test_data.image_path + " IOU: " + str(iou))
        else:
            print(test_data.image_path + " IOU: Failed to locate plate")
        # show_result(test_data.expected_plate_pos, input_file, plate, test_data.image_path, iou)
    print("Average IOU was: " + str(total_iou / len(get_test_data())))


def grid_search():
    extend = np.linspace(0.3, 0.5, 5)
    aspect_ratio_min = np.linspace(1.5, 3.5, 5)
    aspect_ratio_max = np.linspace(6.5, 12, 5)
    se_x_factor = np.linspace(30, 40, 5)
    se_y_factor = np.linspace(210, 240, 5)
    morph_opening_size = (3,)
    max_angle = np.linspace(15, 30, 4)

    number_of_iterations = extend.size * aspect_ratio_min.size * aspect_ratio_max.size * se_x_factor.size * se_y_factor.size * len(morph_opening_size) * max_angle.size
    one_percent_ops = math.ceil(number_of_iterations / 100)
    ops_counter = 0
    percentage_done = 0

    file = open("tests/grid_search_results.csv", "w")
    file.write("IOU; extend; aspect_min; aspect_max; se_x; se_y; opening_size; max_angle \n")
    for ext in extend:
        for aspect_min in aspect_ratio_min:
            for aspect_max in aspect_ratio_max:
                for se_x in se_x_factor:
                    for se_y in se_y_factor:
                        for opening_size in morph_opening_size:
                            for angle in max_angle:
                                total_iou = 0

                                ops_counter += 1
                                if ops_counter % one_percent_ops == 0:
                                    percentage_done += 1
                                    print("Completed " + str(percentage_done) + "%")

                                for test_data in get_test_data():
                                    input_file = load_image(test_data.image_path)

                                    lpd = LicensePlateDetection(input_file, (aspect_min, aspect_max), (se_x, se_y), (opening_size, opening_size), ext, angle)
                                    potential_plates = lpd.detect_license_plate()
                                    if potential_plates is not None:
                                        plate = [[p[0], p[1]] for p in potential_plates]
                                        plate_poly = Polygon(plate)
                                        expected_poly = Polygon(test_data.expected_plate_pos)
                                        iou = calculate_iou(expected_poly, plate_poly)
                                        total_iou += iou
                                avg_iou = total_iou / len(get_test_data())
                                file.write("{:5.5f}".format(avg_iou) + "; " +
                                           "{:5.3f}".format(ext) + "; " +
                                           "{:5.3f}".format(aspect_min) + "; " +
                                           "{:5.3f}".format(aspect_max) + "; " +
                                           "{:5.3f}".format(se_x) + "; " +
                                           "{:5.3f}".format(se_y) + "; " +
                                           "{:5.3f}".format(opening_size) + "; " +
                                           "{:5.3f}".format(angle) + "\n")
    file.close()


if __name__ == '__main__':
    start = time.time()

    test_images()
    # grid_search()

    print("It took " + str(time.time() - start) + " seconds")
