import os
from collections import namedtuple

import cv2
import numpy as np
from shapely.geometry import Polygon

from LicensePlateDetection import LicensePlateDetection


def load_image(path):
    fullpath = os.path.abspath(path)
    input_file = cv2.imread(fullpath)
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


def show_result(expected_points, input_file, plate, iou):
    cv2.drawContours(input_file, [np.array(expected_points)], -1, (0, 255, 0), 1)
    cv2.drawContours(input_file, [np.array(plate)], -1, (0, 0, 255), 1)
    cv2.imshow("IOU is " + str(iou), input_file)
    cv2.waitKey()


def get_test_data():
    data = []
    TestData = namedtuple('TestData', ['image_path', 'expected_plate_pos'])
    data.append(TestData("tests/testImages/standard1.png", [[160, 204], [159, 188], [237, 182], [238, 196]]))
    data.append(TestData("tests/testImages/angle1.png", [[208, 160], [208, 146], [275, 142], [275, 155]]))
    data.append(TestData("tests/testImages/bottom_edge1.png", [[240, 172], [241, 188], [313, 170], [313, 185]]))
    data.append(TestData("tests/testImages/standard2.png", [[341, 183], [341, 198], [416, 180], [416, 195]]))
    data.append(TestData("tests/testImages/small1.png", [[124, 121], [124, 131], [172, 128], [171, 118]]))
    data.append(TestData("tests/testImages/big1.png", [[472, 341], [474, 371], [619, 355], [618, 326]]))
    return data


def test_images():
    total_iou = 0
    for test_data in get_test_data():
        input_file = load_image(test_data.image_path)
        plate = get_plate(input_file)
        if plate is not None:
            plate_poly = Polygon(plate)
            expected_poly = Polygon(test_data.expected_plate_pos)
            iou = calculate_iou(expected_poly, plate_poly)
            # show_result(test_data.expected_plate_pos, input_file, plate, iou)
            total_iou += iou
            print(test_data.image_path + " IOU: " + str(iou))
        else:
            print(test_data.image_path + " IOU: Failed to locate plate")
    print("Average IOU was: " + str(total_iou / len(get_test_data())))


if __name__ == '__main__':
    test_images()
