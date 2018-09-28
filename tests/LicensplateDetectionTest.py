import os
import unittest

import cv2
import numpy as np
from shapely.geometry import Polygon

from LicensePlateDetection import LicensePlateDetection


class LicensePlateDetectionTest(unittest.TestCase):

    def test_image_1(self):
        path = "tests/testImages/car1.png"
        input_file = self.load_image(path)
        plate = self.get_plate(input_file)
        plate_poly = Polygon(plate)
        expected_points = [[160, 204], [159, 188], [237, 182], [238, 196]]
        expected_poly = Polygon(expected_points)
        iou = self.calculate_iou(expected_poly, plate_poly)
        # self.show_result(expected_points, input_file, plate, iou)
        self.assertTrue(iou > 0.7)

    def test_image_2(self):
        path = "tests/testImages/car2.png"
        input_file = self.load_image(path)
        plate = self.get_plate(input_file)
        plate_poly = Polygon(plate)
        expected_points = [[208, 160], [208, 146], [275, 142], [275, 155]]
        expected_poly = Polygon(expected_points)
        iou = self.calculate_iou(expected_poly, plate_poly)
        # self.show_result(expected_points, input_file, plate, iou)
        self.assertTrue(iou > 0.7)

    def show_result(self, expected_points, input_file, plate, iou):
        cv2.drawContours(input_file, [np.array(expected_points)], -1, (0, 255, 0), 1)
        cv2.drawContours(input_file, [np.array(plate)], -1, (0, 0, 255), 1)
        cv2.imshow("IOU is " + str(iou), input_file)
        cv2.waitKey()

    def calculate_iou(self, expected_poly, plate_poly):
        intersection_area = expected_poly.intersection(plate_poly).area
        union_area = plate_poly.area + expected_poly.area - intersection_area
        iou = intersection_area / union_area
        return iou

    def get_plate(self, input_file):
        license_plate_detection = LicensePlateDetection(input_file)
        plate = [[p[0], p[1]] for p in license_plate_detection.detect_license_plate()]
        return plate

    def load_image(self, path):
        fullpath = os.path.abspath(path)
        input_file = cv2.imread(fullpath)
        return input_file


if __name__ == '__main__':
    unittest.main()
