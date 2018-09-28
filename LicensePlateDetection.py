import math

import cv2
import numpy as np

from Utils import show_image


def rad_to_deg(angle):
    return angle * 180 / np.pi


def enhance(img):
    kernel = np.array([[-1, 0, 1], [-2, 0, 2], [1, 0, 1]])
    return cv2.filter2D(img, -1, kernel)


def rotate_and_resize(image, rotation_matrix, old_size, new_size):
    rotated_image = cv2.warpAffine(src=image, M=rotation_matrix, dsize=old_size)
    resized_image = cv2.getRectSubPix(rotated_image, new_size, (int(old_size[0] / 2), int(old_size[1] / 2)))
    return resized_image


class LicensePlateDetection:

    def is_valid_contour(self, contour):
        rect = cv2.minAreaRect(contour)

        if 45 < abs(rect[2]) < 135:  # rect is rotated ~90 degrees
            width = rect[1][1]
            height = rect[1][0]
        else:
            width = rect[1][0]
            height = rect[1][1]

        if self.plate_min_width < width < self.plate_max_width and self.plate_min_height < height < self.plate_max_height:
            aspect_ratio = float(width) / height
            if self.plate_aspect_ratio_range[0] < aspect_ratio < self.plate_aspect_ratio_range[1]:

                box = cv2.boxPoints(rect)
                box = np.int0(box)
                box_copy = list(box)
                point = box_copy[0]
                del (box_copy[0])
                distances_to_first_point = [((p[0] - point[0]) ** 2 + (p[1] - point[1]) ** 2) for p in box_copy]
                sorted_dists = sorted(distances_to_first_point)
                opposite_point = box_copy[distances_to_first_point.index(sorted_dists[1])]

                if abs(point[0] - opposite_point[0]) > 0:
                    contour_angle = abs(float(point[1] - opposite_point[1])) / abs(point[0] - opposite_point[0])
                    contour_angle = rad_to_deg(math.atan(contour_angle))

                    if contour_angle <= 30:
                        return True
        return False

    def process_image(self, debug_mode):
        result_plates = []
        working_image = np.copy(self.input_image)

        if debug_mode: show_image(working_image, "input_image")
        working_image = cv2.cvtColor(working_image, cv2.COLOR_BGR2GRAY)
        if debug_mode: show_image(working_image, "after cvtColor")
        working_image = enhance(working_image)
        if debug_mode: show_image(working_image, "after enhance")
        working_image = cv2.GaussianBlur(working_image, (5, 5), 0)
        if debug_mode: show_image(working_image, "after after gaussian blur")
        working_image = cv2.Sobel(working_image, -1, 1, 0)
        if debug_mode: show_image(working_image, "after sobel")
        _, working_image = cv2.threshold(working_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if debug_mode: show_image(working_image, "after threshold")
        se = cv2.getStructuringElement(cv2.MORPH_RECT, self.se_shape)
        working_image = cv2.morphologyEx(working_image, cv2.MORPH_CLOSE, se)
        if debug_mode: show_image(working_image, "after morphologyEx")

        _, contours, _ = cv2.findContours(working_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(working_image, contours, -1, (0, 0, 0), 1)
        if debug_mode: show_image(working_image, "after morphologyEx 2nd")
        _, contours, _ = cv2.findContours(working_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        tmp_img_for_contours = np.copy(self.input_image)
        cv2.drawContours(tmp_img_for_contours, contours, -1, (255, 0, 255), 2)
        if debug_mode: show_image(tmp_img_for_contours, "all contours")

        tmp_img_for_valid_contours = np.copy(self.input_image)
        valid_contour_counter = 0
        for contour in contours:

            highlighting_current_contour_image = np.copy(self.input_image)
            cv2.drawContours(highlighting_current_contour_image, contour, -1, (255, 0, 255), 2)
            if debug_mode: show_image(highlighting_current_contour_image, "currently processed contour")

            if self.is_valid_contour(contour):

                cv2.drawContours(tmp_img_for_valid_contours, contour, -1, (255, 0, 255), 2)

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)  # cast float to long

                cv2.drawContours(tmp_img_for_valid_contours, [box], 0, (255, 0, 0), 1)

                angle = rect[2]
                if angle < -45:
                    angle += 90

                valid_contour_counter += 1
                extend = cv2.contourArea(contour) / (rect[1][0] * rect[1][1])
                if extend > 0.5:
                    result_plates.append([box, rect[0][0], rect[0][1], angle])
        if debug_mode: show_image(tmp_img_for_valid_contours, "valid contours")
        return result_plates

    def find_best_plate(self, plates):
        if len(plates) < 1:
            return None
        sorted_plates = sorted(plates, key=lambda plate: abs(self.original_img_height * 0.66 - plate[2]))
        return sorted_plates[0][0]

    def __init__(self, image):
        self.original_img_height = image.shape[0]
        self.original_img_width = image.shape[1]

        self.input_image = image
        expected_plate_size = (self.original_img_width / 4.9, self.original_img_height / 12.8)
        self.plate_aspect_ratio_range = (2.2, 8)
        self.plate_min_width = expected_plate_size[0] / 2
        self.plate_max_width = expected_plate_size[0] * 2
        self.plate_min_height = expected_plate_size[1] / 2
        self.plate_max_height = expected_plate_size[1] * 2
        self.se_shape = (math.ceil(self.original_img_width / 30), math.ceil(self.original_img_width / 180))

    def detect_license_plate(self, debug_mode=False):
        potential_plates = self.process_image(debug_mode)
        return self.find_best_plate(potential_plates)
