import math

import cv2
import numpy as np


def save_debug_image_with_description(image, desc):
    cv2.imwrite("debugImages/" + desc + ".png", image)


def deg_to_rad(angle):
    return angle * np.pi / 180.0


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

    def process_image(self):
        result_plates = []
        input_image_copy = np.copy(self.input_image)

        save_debug_image_with_description(self.input_image, "01 - input_image")
        gray_scale_image = cv2.cvtColor(self.input_image, cv2.COLOR_BGR2GRAY)
        save_debug_image_with_description(gray_scale_image, "02 - after cvtColor")
        high_contrast_image = enhance(gray_scale_image)
        save_debug_image_with_description(high_contrast_image, "03 - after enhance")
        blurred_image = cv2.GaussianBlur(high_contrast_image, (5, 5), 0)
        save_debug_image_with_description(blurred_image, "04 - after gaussian blur")
        sobel_edges_detected_image = cv2.Sobel(blurred_image, -1, 1, 0)
        save_debug_image_with_description(sobel_edges_detected_image, "05 - after sobel")
        _, threshholded_image = cv2.threshold(sobel_edges_detected_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        save_debug_image_with_description(threshholded_image, "06 - after threshold")
        se = cv2.getStructuringElement(cv2.MORPH_RECT, self.se_shape)
        morphed_image = cv2.morphologyEx(threshholded_image, cv2.MORPH_CLOSE, se)
        save_debug_image_with_description(morphed_image, "07 - after morphologyEx")

        _, contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(morphed_image, contours, -1, (0, 0, 0), 1)
        save_debug_image_with_description(morphed_image, "07b - after morphologyEx")
        _, contours, _ = cv2.findContours(morphed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        tmp_img_for_contours = np.copy(input_image_copy)
        cv2.drawContours(tmp_img_for_contours, contours, -1, (255, 0, 255), 2)
        save_debug_image_with_description(tmp_img_for_contours, "08 - all contours")

        tmp_img_for_valid_contours = np.copy(input_image_copy)
        valid_contour_counter = 0
        for contour in contours:

            highlighting_current_contour_image = np.copy(input_image_copy)
            cv2.drawContours(highlighting_current_contour_image, contour, -1, (255, 0, 255), 2)
            save_debug_image_with_description(highlighting_current_contour_image, "currently processed contour")

            if self.is_valid_contour(contour):

                cv2.drawContours(tmp_img_for_valid_contours, contour, -1, (255, 0, 255), 2)

                rect = cv2.minAreaRect(contour)
                box = cv2.boxPoints(rect)
                box = np.int0(box)  # cast float to long

                cv2.drawContours(tmp_img_for_valid_contours, [box], 0, (255, 0, 0), 1)
                save_debug_image_with_description(tmp_img_for_valid_contours, "08 - valid contours")

                angle = rect[2]
                if angle < -45:
                    angle += 90

                valid_contour_counter += 1
                extend = cv2.contourArea(contour) / (rect[1][0] * rect[1][1])
                if extend > 0.5:
                    result_plates.append([box, rect[0][0], rect[0][1], angle])
        return result_plates

    def find_best_plate(self, plates):
        if len(plates) < 1:
            return []
        sorted_plates = sorted(plates, key=lambda plate: abs(self.original_img_height * 0.66 - plate[2]))
        return [sorted_plates[0][0]]

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

    def detect_license_plates(self):
        potential_plates = self.process_image()
        return self.find_best_plate(potential_plates)
