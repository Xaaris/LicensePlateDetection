import math

import cv2
import numpy as np


def save_debug_image_with_description(image, desc):
    cv2.imwrite("debugImages/" + desc + ".png", image)


def contour_fills_most_of_image_patch(image):
    _, image_patch = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    white_pixels = 0
    width = image_patch.shape[0]
    height = image_patch.shape[1]

    for x in range(width):
        for y in range(height):
            if image_patch[x][y] == 255:
                white_pixels += 1

    return float(white_pixels) / (width * height) > 0.5


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
        # high_contrast_image = enhance(gray_scale_image)
        # save_debug_image_with_description(high_contrast_image, "03 - after enhance")
        blurred_image = cv2.GaussianBlur(gray_scale_image, (5, 5), 0)
        save_debug_image_with_description(blurred_image, "04 - after gaussian blur")
        sobel_edges_detected_image = cv2.Sobel(blurred_image, -1, 1, 0)
        save_debug_image_with_description(sobel_edges_detected_image, "05 - after sobel")
        _, threshholded_image = cv2.threshold(sobel_edges_detected_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        save_debug_image_with_description(threshholded_image, "06 - after threshold")
        se = cv2.getStructuringElement(cv2.MORPH_RECT, self.se_shape)
        morphed_image = cv2.morphologyEx(threshholded_image, cv2.MORPH_CLOSE, se)
        save_debug_image_with_description(morphed_image, "07 - after morphologyEx")

        raw_image = np.copy(morphed_image)

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
                new_size = (int(rect[1][0]), int(rect[1][1]))
                if angle < -45:
                    new_size = (int(rect[1][1]), int(rect[1][0]))
                    angle += 90

                x_list = [i[0] for i in box]
                y_list = [i[1] for i in box]
                max_box_dimensions = (max(x_list) - min(x_list), max(y_list) - min(y_list))
                global_center = (min(x_list) + max_box_dimensions[0] / 2, min(y_list) + max_box_dimensions[1] / 2)
                box_center = (int(max_box_dimensions[0] / 2), int(max_box_dimensions[1] / 2))
                patch_rotation_matrix = cv2.getRotationMatrix2D(box_center, angle, 1.0)
                boxed_contour_image = cv2.getRectSubPix(raw_image, max_box_dimensions, global_center)
                cropped_contour_image = rotate_and_resize(boxed_contour_image, patch_rotation_matrix,
                                                          max_box_dimensions, new_size)

                valid_contour_counter += 1
                image_patch_with_contour_and_box = cv2.getRectSubPix(tmp_img_for_valid_contours, max_box_dimensions,
                                                                     global_center)
                save_debug_image_with_description(boxed_contour_image, "contour_" + str(valid_contour_counter))
                save_debug_image_with_description(image_patch_with_contour_and_box,
                                                  "contour_and_box_" + str(valid_contour_counter))
                save_debug_image_with_description(cropped_contour_image,
                                                  "cropped_contour_" + str(valid_contour_counter))

                if contour_fills_most_of_image_patch(cropped_contour_image):
                    result_plates.append(box)
        return result_plates

    def __init__(self, image):
        height = image.shape[0]
        width = image.shape[1]

        self.input_image = image
        expected_plate_size = (width / 4.9, height / 12.8)
        self.plate_aspect_ratio_range = (2.2, 8)
        self.plate_min_width = expected_plate_size[0] / 2
        self.plate_max_width = expected_plate_size[0] * 2
        self.plate_min_height = expected_plate_size[1] / 2
        self.plate_max_height = expected_plate_size[1] * 2
        self.se_shape = (math.ceil(width / 30), math.ceil(width / 180))

    def detect_license_plates(self):
        return self.process_image()
