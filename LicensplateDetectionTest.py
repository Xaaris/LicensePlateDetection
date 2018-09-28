import cv2

from LicensePlateDetection import LicensePlateDetection


path = "/Users/hannes/PycharmProjects/LicensePlateDetection/debugImages/cars/frame_1car_1.png"
input_file = cv2.imread(path)
license_plate_detection = LicensePlateDetection(input_file)
plates = license_plate_detection.detect_license_plates(True)
cv2.drawContours(input_file, plates, -1, (127, 0, 255), 2)
cv2.imshow("Result", input_file)
cv2.waitKey()
