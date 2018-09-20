from moviepy.video.io.VideoFileClip import VideoFileClip

from src.CarDetection import *
from src.LicensePlateDetection import *


def save_debug_image_with_description(image, desc):
    cv2.imwrite("../debugImages/" + desc + ".png", image)


def get_image_patch(image, box):
    size = (box[1][0] - box[0][0], box[3][1] - box[0][1])
    center = (box[0][0] + size[0] / 2, box[0][1] + size[1] / 2)
    return cv2.getRectSubPix(image, size, center)


if __name__ == "__main__":

    path = "/Users/hannes/Desktop/IMG_2993.m4v"

    clip = VideoFileClip(path, audio=False).subclip(0, 3)
    frame_counter = 0
    car_counter = 0
    for frame in clip.iter_frames():
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        car_boxes = detect_vehicle(frame)
        frame_counter += 1
        print("Found " + str(len(car_boxes)) + " cars in frame " + str(frame_counter))
        for car_box in car_boxes:
            frame = cv2.rectangle(frame, car_box[0], car_box[3], (0, 0, 255), 3)
            car_image = get_image_patch(frame, car_box)
            car_counter += 1
            print("Detecting LP")
            license_plate_detection = LicensePlateDetection(car_image)
            plates = license_plate_detection.detect_license_plates()
            cv2.drawContours(frame, plates, 0, (127, 0, 255), 2, offset=car_box[0])
        save_debug_image_with_description(frame, "frame_" + str(frame_counter))
    clip.close()
