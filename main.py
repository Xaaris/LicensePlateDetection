import os
import time

import cv2
import numpy as np
from moviepy.video.io.VideoFileClip import VideoFileClip

from CarDetection import detect_vehicle
from LicensePlateDetection import LicensePlateDetection
from Utils import get_image_patch, save_debug_image

if __name__ == "__main__":
    start = time.time()
    fullpath = os.path.abspath("testFiles/IMG_2993.m4v")

    clip = VideoFileClip(fullpath, audio=False).subclip(0, 3)
    frame_counter = 0
    for frame in clip.iter_frames():
        frame_counter += 1
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_copy = np.copy(frame)
        car_boxes = detect_vehicle(frame)
        print("Found " + str(len(car_boxes)) + " cars in frame " + str(frame_counter))
        car_counter = 0
        for car_box in car_boxes:
            car_counter += 1
            frame_copy = cv2.rectangle(frame_copy, car_box[0], car_box[3], (0, 0, 255), 3)
            car_image = get_image_patch(frame, car_box)
            print("Detecting LP")
            save_debug_image(car_image, "frame_" + str(frame_counter) + "car_" + str(car_counter), "cars")
            license_plate_detection = LicensePlateDetection(car_image)
            plate = license_plate_detection.detect_license_plate()
            if plate is not None: cv2.drawContours(frame_copy, [plate], -1, (127, 0, 255), 2, offset=car_box[0])
        save_debug_image(frame_copy, "frame_" + str(frame_counter), "processed_frames")
    clip.close()

    print("It took " + str(time.time() - start) + " seconds")
