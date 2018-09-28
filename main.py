import time

from moviepy.video.io.VideoFileClip import VideoFileClip

from CarDetection import *
from LicensePlateDetection import *


def save_debug_image_with_description(image, filename, folder=None):
    if folder:
        path = "../debugImages/" + folder + "/" + filename + ".png"
    else:
        path = "../debugImages/" + filename + ".png"
    cv2.imwrite(path, image)


def get_image_patch(image, box):
    size = (box[1][0] - box[0][0], box[3][1] - box[0][1])
    center = (box[0][0] + size[0] / 2, box[0][1] + size[1] / 2)
    return cv2.getRectSubPix(image, size, center)


if __name__ == "__main__":
    start = time.time()
    input_file = "/Users/hannes/PycharmProjects/LicensePlateDetection/testFiles/IMG_2993.m4v"

    clip = VideoFileClip(input_file, audio=False).subclip(0, 3)
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
            save_debug_image_with_description(car_image, "frame_" + str(frame_counter) + "car_" + str(car_counter), "cars")
            license_plate_detection = LicensePlateDetection(car_image)
            plates = license_plate_detection.detect_license_plates()
            cv2.drawContours(frame_copy, plates, -1, (127, 0, 255), 2, offset=car_box[0])
        save_debug_image_with_description(frame_copy, "frame_" + str(frame_counter), "processed_frames")
    clip.close()

    print("It took " + str(time.time() - start) + " seconds")
