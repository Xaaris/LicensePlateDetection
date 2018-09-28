import cv2


def show_image(img, desc="image"):
    cv2.imshow(desc, img)
    cv2.waitKey()


def get_image_patch(image, box):
    size = (box[1][0] - box[0][0], box[3][1] - box[0][1])
    center = (box[0][0] + size[0] / 2, box[0][1] + size[1] / 2)
    return cv2.getRectSubPix(image, size, center)


def save_debug_image(image, filename, folder=None):
    if folder:
        path = "../debugImages/" + folder + "/" + filename + ".png"
    else:
        path = "../debugImages/" + filename + ".png"
    cv2.imwrite(path, image)
