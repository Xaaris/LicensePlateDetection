import glob
import os

import cv2


def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append([x, y])


path = '../debugImages/cars/'
for filename in glob.glob(os.path.join(path, '*.png')):

    refPt = []
    image = cv2.imread(filename)
    clone = image.copy()
    cv2.namedWindow(filename)
    cv2.setMouseCallback(filename, click)

    while True:
        if len(refPt) > 1:
            for i in range(len(refPt) - 1):
                cv2.line(image, (refPt[i][0], refPt[i][1]), (refPt[i + 1][0], refPt[i + 1][1]), (0, 255, 0), 1)
        cv2.imshow(filename, image)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("r"):
            image = clone.copy()
            refPt = []

        elif key == ord("c"):
            break
    print("data.append(TestData(\"" + filename + "\", " + str(refPt) + "))")
    cv2.destroyAllWindows()
