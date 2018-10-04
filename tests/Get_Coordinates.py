import cv2

refPt = []


def click(event, x, y, flags, param):

    if event == cv2.EVENT_LBUTTONDOWN:
        refPt.append([x, y])


image = cv2.imread("testImages/difficult_light1.png")
clone = image.copy()
cv2.namedWindow("image")
cv2.setMouseCallback("image", click)

while True:
    if len(refPt) > 1:
        for i in range(len(refPt) - 1):
            cv2.line(image, (refPt[i][0], refPt[i][1]), (refPt[i+1][0], refPt[i+1][1]), (0, 255, 0), 1)
    cv2.imshow("image", image)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("r"):
        image = clone.copy()
        refPt = []

    elif key == ord("c"):
        break

print(refPt)

cv2.destroyAllWindows()
