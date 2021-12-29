import cv2
import imutils
import numpy as np

bgSub = cv2.createBackgroundSubtractorMOG2()
cam = cv2.VideoCapture(700)

if not cam.isOpened():
    print("Ошибка")
    exit()

while True:
    check, img = cam.read()
    fg = bgSub.apply(img)
    img = cv2.bilateralFilter(img, 5, 25, 25)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_gray = cv2.adaptiveThreshold(img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 9, 2)
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, k, iterations=1)
    # img_gray = cv2.Canny(img_gray, 30, 50)

    # ret, img_gray = cv2.threshold(img_gray, 100, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    conture = cv2.findContours(img_gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    conture = imutils.grab_contours(conture)
    conture = sorted(conture, key=cv2.contourArea, reverse=True)
    cv2.drawContours(img, conture, -1, (0, 82, 8), thickness=1)
    fg = bgSub.apply(img)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)
    img[fg == 255] = [0, 0, 255]
    cv2.imshow("1", img)
    cv2.imshow("4", img_gray)
    # cv2.imshow("3",cv2.bitwise_and(img,img,mask=fg))
    cv2.imshow("2", fg)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# while True:
#     check, img = cam.read()
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     low_lavel = np.uint8([90, 70, 70])
#     high_lavel = np.uint8([150, 255, 255])
#     mask = cv2.inRange(img, low_lavel, high_lavel)
#     k = np.ones((2, 2))
#     mask = cv2.morphologyEx(mask, cv2.MORPH_HITMISS, k, iterations=2)
#     img[mask > 0] = [150, 255, 255]
#     img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
#     cv2.imshow("", img)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

cv2.destroyWindow()
cam.release()
