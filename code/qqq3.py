# import matplotlib.pyplot as plt
# import numpy as np
# import cv2
#
# img = cv2.imread("../sorce/picture/cow).jpg")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow("", img_gray)
#
# cv2.waitKey(0)
#
# high = 255
# shag = 15
#
# while True:
#     print(high)
#     low = high - shag
#     mask_gray = cv2.inRange(img_gray, low, high)
#     # print(mask_gray)
#     img_gray[mask_gray > 0] = high
#     high -= 20
#     if high <= 0:
#         break
# # print(mask_gray>0)
#
# cv2.imshow('', img_gray)
# cv2.waitKey(0)
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np


def f1(name_img: str) -> None:
    imgg = cv2.imread(f"{name_img}")
    imgg = cv2.resize(imgg, (500, 500))
    cv2.imshow("", imgg)
    cv2.waitKey(0)

    img = cv2.cvtColor(imgg, cv2.COLOR_BGR2HSV)
    cv2.imshow("", img)
    cv2.waitKey(0)
    # img = cv2.Canny(img, 50, 90, True)
    # print(img.shape)
    low_green = np.array([225, 3, 3])
    high_green = np.array([110, 32, 32])
    mask = cv2.inRange(img, low_green, high_green)
    img[mask > 0] = ([75, 225, 240])
    cv2.imshow("", img)
    cv2.waitKey(0)
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    img = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
    r, img = cv2.threshold(img, 40, 255, 0)

    #
    # img = cv2.Canny(img, 50, 90, True)
    # k=np.ones((2,2))
    # img=cv2.dilate(img,k,iterations=2)
    cv2.imshow("", img)
    cv2.waitKey(0)

    # k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # img = cv2.morphologyEx(img,cv2.MORPH_HITMISS,k)
    # cv2.imshow("", img)
    # cv2.waitKey(0)

    conture = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    conture = imutils.grab_contours(conture)
    conture = sorted(conture, key=cv2.contourArea, reverse=True)
    print(f"{len(conture)=}")
    conture = conture[0:1]
    cv2.drawContours(imgg, conture[0], -1, (0, 0, 255), thickness=3)
    print(f"{conture[0][0][0][0]=}")
    x, y = conture[0][0][0][0], conture[0][0][0][1]
    print(x, y)
    cv2.putText(imgg, "Listik", (x, y), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), thickness=4)
    cv2.imshow("", imgg)
    cv2.waitKey(0)


f1("../sorce/picture/list.jpeg")
