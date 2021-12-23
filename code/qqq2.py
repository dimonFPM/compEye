import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils

k = np.ones((2, 2))
imgg = cv2.imread("../sorce/picture/auto2.webp")
# imgg = cv2.imread("../sorce/picture/auto1.jpg")

img = cv2.cvtColor(imgg, cv2.COLOR_BGR2GRAY)
img = cv2.bilateralFilter(img, 5, 20, 20)
img = cv2.Canny(img, 50, 90)
# img = cv2.dilate(img, k, iterations=1)
# img = cv2.erode(img, k, iterations=1)

plt.imshow(imgg)
plt.show()
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGBA))
plt.show()
# cv2.imshow("", img)
cv2.waitKey(0)

conturs = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
conturs = imutils.grab_contours(conturs)
conturs = sorted(conturs, key=cv2.contourArea, reverse=True)

p = []
for i in conturs:
    approx = cv2.approxPolyDP(i, 5, True)
    if len(approx) == 4:
        p.append(approx)
        break

mask = np.zeros(img.shape[:2], dtype="uint8")
for i in p:
    cv2.drawContours(mask, [i], -1, 255, thickness=cv2.FILLED)
a = cv2.bitwise_and(imgg, imgg, mask=mask)
cv2.imshow("", a)
cv2.waitKey(0)
