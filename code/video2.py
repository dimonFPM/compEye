import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

cam = cv.VideoCapture(700)
# cam = cv.VideoCapture("../sorce/video/v1.mp4")
check, frame = cam.read()

w = h = 200
x, y = frame.shape[1] // 2 - h // 2, frame.shape[0] // 2 - w // 2
# x, y = frame.shape[1] // 2-270 - h // 2, frame.shape[0] // 2-20 - w // 2
track_win = (x, y, w, h)
roi = np.copy(frame[y:y + h, x:x + w])
# roi = cv.resize(roi, (roi.shape[0] * 5, roi.shape[1] * 5), interpolation=cv.INTER_CUBIC)
# roi = cv.GaussianBlur(roi, (3, 3), 121)
# roi = cv.bilateralFilter(roi, 5, 120, 120)
roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
mask = cv.inRange(roi, np.array((0, 0, 0)), np.array((180, 255, 255)))
hist = cv.calcHist([roi], [0], mask, [180], [0, 180])
cv.normalize(hist, hist, 0, 255, cv.NORM_MINMAX)
term_crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
cv.rectangle(frame, (x - 2, y - 2), (x + w + 1, y + h + 1), (255, 0, 0), thickness=2)
cv.imshow("frame", frame)
cv.waitKey(0)
while True:
    check, frame = cam.read()
    if check == True:
        hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        dst = cv.calcBackProject([frame], [0], hist, [0, 180], 1)
        ret, track_win = cv.meanShift(dst, track_win, term_crit)
        x, y, w, h = track_win
        # pts = cv.boxPoints(ret)
        # pts = np.int0(pts)
        # cv.polylines(frame, [pts], True, 255, 2)
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
        cv.imshow("frame", frame)
        if cv.waitKey(1) & 0xff == ord("q"):
            break
    else:
        break
cv.destroyWindow()
cam.release()
