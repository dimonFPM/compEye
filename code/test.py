# # import sqlite3 as sq
# #
# # try:
# #     db = sq.connect("qqq.db", timeout=5)
# #     cursor = db.cursor()
# #     cursor.execute('''CREATE TABLE student (id INTEGER,
# #                                             name TEXT,
# #                                             born_date DATE);''')
# #     cursor.close()
# # except sq.Error as error:
# #     print("Ошибка")
# #
# # if db:
# #     db.close()
# #     print("база данных закрыта")
#
#
# # import cv2 as cv
# #
# # a = []
# # for i in range(1001):
# #     cam = cv.VideoCapture(i)
# #     print(i, cam.isOpened())
# #     if cam.isOpened():
# #         check, frame = cam.read()
# #         if check:
# #             a.append(i)
# #             print(f"camera={i}")
# # cam.release()
# # print(*a)
#
#
# # import cv2 as cv
# # from sklearn.metrics.pairwise import euclidean_distances as edist
# #
# # count_red = 0
# # cam = cv.VideoCapture(700)
# # width = 800
# # cam.set(3, width)
# # cam.set(4, width // 1.777777)
# # cannyX, cannyY = 30, 150
# # if cam.isOpened():
# #     check, frame = cam.read()
# #     # print(frame.shape)
# #     if check:
# #         w = h = 200
# #         x, y = frame.shape[1] // 2 - w // 2, frame.shape[0] // 2 - h // 2
# #         win = (x, y, w, h)
# #         cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
# #         cv.imshow("start cadr", frame)
# #         cv.waitKey(0)
# #         cv.destroyAllWindows()
# #         roi = frame[y:y + h, x:x + w]
# #         roiGray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
# #         momRoi = cv.HuMoments(cv.moments(roiGray)).flatten()
# #         roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
# #         hist = cv.calcHist([roi], [0], None, [180], [0, 180])
# #         cr = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
# #     while True:
# #         check, frame = cam.read()
# #         if check:
# #             frameHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
# #             backProject = cv.calcBackProject([frameHSV], [0], hist, [0, 180], 1)
# #             ret, dst = cv.meanShift(backProject, win, cr)
# #             x, y, w, h = dst
# #             roiGray = frame[y:y + h, x:x + w]
# #             roiGray = cv.cvtColor(roiGray, cv.COLOR_BGR2GRAY)
# #             momRoi2 = cv.HuMoments(cv.moments(roiGray)).flatten()
# #
# #             change = edist([momRoi], [momRoi2])
# #             if change < 0.001:
# #                 cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)
# #                 cv.putText(frame, f"{change=}", (x, y - 10), cv.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0), thickness=1)
# #             else:
# #                 cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)
# #                 cv.putText(frame, f"{change=}", (x, y - 10), cv.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255), thickness=1)
# #                 count_red += 1
# #             print(f"{change=}")
# #             cv.imshow("", frame)
# #             if cv.waitKey(1) & 0xff == ord("q"):
# #                 break
# #         else:
# #             print("Кадры кончились")
# # else:
# #     print("видео не отрыто")
# # print(f"{count_red=}")
# # cam.release()
# # cv.destroyAllWindows()
#
#
# import cv2 as cv
# img = cv.imread("../sorce/picture/ab92b72bd2d64c4af46bd3972ebfd8d7.jpg")
#
# hog = cv.HOGDescriptor()
# hog.setSVMDetector(cv.HOGDescriptor_getDefaultPeopleDetector())
# ret, f = hog.detectMultiScale(img, winStride=(4, 4), padding=(8, 8), scale=1.25,
#                               useMeanshiftGrouping=False)
# for x, y, w, h in ret:
#     cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
# cv.imshow("", img)
#
# # surf = cv.xfeatures2d.SURF_create(400)
# # ret, rez = surf.detectAndCompute(img, None)
# # print(len(ret))
#
# cv.waitKey(0)


import matplotlib.pyplot as plt

x = [1, 2, 3, 4, 5]
y = [i ** 2 for i in x]
y2 = [i ** 3 for i in x]

# fig, ax = plt.subplots(1, 2, figsize=(2, 2))
#
#
# ax.title("название графика")
# ax[0].plot(x, y, "r--", label="график 1")
# ax[1].plot(x,y,label="график 3")
# ax[1].plot(x, y2, label="график 2")
# fig.legend()


# fig=plt.figure()
# plt.subplot(122)
# plt.title("график какой-то фигни")
# plt.plot(x,y,"r--",label="график 1")
# plt.legend()
# plt.subplot(121)
# plt.title("другой график")
# plt.plot(x,y2,label="график 2")
# plt.legend()


fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
ax1.plot(x, y, "r--")
ax2.plot(x, y2, )
plt.show()
