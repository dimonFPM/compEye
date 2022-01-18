import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tkinter as tk
import tkinter.messagebox as mb
from loguru import logger
from sklearn.metrics.pairwise import euclidean_distances as edist

# region logger
# logger.remove() #при включении не пишет логи в консоль
dbug = logger.add("log.log", level="DEBUG")

logger.remove(dbug)  # при включении не пишет логи в файл


# endregion

class Video:

    def __init__(self, name="", width=1280, height=720) -> None:
        self._cam = None
        self.videoName = None

    def videoConnect(self, name, width, height) -> None:

        '''Проверяет открыто ли видео и настраивает разрешение камеры, если происходит захват камеры.
        Получает на вход путь до видео, ширину и высоту кадра.
        При открытом видео сохраняет его в self.__cam, иначе сохраняет None в self.__cam '''

        self._camName = name
        self._camWidth = width
        self._camHeight = height
        cam = cv.VideoCapture(name)
        if not cam.isOpened():
            mb.showerror("Ошибка", "Видео невозможно прочитать")
            logger.debug("Видео невозможно прочитать,нет подключения к видео")
            self._cam = None
        else:
            logger.debug("Видео прочитано")
            self._cam = cam
            self._cam.set(3, self._camWidth)
            self._cam.set(4, self._camHeight)

    def findName(self, name) -> None:

        '''Вычленяе имя видео из пути до его файла, при пустом названии ничего не делает,
        если название int добавляет к нему слово "camera".
        На вход получает путь до файла.'''

        match name:
            case "":
                pass
            case n if type(n) == str:
                name = name.split("/")
                self.videoName = name[-1]
            case n if type(n) == int:
                self.videoName = "camera " + str(name)

    def viewNormalVideo(self, name="", width=1280, height=720) -> None:

        '''Выводит на экран видео без эфектов.
        На вход поступает путь файла, ширина и высота видео.'''

        self.videoConnect(name, width, height)
        if self._cam:
            self.findName(name)
            while True:
                check, frame = self._cam.read()
                if check:
                    cv.imshow(f"normalVideo ({self.videoName})", frame)
                    if cv.waitKey(10) & 0xff == ord("q"):
                        break
                else:
                    logger.debug("кадры кончились")
                    break


class Tracking(Video):
    def __init__(self, name="", width=1280, height=720) -> None:
        super().__init__(name, width, height)
        self.videoConnect(name, width, height)
        if self._cam != None:
            self.__countRedLabel = 0
            self.__countRedLabelLst = []
            check, frame = self._cam.read()
            if check:
                h = w = 200
                x, y = frame.shape[1] // 2 - w // 2, frame.shape[0] // 2 - h // 2
                self.__window = (x, y, w, h)
                roi = frame[y:y + h, x:x + w]
                self.__startMoment = cv.HuMoments(cv.moments(cv.cvtColor(roi, cv.COLOR_BGR2GRAY))).flatten()
                roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
                self.__startHistRoi = cv.calcHist([roi], [0], None, [180], [0, 180])
                self.__startHistRoi = cv.normalize(self.__startHistRoi, self.__startHistRoi)
                self.crit = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), thickness=2)

                cv.imshow("start_frame", frame)
                cv.waitKey(0)
                cv.destroyAllWindows()
            else:
                print("Нет кадров")
                logger.debug("Нет кадров")
        else:
            print("нет видео")

    def track(self):
        check, frame = self._cam.read()
        if check:
            # x, y, w, h = self.__window
            while True:
                check, frame = self._cam.read()
                frameHSV = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
                backProject = cv.calcBackProject([frameHSV], [0], self.__startHistRoi, [0, 180], 1)
                ret, cord = cv.meanShift(backProject, self.__window, self.crit)
                x, y, w, h = cord
                roi = frame[y:y + h, x:x + w]
                moment = cv.HuMoments(cv.moments(cv.cvtColor(roi, cv.COLOR_BGR2GRAY))).flatten()
                change = edist([self.__startMoment], [moment])
                if change[0, 0] > 0.001:
                    cv.putText(frame, f"change={change[0, 0]}", (x, y - 3), cv.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 255),
                               thickness=1)
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    self.__countRedLabel += 1
                    self.__countRedLabelLst.append(change)
                else:
                    cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv.putText(frame, f"change={change[0, 0]}", (x, y - 3), cv.FONT_HERSHEY_PLAIN, 0.7, (0, 255, 0),
                               thickness=1)
                cv.imshow("track", frame)
                if cv.waitKey(1) & 0xff == ord("q"):
                    break

        else:
            print("нет кадров")
        print(f"redLabel={self.__countRedLabel}")
        print(f"max change={max(self.__countRedLabelLst)}")
        print(f"min change={min(self.__countRedLabelLst)}")

    def __del__(self):
        if self._cam != None:
            self._cam.release()


class findRoad(Video):
    def __init__(self, name="", width=1280, height=720):
        super().__init__(name, width, height)
        self.videoConnect("../sorce/video/v1.mp4", 1280, 720)
        if self._cam:
            self.__poly = np.array([[200, 650], [1100, 650], [800, 300], [400, 300]])
            check, frame = self._cam.read()
            self.__mask = np.zeros_like(frame)
            self.__mask = cv.cvtColor(self.__mask, cv.COLOR_BGR2GRAY)
            cv.fillPoly(self.__mask, [self.__poly], 255)

    def viewRoad(self, name="", width=1280, height=720):
        self.videoConnect(name, width, height)
        if self._cam:
            self.findName(name)

            while True:
                check, frame = self._cam.read()
                if check:
                    frame1 = frame.copy()
                    # frame1=frame1[0:435,:,:]
                    frame1 = cv.bilateralFilter(frame1, 5, 19, 19)
                    frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
                    frame1 = cv.bitwise_and(frame1, frame1, mask=self.__mask)
                    th = cv.adaptiveThreshold(frame1, 100, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, 5, 3)
                    # ret, th = cv.threshold(frame1, 200, 255, cv.THRESH_TOZERO_INV)
                    # ret,conture
                    cv.imshow(f"normalVideo ({self.videoName})", th)
                    # plt.imshow(frame1)
                    # plt.show()
                    if cv.waitKey(1) & 0xff == ord("q"):
                        break
                else:
                    logger.debug("кадры кончились")
                    break


if __name__ == '__main__':
    a = Tracking(700)
    a.track()
    # a.viewRoad("../sorce/video/v1.mp4")
    # a.viewRoad("../sorce/video/автобан.mp4")
    # a.viewNormalVideo()
    # a.viewNormalVideo(700)
    # a.createMask()
    del a
    cv.destroyAllWindows()
    exit("конец программы")
