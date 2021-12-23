import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np
import find_coine


class QQQ4:
    img_norm = None

    @staticmethod
    def show_image(img, metod=0, name="") -> None:
        if metod == 0:
            cv2.imshow(name, img)
            cv2.waitKey(0)
        elif metod == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fig = plt.figure(name)
            plt.imshow(img)

            # print(img[131][545])
            plt.show()

    @staticmethod
    def test() -> None:
        a = np.array([[[0, 0, 0]]], dtype='uint8')
        a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
        print(a[131][545])

    def f(self) -> None:
        img = self.img_norm.copy()
        img = img[:150]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        self.show_image(img)
        img = img[:, :, 0]
        img = img.ravel()
        img = list(img)
        # print(img)
        l = len(img)
        print(f"{sum(img)/l=}")
        # print(img)

    def find_nomer(self):
        img = self.img_norm.copy()
        b, g, r = cv2.split(img)
        self.show_image(r)
        # img = cv2.bilateralFilter(img, 20, 45, 45)
        # self.show_image(img, name="filter")

        print(img[0][0])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        print(img[0][0])

        # img=img[:,:,0]
        self.show_image(img)
        low_color = np.array([102, 53, 100])
        high_color = np.array([102, 53, 255])
        mask = cv2.inRange(img, low_color, high_color)
        # print(list(mask))
        img[mask > 0] = ([120, 53, 233])
        img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        self.show_image(mask)
        self.show_image(img)
        #

        # # img=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        # low_color = np.array([0, 0, 0])
        # high_color = np.array([176, 176, 176])
        # mask = cv2.inRange(img, low_color, high_color)
        # img[mask > 0] = ([0, 0, 0])
        # # print(list(mask))
        # print(list(mask[:200]))
        # # img=cv2.cvtColor(img,cv2.COLOR_HSV2BGR)
        # self.show_image(mask)
        # self.show_image(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        r, img = cv2.threshold(img, 130, 255, 0)
        # r, img = cv2.threshold(img, 130, 255, cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        # img=cv2.Canny(img,50,90)
        # print(img)
        self.show_image(img, 1, "Threshold")

        # k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # img = cv2.erode(img, k, iterations=1)
        # self.show_image(img, 1, "Threshold1")
        # img = cv2.dilate(img, k, iterations=1)
        # self.show_image(img, 1, "Threshold2")

        conture = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        conture = imutils.grab_contours(conture)
        conture = sorted(conture, key=cv2.contourArea, reverse=True)
        # cv2.drawContours(self.img_norm, conture, -1, (255, 0, 0), thickness=2)
        # self.show_image(self.img_norm)

        need_conture = []
        for i in conture:
            # approx = cv2.approxPolyDP(i, 0.1 * cv2.arcLength(i, True), True)
            approx = cv2.approxPolyDP(i, 5, True)
            if len(approx) == 4:
                need_conture.append(i)
                x, y, w, h = cv2.boundingRect(approx)
                break
        cv2.drawContours(self.img_norm, need_conture, -1, (255, 0, 0), thickness=2)
        # self.img_norm = self.img_norm[y:y + h, x:x + w]
        # cv2.rectangle(self.img_norm, (x, y), (x + w, y + h), (255, 0, 0), thickness=2)
        self.show_image(self.img_norm, 1, "final")

        # img_HSV = cv2.cvtColor(self.img_norm, cv2.COLOR_BGR2HSV)
        # self.show_image(img_HSV)
        #
        # low_color = np.array([82,100,15])
        # high_color = np.array([154,100,100])
        # mask = cv2.inRange(img_HSV, low_color, high_color)
        # img_HSV[mask > 0] = ([0,0,0])
        # # mask=list(mask)
        # # print(*mask,sep="\n")
        # self.show_image(img_HSV)
        # self.show_image(cv2.cvtColor(img_HSV, cv2.COLOR_HSV2BGR))

    def color_chech(self):
        img = self.img_norm.copy()

        b, g, r = cv2.split(img)
        ret,b = cv2.threshold(b, 90, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print(b)
        self.show_image(b)

        ret, r = cv2.threshold(r, 90, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print(r)
        self.show_image(r)

        ret, g = cv2.threshold(g, 90, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        print(g)
        self.show_image(g)

        self.img_norm[r>0]=([0,0,0])
        self.show_image(self.img_norm)

    def __init__(self, sorce):
        self.img_norm = cv2.imread(sorce)
        self.show_image(self.img_norm, 1)


if __name__ == "__main__":
    a = QQQ4('../sorce/picture/auto2.webp')
    # a = QQQ4('../sorce/picture/auto1.jpg')
    # a = QQQ4('../sorce/picture/car.webp')
    # a = QQQ4('../sorce/picture/images.jpeg')
    # a = QQQ4('../sorce/picture/auto3.jpg')
    # a = QQQ4('../sorce/picture/list.jpeg')
    # QQQ4.test()
    # a.find_nomer()
    # a.f()
    a.color_chech()
