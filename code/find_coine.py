import cv2
import numpy as np
from matplotlib import pyplot as plt


class Coins:
    img_norm = None

    @staticmethod
    def show_image(img: np.ndarray, show_metod=0, name="") -> None:
        if show_metod == 0:
            cv2.imshow(name, img)
            cv2.waitKey(0)
        elif show_metod == 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            fig = plt.figure(name)
            plt.imshow(img)
            plt.show()

    def find(self):
        img = cv2.cvtColor(self.img_norm, cv2.COLOR_BGR2GRAY)
        r, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        self.show_image(img)

        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k, iterations=2)
        self.show_image(img)

        bg = cv2.dilate(img, k, iterations=3)
        self.show_image(bg)

        dist_tranform = cv2.distanceTransform(img, cv2.DIST_L2, 5)
        self.show_image(dist_tranform)
        print(dist_tranform)

        r, fg = cv2.threshold(dist_tranform, dist_tranform.max() * 0.7, 255, 0)
        self.show_image(fg)
        fg = np.uint8(fg)

        un = cv2.subtract(bg, fg)
        self.show_image(un)

        ret, markers = cv2.connectedComponents(fg)
        markers = markers + 1
        markers[un == 255] = 0

        markers = cv2.watershed(self.img_norm, markers)
        self.img_norm[markers == -1] = [255, 0, 0]
        self.show_image(self.img_norm)

    def find1(self):
        img = self.img_norm.copy()
        # img = cv2.resize(img, (img.shape[1] * 2, img.shape[0] * 2))
        self.show_image(img)

        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.show_image(img)
        self.show_image(cv2.applyColorMap(img, cv2.COLORMAP_JET))
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

        self.show_image(img)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k, iterations=1)
        self.show_image(img)

        bg = cv2.dilate(img, k, iterations=3)
        self.show_image(bg)

        dist = cv2.distanceTransform(img, cv2.DIST_L2, 5)
        ret, fg = cv2.threshold(dist, 0.7 * dist.max(), 255, 0)
        fg = np.uint8(fg)
        un = cv2.subtract(bg, fg)
        self.show_image(un)

        ret, markers = cv2.connectedComponents(fg)
        markers = markers + 1
        markers[un == 255] = 0
        markers = cv2.watershed(self.img_norm, markers)
        self.img_norm[markers == -1] = [0, 0, 255]
        # self.img_norm = cv2.dilate(self.img_norm[markers == -1], k, iterations=1)
        self.show_image(self.img_norm)

    def find3(self):
        img = cv2.cvtColor(self.img_norm, cv2.COLOR_BGR2GRAY)
        self.show_image(img)

        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, k, iterations=2)
        self.show_image(img)

        bg = cv2.dilate(img, k, iterations=3)
        self.show_image(bg, name="bg")

        fg = cv2.distanceTransform(img, cv2.DIST_L2, 5)
        ret, fg = cv2.threshold(fg, 0.85 * fg.max(), 255, 0)
        self.show_image(fg)
        fg = np.uint8(fg)
        un = cv2.subtract(bg, fg)
        self.show_image(un, name="un")

        col, markers = cv2.connectedComponents(fg)
        markers = markers + 1
        markers[un > 0] = 0

        markers = cv2.watershed(self.img_norm, markers)
        self.img_norm[markers == -1] = (255, 0, 0)
        self.show_image(self.img_norm)

    def __init__(self, name: str) -> None:
        self.img_norm = cv2.imread(name)
        self.img_norm = cv2.resize(self.img_norm, (self.img_norm.shape[1] * 2, self.img_norm.shape[0] * 2))
        self.show_image(self.img_norm, name="norm_coins")
        self.find3()


if __name__ == "__main__":
    a = Coins("../sorce/picture/water_coins.jpg")
