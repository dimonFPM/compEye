import cv2
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    # region поиск доступных камер
    # frame = False
    # i = 0
    # trueList = []
    # while True:
    #     cam = cv2.VideoCapture(i)
    #     ret, frame = cam.read()
    #     if ret == True:
    #         trueList.append(i)
    #     if i == 1000:
    #         break
    #     i += 1
    # print(f"найдено доступных камер {len(trueList)}\n {trueList}")
    # endregion

    nCam = 700
    cam = cv2.VideoCapture(nCam)
check, frame = cam.read()
if check == False:
    print(f"Данное устройство ({nCam}) не доступно по не изветсной причине")
    exit()
cam.set(3, frame.shape[1] * 2)
cam.set(4, frame.shape[0] * 2)

while True:
    check, frame = cam.read()
    green = np.zeros_like(frame)
    green[:, :, 2] = frame[:, :, 2]
    cv2.imshow("norm frame", green)
    plt.show()
    if cv2.waitKey(1) & 0xff == ord("q"):
        break

cam.release()
cv2.destroyAllWindows()
print("конец")

# a = np.zeros((800, 300, 3), dtype="uint8")
# a = cv2.cvtColor(a, cv2.COLOR_BGR2HSV)
# k = 0
# for i in range(180):
#     a[k:k + 3, :, 0] = i
#     a[k:k + 3, :, 1:3] = 200
#     if i % 15 == 0:
#         a[k:k + 3, :, 1:3] = 0
#     k += 3
# a = cv2.cvtColor(a, cv2.COLOR_HSV2BGR)
# cv2.imshow("", a)
# cv2.waitKey(0)
