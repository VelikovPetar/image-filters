import cv2
import matplotlib.pyplot as plt
import numpy as np


def show_diff(grey, orig):
    x = np.linspace(0, 255, num=256, endpoint=True)
    y = {}
    h = np.size(orig, 0)
    w = np.size(orig, 1)
    for i in range(0, h):
        for j in range(0, w):
            key = grey[i][j]
            val = orig[i][j]
            # if key in y.keys() and y[key] != val:
            #     print("Differnt val for: %d. Found: %d. New: %d" % (key, y[key], val))
            y[key] = val
    sort = sorted(y.items())
    xn = []
    yn = []
    for k in sorted(y.keys()):
        xn.append(k)
        yn.append(y[k])
    print(xn)

    for i in range(0, len(xn)):
        if i % 10 == 0:
            print("(%d, %d)," % (xn[i], yn[i]))

    plt.plot(xn, yn, 'o')
    plt.xticks(np.arange(0, 255, step=5))
    plt.yticks(np.arange(0, 255, step=5))
    plt.show()


if __name__ == '__main__':
    reyes = cv2.imread('res/reyes_comp.jpg', cv2.IMREAD_COLOR)
    image = cv2.imread('res/image-me.jpg', cv2.IMREAD_COLOR)
    (b, g, r) = cv2.split(reyes)
    (b1, g1, r1) = cv2.split(image)
    show_diff(r, r1)
