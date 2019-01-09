import cv2
import numpy as np


class ReyesFilter:
    def apply(self, source_img):
        image = cv2.cvtColor(source_img, cv2.COLOR_BGR2Lab).astype("float32")
        (L, a, b) = cv2.split(image)
        b = b + 15
        L = L + 20
        L = np.clip(L, 0, 255)
        image = cv2.merge([L, a, b])
        image = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_Lab2BGR)
        return image
