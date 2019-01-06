from utils import image_utils
import cv2
import numpy as np


class ClarendonFilter:

    SATURATION_CHANGE_RATE = 20

    BLUE_TONE_CURVE_ANCHORS = [
        (0, 0),
        (28, 38),
        (56, 66),
        (85, 104),
        (113, 139),
        (141, 175),
        (170, 206),
        (198, 226),
        (255, 255)
    ]
    GREEN_TONE_CURVE_ANCHORS = [
        (0, 0),
        (28, 24),
        (56, 49),
        (85, 98),
        (113, 141),
        (141, 174),
        (170, 201),
        (198, 223),
        (227, 239),
        (255, 255)
    ]
    RED_TONE_CURVE_ANCHORS = [
        (0, 0),
        (28, 16),
        (56, 35),
        (85, 64),
        (113, 117),
        (141, 163),
        (170, 200),
        (198, 222),
        (227, 237),
        (255, 249)
    ]

    def __init__(self):
        pass

    def apply(self, source_img):
        image = image_utils.change_saturation(source_img, self.SATURATION_CHANGE_RATE)
        (b, g, r) = cv2.split(image)
        b = self.map_pixel_values(b, image_utils.create_tone_curve(self.BLUE_TONE_CURVE_ANCHORS))
        g = self.map_pixel_values(g, image_utils.create_tone_curve(self.GREEN_TONE_CURVE_ANCHORS))
        r = self.map_pixel_values(r, image_utils.create_tone_curve(self.RED_TONE_CURVE_ANCHORS))
        return cv2.merge([b, g, r])


    def map_pixel_values(self, image, pixel_values):
        h = np.size(image, 0)
        w = np.size(image, 1)
        for i in range(0, h):
            for j in range(0, w):
                image[i][j] = pixel_values[image[i][j]]
        return image