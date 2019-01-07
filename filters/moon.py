import cv2

from utils import image_utils


class MoonFilter:
    # The points defining the tone curve
    TONE_CURVE_ANCHORS = [
        (0, 0),
        (22, 42),
        (56, 60),
        (91, 120),
        (200, 219),
        (246, 255)
    ]

    def apply(self, source_img):
        image = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
        image = image_utils.map_pixel_values(image, image_utils.create_tone_curve(self.TONE_CURVE_ANCHORS))
        return image
