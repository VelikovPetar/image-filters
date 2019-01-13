import cv2

from utils import image_utils


class ClarendonFilter:
    TONE_CURVE_ANCHORS = [
        (0, 0),
        (26, 10),
        (52, 36),
        (85, 105),
        (113, 145),
        (141, 175),
        (170, 210),
        (198, 230),
        (255, 255)
    ]

    def apply(self, image):
        # Apply the tone curve transformation on each layer (b, g, r)
        (b, g, r) = cv2.split(image)
        b = image_utils.map_pixel_values(b, image_utils.create_tone_curve(self.TONE_CURVE_ANCHORS, True))
        g = image_utils.map_pixel_values(g, image_utils.create_tone_curve(self.TONE_CURVE_ANCHORS))
        r = image_utils.map_pixel_values(r, image_utils.create_tone_curve(self.TONE_CURVE_ANCHORS))
        image = cv2.merge([b, g, r])
        return image_utils.change_brightness(image, -20)
