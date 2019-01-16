import cv2

from utils import image_utils


class LarkFilter:
    CHANGE_SATURATION_RATE = 10

    CHANGE_BRIGHTNESS_RATE = 30

    # The points defining the blue tone curve
    BLUE_TONE_CURVE_ANCHORS = [
        (0, 0),
        (100, 95),
        (200, 205),
        (255, 230)
    ]

    # The points defining the green tone curve
    GREEN_TONE_CURVE_ANCHORS = [
        (0, 0),
        (100, 95),
        (200, 205),
        (255, 230)
    ]

    # The points defining the red tone curve
    RED_TONE_CURVE_ANCHORS = [
        (0, 0),
        (100, 95),
        (200, 205),
        (255, 230)
    ]

    def apply(self, image):
        image = image_utils.change_contrast_and_brightness(image, 0.95, 30)
        image = image_utils.change_saturation(image, 10)
        (b, g, r) = cv2.split(image)
        b = image_utils.map_pixel_values(b, image_utils.create_tone_curve(self.BLUE_TONE_CURVE_ANCHORS, True))
        g = image_utils.map_pixel_values(g, image_utils.create_tone_curve(self.GREEN_TONE_CURVE_ANCHORS))
        r = image_utils.map_pixel_values(r, image_utils.create_tone_curve(self.RED_TONE_CURVE_ANCHORS))
        image = cv2.merge([b, g, r])
        return image
