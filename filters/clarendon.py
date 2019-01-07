import cv2

from utils import image_utils


class ClarendonFilter:
    SATURATION_CHANGE_RATE = 20

    # The points defining the blue tone curve
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

    # The points defining the green tone curve
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

    # The points defining the red tone curve
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

    def apply(self, source_img):
        # Increase the saturation of the image
        image = image_utils.change_saturation(source_img, self.SATURATION_CHANGE_RATE)
        # Apply the tone curve transformation on each layer (b, g, r)
        (b, g, r) = cv2.split(image)
        b = image_utils.map_pixel_values(b, image_utils.create_tone_curve(self.BLUE_TONE_CURVE_ANCHORS))
        g = image_utils.map_pixel_values(g, image_utils.create_tone_curve(self.GREEN_TONE_CURVE_ANCHORS))
        r = image_utils.map_pixel_values(r, image_utils.create_tone_curve(self.RED_TONE_CURVE_ANCHORS))
        return cv2.merge([b, g, r])

