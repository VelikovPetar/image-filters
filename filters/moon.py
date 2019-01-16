import cv2

from utils import image_utils


class MoonFilter:
    # The points defining the tone curve
    TONE_CURVE_ANCHORS = [
        (20, 52),
        (30, 54),
        (40, 67),
        (50, 74),
        (60, 82),
        (70, 94),
        (80, 122),
        (90, 148),
        (100, 161),
        (110, 179),
        (120, 186),
        (130, 197),
        (140, 198),
        (150, 203),
        (160, 208),
        (170, 216),
        (180, 223),
        (190, 226),
        (200, 231),
        (210, 235),
        (220, 243),
        (230, 243),
        (240, 246),
        (250, 249),
        (255, 255)
    ]

    def apply(self, source_img):
        image = cv2.cvtColor(source_img, cv2.COLOR_BGR2GRAY)
        image = image_utils.map_pixel_values(image, image_utils.create_tone_curve(self.TONE_CURVE_ANCHORS, True))
        alpha = 1
        beta = 0
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = image_utils.change_contrast_and_brightness(image, alpha, beta)
        return image
