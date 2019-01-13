import cv2
import numpy as np

from utils import image_utils


class GinghamFilter:
    TONE_CURVE = [
        (0, 0),
        (32, 32),
        (64, 64),
        (96, 118),
        (140, 164),
        (160, 184),
        (192, 208),
        (224, 224),
        (255, 255)
    ]

    def apply(self, image):
        # Decrease contrast and increase brightness
        alpha = 0.95
        beta = 50
        image = image_utils.change_contrast_and_brightness(image, alpha, beta)

        # Give a yellowish tint (maybe this is not necessary)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab).astype("float32")
        (l, a, b) = cv2.split(image)
        l = np.clip(l - 30, 0, 255)
        b = b + 3
        image = cv2.merge([l, a, b])
        image = cv2.cvtColor(image.astype("uint8"), cv2.COLOR_Lab2BGR)

        # Enhance the medium highlights
        (b, g, r) = cv2.split(image)
        g = image_utils.map_pixel_values(g, image_utils.create_tone_curve(self.TONE_CURVE, True))
        r = image_utils.map_pixel_values(r, image_utils.create_tone_curve(self.TONE_CURVE))
        b = image_utils.map_pixel_values(b, image_utils.create_tone_curve(self.TONE_CURVE))
        image = cv2.merge([b, g, r])
        return image
