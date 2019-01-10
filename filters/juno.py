import cv2

from utils import image_utils


class JunoFilter:
    SATURATION_CHANGE_RATE = 10

    RED_TONE_CURVE_ANCHORS = [
        (0, 0),
        (64, 68),
        (128, 136),
        (192, 204),
        (255, 255)
    ]

    def apply(self, image):
        # Increase the saturation to obtain more vibrant colors
        image = image_utils.change_saturation(image, self.SATURATION_CHANGE_RATE)
        # Enhance the red colors by adding small boost to the 'a' layer in the Lab color space (the 'a' layer represents
        # the green -> red axis)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        (l, a, b) = cv2.split(image)
        a = a + 5
        image = cv2.merge([l, a, b])
        image = cv2.cvtColor(image, cv2.COLOR_Lab2BGR)
        # Adjust the red layer to further enhance the red colors
        (b, g, r) = cv2.split(image)
        r = image_utils.map_pixel_values(r, image_utils.create_tone_curve(self.RED_TONE_CURVE_ANCHORS))
        image = cv2.merge([b, g, r])
        # Increase the contrast and reduce the brightness
        alpha = 1.4
        beta = -35
        image = image_utils.change_contrast_and_brightness(image, alpha, beta)
        return image
