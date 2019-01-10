import cv2

from utils import image_utils


class SlumberFilter:
    def apply(self, image):
        # Convert to Lab to give yellow-ish glow (increase the b - layer)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        (l, a, b) = cv2.split(image)
        # l = l + 7
        # l = np.clip(l, 0, 255)
        b = b + 7
        image = cv2.merge([l, a, b])
        image = cv2.cvtColor(image, cv2.COLOR_Lab2BGR)
        # Decrease contrast and increase brightness
        alpha = 0.95
        beta = 20
        image = image_utils.change_contrast_and_brightness(image, alpha, beta)
        # Blur the image
        image = cv2.blur(image, ksize=(2, 2))
        return image
