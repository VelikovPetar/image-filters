import cv2
import numpy as np

from utils import image_utils


class GinghamFilter:
    SATURATION_CHANGE_RATE = -20

    # A modified Sepia kernel ('Sepia' / 3.4)
    GINGHAM_KERNEL = np.array([
        [0.213, 0.157, 0.038],
        [0.102, 0.201, 0.049],
        [0.115, 0.226, 0.055]
    ])

    def apply(self, source_img):
        # Apply a slight gaussian blur to smooth the edges
        image = cv2.GaussianBlur(source_img, (3, 3), 0)
        # Decrease the saturation of the image
        image = image_utils.change_saturation(image, self.SATURATION_CHANGE_RATE)
        # Filter the image with a 'Gingham kernel(modified Sepia kernel)'
        image = cv2.filter2D(image, -1, kernel=self.GINGHAM_KERNEL)
        return image
