import cv2
import numpy as np
from utils import image_utils


class ReyesFilter:

    B_TONE_CURVE = [
        (20, 6),
        (33, 4),
        (43, 13),
        (53, 23),
        (63, 19),
        (73, 30),
        (83, 42),
        (93, 45),
        (103, 46),
        (113, 61),
        (123, 71),
        (133, 85),
        (143, 91),
        (153, 109),
        (163, 115),
        (173, 132),
        (183, 169),
        (193, 167),
        (203, 200),
        (213, 250)
    ]

    G_TONE_CURVE = [
        (34, 19),
        (47, 17),
        (57, 30),
        (67, 27),
        (77, 35),
        (87, 41),
        (97, 51),
        (107, 61),
        (117, 64),
        (127, 70),
        (137, 74),
        (147, 89),
        (157, 90),
        (167, 126),
        (177, 116),
        (187, 147),
        (197, 160),
        (207, 206),
        (217, 222),
        (227, 255)
    ]

    R_TONE_CURVE = [
        (45, 14),
        (56, 24),
        (66, 37),
        (76, 45),
        (86, 52),
        (96, 60),
        (106, 69),
        (116, 76),
        (126, 85),
        (136, 95),
        (146, 101),
        (156, 110),
        (166, 113),
        (176, 126),
        (186, 135),
        (196, 147),
        (206, 170),
        (216, 212),
        (226, 247),
        (237, 245)
    ]
    TONE_CURVE = [
        (0, 32),
        (127, 164),
        (224, 224),
        (255, 216)
    ]

    RED_TONE_CURVE = [
        (0, 32),
        (127, 140),
        (224, 214),
        (255, 236)
    ]


    def apply(self, image):
        image = image_utils.change_saturation(image, -50)
        # image = image_utils.change_brightness(image, 30)
        (b, g, r) = cv2.split(image)
        b = image_utils.map_pixel_values(b, image_utils.create_tone_curve(self.TONE_CURVE))
        g = image_utils.map_pixel_values(g, image_utils.create_tone_curve(self.TONE_CURVE))
        r = image_utils.map_pixel_values(r, image_utils.create_tone_curve(self.TONE_CURVE))
        image = cv2.merge([b, g, r])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)
        (l, a, b) = cv2.split(image)
        # l = l + 25
        # np.clip(l, 0, 255)
        # a = a - 5
        b = b + 5
        image = cv2.merge([l, a, b])
        image = cv2.cvtColor(image, cv2.COLOR_Lab2BGR)
        image = image_utils.change_contrast_and_brightness(image, 1, -5)
        return image
