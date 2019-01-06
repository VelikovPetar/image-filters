import cv2
import numpy as np
from scipy.interpolate import interp1d


def read_image(image_name):
    return cv2.imread(image_name, cv2.IMREAD_COLOR)


def change_saturation(source_image, change_rate):
    """
    Applies a change in saturation of a given image. (+ or -)

    :param source_image: the source image
    :param change_rate: the fixed applied to change the saturation
    :return: the image with the changed saturation
    """
    hsv_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(hsv_image)
    s = s + change_rate
    s = np.clip(s, 0, 255)
    hsv_image = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_image.astype("uint8"), cv2.COLOR_HSV2BGR)


def create_tone_curve(anchors):
    """
    Creates a tone curve based on given anchor points.

    The tone curve is a curve that shows how the pixel values from an input image should be displayed in the output
    image. In the case where the pixel values should not not be modified between the input and output images, this curve
    is a line from (0, 0) to (255, 255), meaning that each pixel value from the input image is mapped to the same value.
    This curve can be altered if we want to map a concrete value to a different value in the output image. If we only
    replace those values, we will get unnatural changes in color in the output image, so we need to fit the actual curve
    to the given mapping values(anchors). In order to fit the curve according to the given anchors, we are using
    'Cubic spline interpolation', which fits the curve by connecting the given anchors with a cubic polynomial,
    resulting in a smooth curve.

    :param anchors: the anchor points
    :return: a list of 256 number elements(between 0 and 255 included), representing the mapped pixel values, given by
    the fitted curve to the anchor points
    """
    input_pixel_values = []
    output_pixel_values = []

    for anchor in anchors:
        input_value = anchor[0]
        output_value = anchor[1]
        if 0 <= input_value <= 255 and 0 <= output_value <= 255:
            input_pixel_values.append(input_value)
            output_pixel_values.append(output_value)

    if 0 not in input_pixel_values:
        input_pixel_values.insert(0, 0)
        output_pixel_values.insert(0, 0)

    if 255 not in input_pixel_values:
        input_pixel_values.append(255)
        output_pixel_values.append(255)

    curve = interp1d(input_pixel_values, output_pixel_values, kind='cubic')
    output_pixel_values = curve(np.linspace(0, 255, num=256, endpoint=True))
    return output_pixel_values.astype(int)
