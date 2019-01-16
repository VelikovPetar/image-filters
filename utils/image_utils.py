import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d


def read_image(image_name):
    return cv2.imread(image_name, cv2.IMREAD_COLOR)


def change_saturation(source_image, change_rate):
    """
    Applies a change in saturation of a given image. (+ or -)

    :param source_image: the source image
    :param change_rate: the fixed amount applied to change the saturation
    :return: the image with the changed saturation
    """
    hsv_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(hsv_image)
    s = s + change_rate
    s = np.clip(s, 0, 255)
    hsv_image = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_image.astype("uint8"), cv2.COLOR_HSV2BGR)


def change_brightness(source_image, brighten_rate):
    """
    Increases the brightness of the image.

    :param source_image: the source image
    :param brighten_rate: the fixed amount applied to change the brightness
    :return: the image with changed brightness
    """
    hsv_image = cv2.cvtColor(source_image, cv2.COLOR_BGR2HSV).astype("float32")
    (h, s, v) = cv2.split(hsv_image)
    v = v + brighten_rate
    v = np.clip(v, 0, 255)
    hsv_image = cv2.merge([h, s, v])
    return cv2.cvtColor(hsv_image.astype("uint8"), cv2.COLOR_HSV2BGR)


def change_contrast_and_brightness(image, alpha, beta):
    """
    Changes the image contrast and brightness:
    The new image is calculated as: new_image = old_image * alpha + beta
    alpha 1  beta 0      --> no change
    0 < alpha < 1        --> lower contrast
    alpha > 1            --> higher contrast
    -127 < beta < +127   --> good range for brightness values

    :param image: the source image
    :param alpha: the parameter affecting the contrast
    :param beta: the parameter affecting the brightness
    :return: the modified image
    """
    image = np.float32(image)
    image = image * alpha + beta
    image = np.clip(image, 0, 255)
    image = np.uint8(image)
    return image


def create_tone_curve(anchors, should_draw_curve=False):
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
    :param should_draw_curve: whther the method should draw the interpolated curve
    :return: a list of 256 number elements(between 0 and 255 included), representing the mapped pixel values, given by
    the fitted curve to the anchor points
    """
    curve = generate_interpolation_function(anchors)
    if should_draw_curve:
        draw_curve(curve, anchors)
    output_pixel_values = curve(np.linspace(0, 255, num=256, endpoint=True))
    return output_pixel_values.astype(int)


def generate_interpolation_function(anchors):
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
    kind = 'cubic'
    if len(input_pixel_values) <= 2:
        kind = 'linear'
    return interp1d(input_pixel_values, output_pixel_values, kind=kind)


def draw_curve(func, anchors):
    x = []
    y = []
    for a in anchors:
        x.append(a[0])
        y.append(a[1])
    line = generate_interpolation_function([(0, 0), (255, 255)])
    xnew = np.linspace(0, 255, 256, endpoint=True)
    plt.plot(x, y, 'o', xnew, func(xnew), '-', xnew, line(xnew), '--')
    plt.legend(['Anchors', 'Tone curve', 'No tone curve'], loc='best')
    plt.show()


def map_pixel_values(image, pixel_values):
    """
    Transorms the input image to an output image, by replacing each pixel value with different one, given in the
    'pixel_values' list.

    :param image: the input image
    :param pixel_values: the list containing the new pixel values
    :return:
    """
    h = np.size(image, 0)
    w = np.size(image, 1)
    for i in range(0, h):
        for j in range(0, w):
            image[i][j] = np.clip(pixel_values[image[i][j]], 0, 255)
    return image


rgb_scale = 255
cmyk_scale = 100


def rgb_to_cmyk(r, g, b):
    if (r == 0) and (g == 0) and (b == 0):
        # black
        return 0, 0, 0, cmyk_scale

    # rgb [0,255] -> cmy [0,1]
    c = 1 - r / float(rgb_scale)
    m = 1 - g / float(rgb_scale)
    y = 1 - b / float(rgb_scale)

    # extract out k [0,1]
    min_cmy = min(c, m, y)
    c = (c - min_cmy)
    m = (m - min_cmy)
    y = (y - min_cmy)
    k = min_cmy

    # rescale to the range [0,cmyk_scale]
    return c * cmyk_scale, m * cmyk_scale, y * cmyk_scale, k * cmyk_scale


def cmyk_to_rgb(c, m, y, k):
    """
    """
    r = rgb_scale * (1.0 - (c + k) / float(cmyk_scale))
    g = rgb_scale * (1.0 - (m + k) / float(cmyk_scale))
    b = rgb_scale * (1.0 - (y + k) / float(cmyk_scale))
    return r, g, b
