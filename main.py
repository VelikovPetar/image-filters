from filters.clarendon import ClarendonFilter
from filters.gingham import GinghamFilter
from utils import image_utils
import numpy as np
from matplotlib import pyplot as plt
import cv2


FILTERS = {
    "clarendon": ClarendonFilter(),
    "gingham": GinghamFilter()
}

if __name__ == '__main__':
    img_name = 'res/image-me.jpg'
    image = image_utils.read_image(img_name)
    filter_name = "clarendon"
    filtered_image = FILTERS[filter_name].apply(image)

    # Convert images from BGR to RGB so 'pyplot' can display them correctly
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
    plt.subplot(121), plt.imshow(image, 'gray'), plt.title("Original image"), plt.axis("off")
    plt.subplot(122), plt.imshow(filtered_image, 'gray'), plt.title("Filtered image [%s]" % filter_name), plt.axis("off")
    plt.show()
