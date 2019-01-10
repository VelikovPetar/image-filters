import cv2
from matplotlib import pyplot as plt

from filters.clarendon import ClarendonFilter
from filters.gingham import GinghamFilter
from filters.juno import JunoFilter
from filters.moon import MoonFilter
from filters.reyes import ReyesFilter
from filters.slumber import SlumberFilter
from utils import image_utils

FILTERS = {
    "clarendon": ClarendonFilter(),
    "gingham": GinghamFilter(),
    "moon": MoonFilter(),
    "reyes": ReyesFilter(),
    "juno": JunoFilter(),
    "slumber": SlumberFilter()
}

if __name__ == '__main__':
    img_name = 'res/image-me.jpg'
    image = image_utils.read_image(img_name)
    filter_name = "slumber"
    filtered_image = FILTERS[filter_name].apply(image)
    cv2.imwrite('res/slumber.jpg', filtered_image)

    # Convert images from BGR to RGB so 'pyplot' can display them correctly
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    filtered_image = cv2.cvtColor(filtered_image, cv2.COLOR_BGR2RGB)
    plt.subplot(121), plt.imshow(image, 'gray'), plt.title("Original image"), plt.axis("off")
    plt.subplot(122), plt.imshow(filtered_image, 'gray'), plt.title("Filtered image [%s]" % filter_name), plt.axis("off")
    plt.show()
