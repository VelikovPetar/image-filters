from filters.clarendon import ClarendonFilter
from utils import image_utils
import cv2

if __name__ == '__main__':
    img_name = 'res/image-me.jpg'
    image = image_utils.read_image(img_name)
    clarendon_filter = ClarendonFilter()
    filtered_image = clarendon_filter.apply(image)
    cv2.imshow("Image", image)
    cv2.imshow("Clarendon", filtered_image)
    cv2.waitKey(0)
