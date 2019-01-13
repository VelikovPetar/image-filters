from utils import image_utils
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    start = 0
    end = 255
    # for i in range(start, end):
    #     anchors.append((i, i * 1.5))
    x = [0, 28, 56, 85, 113, 141, 170, 198, 255]
    y = [0, 18, 46, 104, 145, 175, 206, 226, 255]
    '''
    (0, 0),
        (28, 18),
        (56, 46),
        (85, 104),
        (113, 145),
        (141, 175),
        (170, 206),
        (198, 226),
        (255, 255)
    '''
    anchors = []
    for i in range(0, len(x)):
        anchors.append((x[i], y[i]))
    func = image_utils.generate_interpolation_function(anchors)
    line = image_utils.generate_interpolation_function([(0, 0), (255, 255)])
    xnew = np.linspace(0, 255, 256, endpoint=True)
    plt.plot(x, y, 'o', xnew, func(xnew), '-', xnew, line(xnew), '--')
    plt.legend(['Anchors', 'Tone curve', 'No tone curve'], loc='best')
    plt.show()
