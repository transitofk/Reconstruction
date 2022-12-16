import cv2
import numpy as np
#import math
#import matplotlib.pyplot as plt

#from saving_results import save_book
#from counting_metrics import metrics
from constructing_a_super_resolution import tests, reconstruction, reconstruction_primitive, choose_filter
from view import main_page


def run():
    main_page()


if __name__ == '__main__':
    run()

"""    
    img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    h, w = img.shape[:2]
    center = (int(w / 2), int(h / 2))
    img_HR_cur = np.zeros((h, w), np.uint8)
    num_frame = 10
    shift_x_y = []
    rot = []
    list_min = []
    size_kernel=5
    num_filter=2
    num_rec=2
    num_cycle=18
    alpha=0.15


    for i in range(num_frame):
        shift_x_y.append(np.random.randint(1, 15))
        rot.append(np.random.randint(-15, -1))


    list_min = tests(num_frame, list_min, img, rot, shift_x_y, center, w, h, size_kernel, num_filter)
    res = reconstruction(num_cycle, img_HR_cur, list_min, rot, shift_x_y, alpha, center, w, h, size_kernel,
                             num_filter)
    cv2.imshow('image', res)
    cv2.waitKey(0)"""




