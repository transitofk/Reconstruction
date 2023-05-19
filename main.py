import cv2
import numpy as np
# import math
# import matplotlib.pyplot as plt

# from saving_results import save_book
from counting_metrics import metrics_new
from constructing_a_super_resolution import tests, reconstruction, reconstruction_primitive, choose_filter, add_noise
from view import main_page


def run():
    main_page()

def triangle_kernel(kerlen_size):
    r = np.arange(kerlen_size)
    kernel1d = (kerlen_size + 1 - np.abs(r - r[::-1])) / 2
    kernel2d = np.outer(kernel1d, kernel1d)
    kernel2d /= kernel2d.sum()
    return kernel2d


if __name__ == '__main__':
    run()
