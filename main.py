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

    #для cv2 BGR
    #
    # img = cv2.imread('4.jpg', cv2.IMREAD_COLOR)
    # # img = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
    # h, w = img.shape[:2]
    # center = (int(w / 2), int(h / 2))
    # # print(img.shape)
    # # print(img.dtype)
    # color = cv2.split(img) # color = [b, g, r]
    # # print(color)
    # #
    # #
    # img_HR_cur = np.zeros((h, w), np.float64)
    #
    # shift_x_y = []
    # rot = []
    # list_min = []
    #
    # size_kernel = 3
    # num_frame = 12
    # num_filter = 2
    # num_cycle = 8
    # alpha = 0.2
    # beta = 0.001

    # cv2.imshow('image', color[0])
    # cv2.waitKey(0)
    # cv2.imwrite('res_4_color[0].jpg', color[0])


    # для главы 2.3
    # translation_matrix = np.float32([[1, 0, 10], [0, 1, 10]])
    # rotation_matrix = cv2.getRotationMatrix2D(center, -5, 1)
    # img_tr = cv2.warpAffine(img, translation_matrix, (w, h))
    # img_tr_rot = cv2.warpAffine(img_tr, rotation_matrix, (w, h))
    # # cv2.imwrite('warpAffine.jpg', img_tr_rot)
    # img_rot_blur3 = choose_filter(img_tr_rot, 2, 3)
    # # cv2.imwrite('filter.jpg', img_rot_blur3)
    # img_min = img_rot_blur3[1:h:3, 1:w:3]
    # # cv2.imwrite('img_min.jpg', img_min)
    # img_min_noise = add_noise(img_min)
    # # cv2.imwrite('noise.jpg', img_min_noise)
    #
    # img_obr = np.zeros((h, w), np.float64)
    # img_obr[1:h:3, 1:w:3] = img_min_noise
    # cv2.imwrite('img_obr.jpg', img_obr)
    # img_obr_blur3 = choose_filter(img_obr, 2, 3)
    # cv2.imwrite('img_obr_blur3.jpg', img_obr_blur3)
    # rotation_matrix_obr = cv2.getRotationMatrix2D(center, -5 * (-1), 1)
    # translation_matrix_obr = np.float32([[1, 0, 10 * (-1)], [0, 1, 10 * (-1)]])
    # img_obr_blur3_rot = cv2.warpAffine(img_obr_blur3, rotation_matrix_obr, (w, h))
    # img_obr = cv2.warpAffine(img_obr_blur3_rot, translation_matrix_obr, (w, h))
    # cv2.imwrite('img_obr_res.jpg', img_obr)

    # for i in range(num_frame):
    #     shift_x_y.append(np.random.randint(1, 15))
    #     rot.append(np.random.randint(-10, 10))
    #
    # list_min = tests(num_frame, list_min, img, rot, shift_x_y, center, w, h, size_kernel, num_filter)
    # cv2.imwrite('list_min[0].jpg', list_min[0])
    # cv2.imwrite('list_min[1].jpg', list_min[1])
    # cv2.imwrite('list_min[2].jpg', list_min[2])


    # -----------АЛГОРИТМ--------------------------------------------------------------------------------------------------------

    # for i in range(num_frame):
    #     shift_x_y.append(np.random.randint(1, 15))
    #     rot.append(np.random.randint(-10, 10))
    #
    # res_color = []
    # synthetic_tests_b = []
    # synthetic_tests_g = []
    # synthetic_tests_r = []
    #
    # for i, cur_img_color in enumerate(color):
    #     list_min = tests(num_frame, list_min, cur_img_color, rot, shift_x_y, center, w, h, size_kernel, num_filter)
    #     res = reconstruction(num_cycle, img_HR_cur, list_min, rot, shift_x_y, alpha, beta, center, w, h, size_kernel, num_filter)
    #
    #     res_color.append(res)
    #     if i == 0:
    #         synthetic_tests_b=list_min
    #     elif i == 1:
    #         synthetic_tests_g=list_min
    #     else:
    #         synthetic_tests_r=list_min
    #
    #     print("i \n", i)
    #     print("list_min \n", list_min)
    #     list_min=[]
    #     img_HR_cur = np.zeros((h, w), np.float64)
    #
    #
    # print("synthetic_tests_b ",synthetic_tests_b)
    # imgMerged = cv2.merge(res_color)
    # cv2.imshow('image', imgMerged)
    # cv2.waitKey(0)
    # # cv2.imwrite('res_4.jpg', imgMerged)
    #
    # for i, (frame_b, frame_g, frame_r) in enumerate(
    #         zip(synthetic_tests_b, synthetic_tests_g, synthetic_tests_r), start=1):
    #     # print("frame_b", synthetic_tests_b[0])
    #     # cv2.imshow('image', synthetic_tests_b[0])
    #     # cv2.waitKey(0)
    #     imgMerged = cv2.merge([frame_b, frame_g, frame_r])
    #     cv2.imshow('image', imgMerged)
    #     cv2.waitKey(0)
    #     cv2.imwrite('frame_b.jpg', imgMerged)
    # --------------------------------------------------------------------------------------------------------------------------


    # cv2.imwrite('res_4_res_color[0].jpg', res_color[0])
    # cv2.imwrite('res_4_res_color[1].jpg', res_color[1])
    # cv2.imwrite('res_4_res_color[2].jpg', res_color[2])

    # для первой картинки с фотоаппаратом

    # res_color = []
    # for cur_img_color in color:
    #
    #     noise = (np.random.rand(cur_img_color.shape[0], cur_img_color.shape[1]) * 6*20 - 6*10)
    #     cur_img_color = choose_filter(cur_img_color, 2, 7)
    #     noise_img = (cur_img_color + noise)
    #     noise_img[noise_img < 0] = 0
    #     noise_img[noise_img > 255] = 255
    #     noise_img = noise_img.astype(np.uint8)
    #     res_color.append(noise_img)
    #
    # imgMerged = cv2.merge(res_color)
    # cv2.imshow('image', imgMerged)
    # cv2.waitKey(0)
    # cv2.imwrite('res_7.jpg', imgMerged)

    # зависимость параметров
    # for i in range(num_frame):
    #     shift_x_y.append(np.random.randint(1, 15))
    #     rot.append(np.random.randint(-10, 10))
    #
    # list_min = tests(num_frame, list_min, img, rot, shift_x_y, center, w, h, size_kernel, num_filter)
    # res = reconstruction(num_cycle, img_HR_cur, list_min, rot, shift_x_y, alpha, beta, center, w, h, size_kernel, num_filter)
    #
    # cv2.imshow('image', res)
    # cv2.waitKey(0)
    # cv2.imwrite('res_alfa.jpg', res)





