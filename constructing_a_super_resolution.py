import cv2
import numpy as np
from skimage.util import random_noise

def triangle_kernel(kerlensize):
    r = np.arange(kerlensize)
    kernel1d = (kerlensize + 1 - np.abs(r - r[::-1])) / 2
    kernel2d = np.outer(kernel1d, kernel1d)
    kernel2d /= kernel2d.sum()
    return kernel2d

def choose_filter(img, count_blur, step):
    filter_list = {
        1: cv2.blur(img, (step, step)),
        2: cv2.GaussianBlur(img, (step, step*2-1), 0),
        3: cv2.filter2D(img, -1, triangle_kernel(step))
    }
    return filter_list.get(count_blur, "Invalid count blur")


# def add_noise(img, id_noise):
#     if id_noise == 1:
#         # Generate Gaussian noise = 1
#         gauss = np.random.normal(0, 0.5, img.size)
#         # gauss = gauss.reshape(img.shape[0], img.shape[1], img.shape[2]).astype('uint8')
#         gauss = gauss.reshape(img.shape[0], img.shape[1]).astype('uint8')
#         img_gauss = cv2.add(img, gauss)
#         return img_gauss
#     elif id_noise == 2:
#         # Speckle noise = 2
#         gauss = np.random.normal(0, 0.5, img.size)
#         gauss = gauss.reshape(img.shape[0], img.shape[1]).astype('uint8')
#         noise = img + img * gauss
#         return noise
#     else:
#         # Add salt-and-pepper noise to the image = 3
#         noise_img = random_noise(img, mode='s&p', amount=0.1)
#         # The above function returns a floating-point image
#         # on the range [0, 1], thus we changed it to 'uint8' and from [0,255]
#         noise_img = np.array(255 * noise_img, dtype='uint8')
#         return noise_img

def add_noise(img):
    noise = (np.random.rand(img.shape[0], img.shape[1]) * 20 - 10)
    noise_img = (img + noise)
    noise_img[noise_img < 0] = 0
    noise_img = noise_img.astype(np.uint8)
    return noise_img


def scale(img, item_r, item_tr, center, w, h, step, count_blur):
    translation_matrix = np.float32([[1, 0, item_tr], [0, 1, item_tr]])
    rotation_matrix = cv2.getRotationMatrix2D(center, item_r/10.0, 1)
    img_tr = cv2.warpAffine(img, translation_matrix, (w, h))
    img_tr_rot = cv2.warpAffine(img_tr, rotation_matrix, (w, h))
    img_rot_blur3 = choose_filter(img_tr_rot, count_blur, step)
    img_min = img_rot_blur3[1:h:step, 1:w:step]
    img_min_noise = add_noise(img_min)
    return img_min_noise


def upscale(img_min, item_r, item_tr, center, w, h, step, count_blur):
    img_obr = np.zeros((h, w), np.float64)
    # img_min_noise = cv2.fastNlMeansDenoising(img_min, None, 3, 7, 21)
    img_obr[1:h:step, 1:w:step] = img_min
    img_obr_blur3 = choose_filter(img_obr, count_blur, step)
    rotation_matrix_obr = cv2.getRotationMatrix2D(center, item_r * (-1)/10.0, 1)
    translation_matrix_obr = np.float32([[1, 0, item_tr * (-1)], [0, 1, item_tr * (-1)]])
    img_obr_blur3_rot = cv2.warpAffine(img_obr_blur3, rotation_matrix_obr, (w, h))
    img_obr = cv2.warpAffine(img_obr_blur3_rot, translation_matrix_obr, (w, h))
    return img_obr

def tests(count, list_min, img, rot, shift_x_y, center, w, h, step, count_blur):
    for item_r, item_tr in zip(rot[:count:], shift_x_y[:count:]):
        list_min.append(scale(img, item_r, item_tr, center, w, h, step, count_blur).astype(np.float64))
    return list_min


def reconstruction(iter, img_HR, list_min, rot, shift_x_y, alpha, center, w, h, step, count_blur):
    for it in range(iter):
        for img_min_LR, item_r, item_tr in zip(list_min, rot, shift_x_y):
            img_min_cur = scale(img_HR, item_r, item_tr, center, w, h, step, count_blur)
            diff_img_min = img_min_LR - img_min_cur
            diff_HR = upscale(diff_img_min, item_r, item_tr, center, w, h, step, count_blur)
            img_HR = np.add(img_HR, alpha * diff_HR, out=img_HR, casting="unsafe")
    img_HR[img_HR < 0] = 0
    img_HR[img_HR > 255] = 255
    return img_HR.astype(np.uint8)


def upscale_primitive(img_min, item_r, item_tr, center, w, h):
    img_obr = cv2.resize(img_min, [w, h], interpolation=cv2.INTER_NEAREST)
    rotation_matrix_obr = cv2.getRotationMatrix2D(center, item_r * (-1), 1)
    translation_matrix_obr = np.float32([[1, 0, item_tr * (-1)], [0, 1, item_tr * (-1)]])
    img_obr_blur3_rot = cv2.warpAffine(img_obr, rotation_matrix_obr, (w, h))
    img_obr = cv2.warpAffine(img_obr_blur3_rot, translation_matrix_obr, (w, h))
    return img_obr


def reconstruction_primitive(img_HR, list_min, rot, shift_x_y, center, w, h):
    for img_min_LR, item_r, item_tr in zip(list_min, rot, shift_x_y):
        img = upscale_primitive(img_min_LR, item_r, item_tr, center, w, h)
        np.add(img_HR, img / len(list_min), out=img_HR, casting="unsafe")
    return img_HR
