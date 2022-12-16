import cv2
import numpy as np


def choose_filter(img, count_blur, step):
    if count_blur == 1:
        return cv2.blur(img, (step, step))
    elif count_blur == 2:
        return cv2.GaussianBlur(img, (step, step), 0)
    else:
        return cv2.bilateralFilter(img, step, 75, 75)



def scale(img, item_r, item_tr, center, w, h, step, count_blur):
    translation_matrix = np.float32([[1, 0, item_tr], [0, 1, item_tr]])
    rotation_matrix = cv2.getRotationMatrix2D(center, item_r, 1)
    img_tr = cv2.warpAffine(img, translation_matrix, (w, h))
    img_tr_rot = cv2.warpAffine(img_tr, rotation_matrix, (w, h))
    img_rot_blur3 = choose_filter(img_tr_rot, count_blur, step)
    img_min = img_rot_blur3[1:h:step, 1:w:step]
    return img_min


def upscale(img_min, item_r, item_tr, center, w, h, step, count_blur):
    img_obr = np.zeros((h, w), np.uint8)
    img_obr[1:h:step, 1:w:step] = img_min
    img_obr_blur3 =choose_filter(img_obr, count_blur, step)
    rotation_matrix_obr = cv2.getRotationMatrix2D(center, item_r * (-1), 1)
    translation_matrix_obr = np.float32([[1, 0, item_tr * (-1)], [0, 1, item_tr * (-1)]])
    img_obr_blur3_rot = cv2.warpAffine(img_obr_blur3, rotation_matrix_obr, (w, h))
    img_obr = cv2.warpAffine(img_obr_blur3_rot, translation_matrix_obr, (w, h))
    return img_obr


def tests(count, list_min, img, rot, shift_x_y, center, w, h, step, count_blur):
    for item_r, item_tr in zip(rot[:count:], shift_x_y[:count:]):
        list_min.append(scale(img, item_r, item_tr, center, w, h, step, count_blur))
    return list_min


def reconstruction(iter, img_HR, list_min, rot, shift_x_y, alpha, center, w, h, step, count_blur):
    for it in range(iter):
        for img_min_LR, item_r, item_tr in zip(list_min, rot, shift_x_y):
            img_min_cur = scale(img_HR, item_r, item_tr, center, w, h, step, count_blur)
            diff_img_min = img_min_LR - img_min_cur
            diff_HR = upscale(diff_img_min, item_r, item_tr, center, w, h, step, count_blur)
            img_HR = np.add(img_HR, alpha * diff_HR, out=img_HR, casting="unsafe")
    return img_HR


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