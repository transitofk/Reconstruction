import cv2


def metrics(metric, img, img_HR):
    if metric == "GMSD":
        return cv2.quality.QualityGMSD_compute(img, img_HR)[0][0]
    elif metric == "MSE":
        return cv2.quality.QualityMSE_compute(img, img_HR)[0][0]
    elif metric == "PSNR":
        return cv2.quality.QualityPSNR_compute(img, img_HR)[0][0]
    else:
        return cv2.quality.QualitySSIM_compute(img, img_HR)[0][0]


