import cv2


def metrics(metric, img, img_HR):
    metric_list = {
        "GMSD": cv2.quality.QualityGMSD_compute(img, img_HR)[0][0],
        "MSE": cv2.quality.QualityMSE_compute(img, img_HR)[0][0],
        "PSNR": cv2.quality.QualityPSNR_compute(img, img_HR)[0][0],
        "SSIM": cv2.quality.QualitySSIM_compute(img, img_HR)[0][0]
    }
    return metric_list.get(metric, "Invalid metric")

def metrics_new(img, img_HR):
  psnr = cv2.quality.QualityPSNR_create(img)
  ssim = cv2.quality.QualitySSIM_create(img)
  psnr_score = psnr.compute(img_HR)[0]
  ssim_score = ssim.compute(img_HR)[0]
  return psnr_score, ssim_score
