from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2

def calculate_metrics(original, fused_color, fused_dl, final):

    psnr_wavelet = peak_signal_noise_ratio(original, cv2.cvtColor(fused_color, cv2.COLOR_BGR2GRAY))
    ssim_wavelet = structural_similarity(original, cv2.cvtColor(fused_color, cv2.COLOR_BGR2GRAY), data_range=255)

    psnr_dl = peak_signal_noise_ratio(original, fused_dl)
    ssim_dl = structural_similarity(original, fused_dl, data_range=255)

    psnr_final = peak_signal_noise_ratio(original, cv2.cvtColor(final, cv2.COLOR_BGR2GRAY))
    ssim_final = structural_similarity(original, cv2.cvtColor(final, cv2.COLOR_BGR2GRAY), data_range=255)

    return psnr_wavelet, ssim_wavelet, psnr_dl, ssim_dl, psnr_final, ssim_final