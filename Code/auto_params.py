import cv2
import numpy as np


def auto_suggest_params(gray: np.ndarray):
    h, w = gray.shape

    # 1) CONTRAST (độ tương phản)
    contrast = gray.std()

    # 2) EDGE DENSITY (mật độ biên)
    edges_tmp = cv2.Canny(gray, 80, 160) # Sử dụng ngưỡng trung bình để đánh giá mật độ biên
    edge_density = edges_tmp.mean()   # 0–255


    # 3) NOISE LEVEL (độ nhiễu)
    noise_level = cv2.Laplacian(gray, cv2.CV_64F).var()

    # 4) SMOOTHNESS 
    blur9 = cv2.GaussianBlur(gray, (9, 9), 0)
    smoothness = np.mean(np.abs(gray.astype(float) - blur9.astype(float)))

    # smoothness:
    # - cao  (>= 25)  → ảnh tự nhiên (phong cảnh, da mặt…)
    # - thấp (<= 20)  → logo, hình học (màu phẳng => ít khác biệt sau blur)

    # 5) STRONG-EDGE RATIO
    #    -> Nhận biết biên "thật"
    strong_edges = np.sum(edges_tmp > 200)
    weak_edges   = np.sum((edges_tmp > 50) & (edges_tmp <= 200))
    strong_ratio = strong_edges / max(weak_edges, 1) # tỉ lệ biên mạnh / biên yếu
    #strong_ratio > 0.35  => nhiều biên thật (logo, chữ, hình học)
    #strong_ratio < 0.35 và > 0.1   => ảnh tự nhiên (phong cảnh, chân dung)
    #strong_ratio < 0.1   => ảnh mờ, ít chi tiết (phong cảnh sương mù, chân dung thiếu sáng)
    # =======================================================
    #  SHARPNESS 
    sharpness = 30  # Mặc định
    # Trường hợp logo / hình học / chữ:
    # - edge density cao
    # - smoothness thấp (màu phẳng)
    # - strong edge ratio lớn
    if (
        edge_density > 40 and
        contrast > 50 and
        smoothness < 22 and
        strong_ratio > 0.35
    ):
        sharpness = 80   # strong sketch
    else:
        sharpness = 30   # soft sketch

    # Gaussian Blur ksize cho Color Dodge
    if contrast < 30:
        blur_ksize = 25     # rất mềm
    elif contrast < 60:
        blur_ksize = 17
    else:
        blur_ksize = 11     # rõ nét hơn

    #  Bilateral filter parameters
    sigma_color = min(max(int(noise_level / 4), 20), 120) 
    sigma_space = min(max(int(contrast / 2), 10), 50)

    if noise_level > 300:
        iterations = 3
    elif noise_level > 120:
        iterations = 2
    else:
        iterations = 1

    # Canny thresholds (phụ thuộc độ sắc nét ảnh)
    low = int(max(10, contrast * 0.8))
    high = int(min(200, contrast * 1.8))


    return {
        "sharpness": sharpness,
        "blur_ksize": blur_ksize,
        "sigma_color": sigma_color,
        "sigma_space": sigma_space,
        "iterations": iterations,
        "canny_low": low,
        "canny_high": high,

        # (Optional) debug values – nếu bạn muốn hiển thị
        "contrast": contrast,
        "edge_density": edge_density,
        "noise_level": noise_level,
        "smoothness": smoothness,
        "strong_ratio": strong_ratio,
    }
