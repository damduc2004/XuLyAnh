from typing import Dict, Tuple

import cv2
import numpy as np

from config import AppConfig, DEFAULT_CONFIG, EdgeConfig, BilateralConfig


def _ensure_odd(k: int) -> int:
    """Đảm bảo kernel size là số lẻ >= 1."""
    k = max(1, int(k))
    if k % 2 == 0:
        k += 1
    return k

def apply_bilateral(gray: np.ndarray, cfg: BilateralConfig) -> np.ndarray:
    """Làm mịn bằng bilateral filter nhiều lần."""
    result = gray.copy()
    d = max(1, cfg.diameter)
    if d % 2 == 0:
        d += 1
    for _ in range(max(1, cfg.iterations)):
        result = cv2.bilateralFilter(result, d, cfg.sigma_color, cfg.sigma_space)
    return result


def detect_edges(gray: np.ndarray, cfg: EdgeConfig) -> np.ndarray:
    """Phát hiện biên Canny."""
    low = int(cfg.low_threshold)
    high = int(cfg.high_threshold)
    if low > high:
        low, high = high, low
    edges = cv2.Canny(gray, low, high)
    return edges


def pencil_sketch(image_bgr: np.ndarray, config: AppConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Sketch mềm (ít nét, giống phác hoạ)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    # làm mịn trước để nhìn giống vẽ tay
    smooth = apply_bilateral(gray, config.smooth)

    # hiệu ứng dodge blend
    inverted = 255 - smooth
    k = _ensure_odd(config.sketch.blur_ksize)
    blur = cv2.GaussianBlur(inverted, (k, k), 0)

    sketch_gray = cv2.divide(gray, 255 - blur, scale=256)

    sketch_bgr = cv2.cvtColor(sketch_gray, cv2.COLOR_GRAY2BGR)
    return sketch_bgr


def pencil_sketch_strong(image_bgr: np.ndarray, config: AppConfig = DEFAULT_CONFIG) -> np.ndarray:
    """Sketch đậm, nét rõ (dùng thêm biên Canny + sharpen)."""
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

    smooth = apply_bilateral(gray, config.smooth)
    inverted = 255 - smooth
    k = _ensure_odd(config.sketch.blur_ksize)
    blur = cv2.GaussianBlur(inverted, (k, k), 0)
    sketch = cv2.divide(gray, 255 - blur, scale=256)

    # biên Canny
    edges = detect_edges(gray, config.edge)
    edges_inv = cv2.bitwise_not(edges)

    # kết hợp sketch + edges
    combined = cv2.bitwise_and(sketch, edges_inv)

    # sharpen cho nét đậm hơn
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]], dtype=np.float32)
    combined = cv2.filter2D(combined, -1, kernel)

    sketch_bgr = cv2.cvtColor(combined, cv2.COLOR_GRAY2BGR)
    return sketch_bgr


def process_image(
    image_bgr: np.ndarray,
    mode: str = "pencil",
    config: AppConfig = None,
    sharpness: int = 50,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Hàm xử lý ảnh chính.
    Trả về:
      - result_bgr: ảnh kết quả BGR
      - extras: dict (để GUI có thể unpack, tạm để rỗng)
    """
    if config is None:
        config = DEFAULT_CONFIG

    mode = (mode or "pencil").lower().strip()

    if mode == "pencil":
        if sharpness is None:
            sharpness = 50
        if sharpness < 50:
            result = pencil_sketch(image_bgr, config)
        else:
            result = pencil_sketch_strong(image_bgr, config)
        return result, {}

    # fallback an toàn
    result = pencil_sketch(image_bgr, config)
    return result, {}
