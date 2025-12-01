import os
from typing import List

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def is_image_file(path: str) -> bool:
    """Kiểm tra path có phải file ảnh hay không dựa trên phần mở rộng."""
    _, ext = os.path.splitext(path.lower())
    return ext in IMAGE_EXTENSIONS


def list_images_in_folder(folder: str) -> List[str]:
    """Liệt kê tất cả ảnh trong một thư mục (không đệ quy)."""
    files: List[str] = []
    if not os.path.isdir(folder):
        return files

    for name in os.listdir(folder):
        path = os.path.join(folder, name)
        if os.path.isfile(path) and is_image_file(path):
            files.append(path)

    files.sort()
    return files


def load_image(path: str) -> np.ndarray:
    """Đọc ảnh màu (BGR)."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Không tìm thấy file ảnh: {path}")

    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Không đọc được ảnh từ: {path}")

    return image


def save_image(path: str, image: np.ndarray) -> None:
    """Lưu ảnh ra đĩa, tự tạo thư mục nếu cần."""
    folder = os.path.dirname(path)
    if folder and not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    success = cv2.imwrite(path, image)
    if not success:
        raise IOError(f"Lưu ảnh thất bại: {path}")
