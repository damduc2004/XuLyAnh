# XLA -- Ứng dụng Xử Lý Ảnh (Pencil Sketch App)

XLA là một ứng dụng xử lý ảnh đơn giản sử dụng **OpenCV** và **PyQt5**,
cho phép người dùng tạo hiệu ứng **pencil sketch**, làm mịn ảnh bằng bộ
lọc **bilateral**, phát hiện biên bằng **Canny**, và một số thao tác xử
lý ảnh khác thông qua giao diện trực quan.

## 🎯 Tính năng chính

-   ✏️ Chuyển ảnh thành hiệu ứng sketch (nhẹ và mạnh)
-   🔍 Phát hiện biên (Edge Detection)
-   🖼 Làm mịn ảnh bằng Bilateral Filter
-   🖥 Giao diện đồ họa sử dụng PyQt5
-   📤 Hỗ trợ xuất ảnh đã xử lý
-   📂 Xem trước kết quả theo thời gian thực

## 📁 Cấu trúc thư mục

    XLA/
    │── main.py
    │── gui_app.py
    │── image_processing.py
    │── auto_params.py
    │── config.py
    │── io_utils.py
    │── requirements.txt
    │── examples/


## 🚀 Cài đặt

### 1. Tạo môi trường ảo

    python -m venv venv
    source venv/bin/activate
    venv\Scripts\activate

### 2. Cài đặt thư viện

    pip install -r requirements.txt

## ▶️ Chạy ứng dụng

    python main.py

## 🧠 Công nghệ sử dụng

-   OpenCV
-   NumPy
-   PyQt5
-   Python 3.9+

## 📦 Build .exe

    pyinstaller --noconfirm --name XLA --windowed main.py
