import os
from typing import Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from config import AppConfig
from image_processing import process_image
from auto_params import auto_suggest_params   # <=== THÊM IMPORT AUTO


class SketchMainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("SketchLab - Chuyển ảnh thành tranh vẽ")

        self.original_image: Optional[np.ndarray] = None
        self.result_image: Optional[np.ndarray] = None
        self.current_path: Optional[str] = None

        self.original_label: QLabel
        self.result_label: QLabel

        # sliders
        self.low_thresh_slider: QSlider
        self.high_thresh_slider: QSlider

        self.bf_diameter_slider: QSlider
        self.bf_sigma_color_slider: QSlider
        self.bf_sigma_space_slider: QSlider
        self.bf_iterations_slider: QSlider

        self.sketch_blur_slider: QSlider
        self.sharpness_slider: QSlider

        self._build_ui()

    # ---------- UI building ----------

    def _build_ui(self) -> None:
        central = QWidget(self)
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)

        # top: hai ảnh
        images_layout = QHBoxLayout()
        self.original_label = QLabel("Ảnh gốc")
        self.original_label.setAlignment(Qt.AlignCenter)
        self.original_label.setMinimumSize(320, 240)
        self.original_label.setStyleSheet(
            "background: #f0f0f0; border: 1px solid #cccccc;"
        )

        self.result_label = QLabel("Kết quả")
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setMinimumSize(320, 240)
        self.result_label.setStyleSheet(
            "background: #f0f0f0; border: 1px solid #cccccc;"
        )

        images_layout.addWidget(self.original_label, stretch=1)
        images_layout.addWidget(self.result_label, stretch=1)

        # bottom: controls
        controls_layout = QHBoxLayout()
        controls_layout.addWidget(self._create_mode_group(), stretch=1)
        controls_layout.addWidget(self._create_edge_group(), stretch=1)
        controls_layout.addWidget(self._create_bilateral_group(), stretch=1)
        controls_layout.addWidget(self._create_sketch_params_group(), stretch=1)

        main_layout.addLayout(images_layout, stretch=3)
        main_layout.addLayout(controls_layout, stretch=2)

        self.statusBar().showMessage("Chưa mở ảnh nào")

    def _create_mode_group(self) -> QGroupBox:
        group = QGroupBox("Chế độ & thao tác")
        layout = QVBoxLayout()

        mode_label = QLabel("Tranh vẽ chì (đen trắng)")
        mode_label.setStyleSheet("font-weight: bold;")

        btn_open = QPushButton("Mở ảnh...")
        btn_open.clicked.connect(self.open_image)

        btn_save = QPushButton("Lưu kết quả...")
        btn_save.clicked.connect(self.save_result)

        btn_reset = QPushButton("Reset tham số")
        btn_reset.clicked.connect(self.reset_parameters)

        btn_apply = QPushButton("Áp dụng")
        btn_apply.clicked.connect(self.update_preview)

        layout.addWidget(mode_label)
        layout.addSpacing(10)
        layout.addWidget(btn_open)
        layout.addWidget(btn_save)
        layout.addWidget(btn_reset)
        layout.addWidget(btn_apply)
        layout.addStretch(1)

        group.setLayout(layout)
        return group

    def _create_edge_group(self) -> QGroupBox:
        group = QGroupBox("Tham số phát hiện biên (Canny)")
        layout = QVBoxLayout()

        # ---- LOW THRESH ----
        self.low_thresh_slider = QSlider(Qt.Horizontal)
        self.low_thresh_slider.setRange(0, 255)
        self.low_thresh_slider.setValue(50)
        self.low_thresh_slider.valueChanged.connect(self._on_params_changed)

        self.low_label = QLabel(f"{self.low_thresh_slider.value()}")

        row1 = QHBoxLayout()
        row1.addWidget(self.low_thresh_slider)
        row1.addWidget(self.low_label)

        # ---- HIGH THRESH ----
        self.high_thresh_slider = QSlider(Qt.Horizontal)
        self.high_thresh_slider.setRange(0, 255)
        self.high_thresh_slider.setValue(150)
        self.high_thresh_slider.valueChanged.connect(self._on_params_changed)

        self.high_label = QLabel(f"{self.high_thresh_slider.value()}")

        row2 = QHBoxLayout()
        row2.addWidget(self.high_thresh_slider)
        row2.addWidget(self.high_label)

        layout.addWidget(QLabel("Low threshold:"))
        layout.addLayout(row1)
        layout.addWidget(QLabel("High threshold:"))
        layout.addLayout(row2)
        layout.addStretch(1)

        group.setLayout(layout)
        return group


    def _create_bilateral_group(self) -> QGroupBox:
        group = QGroupBox("Tham số làm mịn (bilateral filter)")
        layout = QVBoxLayout()

        # ---- DIAMETER ----
        self.bf_diameter_slider = QSlider(Qt.Horizontal)
        self.bf_diameter_slider.setRange(1, 15)
        self.bf_diameter_slider.setValue(9)
        self.bf_diameter_slider.valueChanged.connect(self._on_params_changed)
        self.bf_diameter_label = QLabel(str(self.bf_diameter_slider.value()))

        row_d = QHBoxLayout()
        row_d.addWidget(self.bf_diameter_slider)
        row_d.addWidget(self.bf_diameter_label)

        # ---- SIGMA COLOR ----
        self.bf_sigma_color_slider = QSlider(Qt.Horizontal)
        self.bf_sigma_color_slider.setRange(10, 150)
        self.bf_sigma_color_slider.setValue(75)
        self.bf_sigma_color_slider.valueChanged.connect(self._on_params_changed)
        self.bf_sigma_color_label = QLabel(str(self.bf_sigma_color_slider.value()))

        row_sc = QHBoxLayout()
        row_sc.addWidget(self.bf_sigma_color_slider)
        row_sc.addWidget(self.bf_sigma_color_label)

        # ---- SIGMA SPACE ----
        self.bf_sigma_space_slider = QSlider(Qt.Horizontal)
        self.bf_sigma_space_slider.setRange(10, 150)
        self.bf_sigma_space_slider.setValue(75)
        self.bf_sigma_space_slider.valueChanged.connect(self._on_params_changed)
        self.bf_sigma_space_label = QLabel(str(self.bf_sigma_space_slider.value()))

        row_ss = QHBoxLayout()
        row_ss.addWidget(self.bf_sigma_space_slider)
        row_ss.addWidget(self.bf_sigma_space_label)

        # ---- ITER ----
        self.bf_iterations_slider = QSlider(Qt.Horizontal)
        self.bf_iterations_slider.setRange(1, 5)
        self.bf_iterations_slider.setValue(1)
        self.bf_iterations_slider.valueChanged.connect(self._on_params_changed)
        self.bf_iterations_label = QLabel(str(self.bf_iterations_slider.value()))

        row_it = QHBoxLayout()
        row_it.addWidget(self.bf_iterations_slider)
        row_it.addWidget(self.bf_iterations_label)

        layout.addWidget(QLabel("Diameter:"))
        layout.addLayout(row_d)
        layout.addWidget(QLabel("Sigma color:"))
        layout.addLayout(row_sc)
        layout.addWidget(QLabel("Sigma space:"))
        layout.addLayout(row_ss)
        layout.addWidget(QLabel("Số lần lặp:"))
        layout.addLayout(row_it)
        layout.addStretch(1)

        group.setLayout(layout)
        return group


    def _create_sketch_params_group(self) -> QGroupBox:
        group = QGroupBox("Tham số Sketch")
        layout = QVBoxLayout()

        # ---- BLUR ----
        self.sketch_blur_slider = QSlider(Qt.Horizontal)
        self.sketch_blur_slider.setRange(1, 50)
        self.sketch_blur_slider.setValue(21)
        self.sketch_blur_slider.valueChanged.connect(self._on_params_changed)
        self.blur_label = QLabel(str(self.sketch_blur_slider.value()))

        row_blur = QHBoxLayout()
        row_blur.addWidget(self.sketch_blur_slider)
        row_blur.addWidget(self.blur_label)

        # ---- SHARPNESS ----
        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider.setRange(0, 100)
        self.sharpness_slider.setValue(50)
        self.sharpness_slider.valueChanged.connect(self._on_params_changed)
        self.sharpness_label = QLabel(str(self.sharpness_slider.value()))

        row_sharp = QHBoxLayout()
        row_sharp.addWidget(self.sharpness_slider)
        row_sharp.addWidget(self.sharpness_label)

        layout.addWidget(QLabel("Độ mịn (blur):"))
        layout.addLayout(row_blur)
        layout.addWidget(QLabel("Độ đậm nét (sharpness):"))
        layout.addLayout(row_sharp)

        # -------------- NÚT AUTO SUGGEST ----------------
        btn_auto = QPushButton("Gợi ý tham số tự động")
        btn_auto.clicked.connect(self.auto_suggest_params_clicked)
        layout.addWidget(btn_auto)
        # ------------------------------------------------

        layout.addStretch(1)

        group.setLayout(layout)
        return group


    # ---------- logic helpers ----------

    def _current_mode_key(self) -> str:
        # chỉ còn 1 mode
        return "pencil"

    def _build_config_from_ui(self) -> AppConfig:
        cfg = AppConfig()
        cfg.edge.low_threshold = self.low_thresh_slider.value()
        cfg.edge.high_threshold = self.high_thresh_slider.value()
        cfg.smooth.diameter = self.bf_diameter_slider.value()
        cfg.smooth.sigma_color = self.bf_sigma_color_slider.value()
        cfg.smooth.sigma_space = self.bf_sigma_space_slider.value()
        cfg.smooth.iterations = self.bf_iterations_slider.value()
        cfg.sketch.blur_ksize = self.sketch_blur_slider.value()
        return cfg

    # ---------- AUTO SUGGEST PARAMS ----------

    def auto_suggest_params_clicked(self):
        if self.original_image is None:
            QMessageBox.warning(self, "Thông báo", "Bạn chưa mở ảnh.")
            return

        gray = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        params = auto_suggest_params(gray)

        self.sharpness_slider.setValue(params["sharpness"])
        self.sketch_blur_slider.setValue(params["blur_ksize"])
        self.bf_sigma_color_slider.setValue(params["sigma_color"])
        self.bf_sigma_space_slider.setValue(params["sigma_space"])
        self.bf_iterations_slider.setValue(params["iterations"])
        self.low_thresh_slider.setValue(params["canny_low"])
        self.high_thresh_slider.setValue(params["canny_high"])

        self.update_preview()

    # ---------- slots ----------

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Chọn ảnh",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return

        data = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(self, "Lỗi khi mở ảnh", f"Không đọc được ảnh từ: {path}")
            return

        self.original_image = img
        self.current_path = path
        self.statusBar().showMessage(f"Đã mở ảnh: {os.path.basename(path)}")

        self.update_preview()

    def save_result(self) -> None:
        if self.result_image is None:
            QMessageBox.warning(self, "Chưa có kết quả", "Bạn chưa xử lý ảnh nào.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "Lưu kết quả",
            "sketch.png",
            "PNG Image (*.png);;JPEG Image (*.jpg *.jpeg);;All files (*.*)",
        )
        if not path:
            return

        ext = os.path.splitext(path)[1]
        if not ext:
            ext = ".png"

        success, buf = cv2.imencode(ext, self.result_image)
        if not success:
            QMessageBox.critical(self, "Lỗi khi lưu", "Không thể mã hóa ảnh để lưu.")
            return

        try:
            buf.tofile(path)
        except Exception:
            QMessageBox.critical(self, "Lỗi khi lưu", "Không thể lưu ảnh kết quả.")
            return

        self.statusBar().showMessage(f"Đã lưu kết quả: {os.path.basename(path)}")

    def reset_parameters(self) -> None:
        self.low_thresh_slider.setValue(50)
        self.high_thresh_slider.setValue(150)
        self.bf_diameter_slider.setValue(9)
        self.bf_sigma_color_slider.setValue(75)
        self.bf_sigma_space_slider.setValue(75)
        self.bf_iterations_slider.setValue(1)
        self.sketch_blur_slider.setValue(21)
        self.sharpness_slider.setValue(50)

        self.update_preview()

    def _on_params_changed(self):
        self.low_label.setText(str(self.low_thresh_slider.value()))
        self.high_label.setText(str(self.high_thresh_slider.value()))
        self.bf_diameter_label.setText(str(self.bf_diameter_slider.value()))
        self.bf_sigma_color_label.setText(str(self.bf_sigma_color_slider.value()))
        self.bf_sigma_space_label.setText(str(self.bf_sigma_space_slider.value()))
        self.bf_iterations_label.setText(str(self.bf_iterations_slider.value()))
        self.blur_label.setText(str(self.sketch_blur_slider.value()))
        self.sharpness_label.setText(str(self.sharpness_slider.value()))

        self.update_preview()

    def update_preview(self) -> None:
        if self.original_image is None:
            return

        try:
            cfg = self._build_config_from_ui()
            sharpness = self.sharpness_slider.value()
            mode = self._current_mode_key()
            result, _ = process_image(
                self.original_image,
                mode=mode,
                config=cfg,
                sharpness=sharpness,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Lỗi xử lý ảnh", str(exc))
            return

        self.result_image = result
        self._refresh_viewers()

    def _refresh_viewers(self) -> None:
        if self.original_image is not None:
            self._set_image_on_label(self.original_image, self.original_label)

        if self.result_image is not None:
            self._set_image_on_label(self.result_image, self.result_label)

    def _set_image_on_label(self, img_bgr: np.ndarray, label: QLabel) -> None:
        if img_bgr is None:
            return

        if len(img_bgr.shape) == 2:
            height, width = img_bgr.shape
            bytes_per_line = width
            qimg = QImage(
                img_bgr.data, width, height, bytes_per_line, QImage.Format_Grayscale8
            )
        else:
            rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            height, width, ch = rgb.shape
            bytes_per_line = ch * width
            qimg = QImage(
                rgb.data, width, height, bytes_per_line, QImage.Format_RGB888
            )

        pix = QPixmap.fromImage(qimg)
        label.setPixmap(
            pix.scaled(label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )


def main() -> None:
    import sys

    app = QApplication(sys.argv)
    window = SketchMainWindow()
    window.resize(1200, 600)
    window.show()
    sys.exit(app.exec_())
