from dataclasses import dataclass, field


@dataclass
class EdgeConfig:
    """Tham số cho thuật toán phát hiện biên."""
    low_threshold: int = 50
    high_threshold: int = 150


@dataclass
class BilateralConfig:
    """Tham số cho bộ lọc bilateral (edge-preserving smoothing)."""
    diameter: int = 9
    sigma_color: float = 75.0
    sigma_space: float = 75.0
    iterations: int = 1


@dataclass
class SketchConfig:
    # kernel size dùng cho blur trong hiệu ứng sketch
    blur_ksize: int = 21


@dataclass
class AppConfig:
    edge: EdgeConfig = field(default_factory=EdgeConfig)
    smooth: BilateralConfig = field(default_factory=BilateralConfig)
    sketch: SketchConfig = field(default_factory=SketchConfig)

# Cấu hình mặc định dùng chung
DEFAULT_CONFIG = AppConfig()
