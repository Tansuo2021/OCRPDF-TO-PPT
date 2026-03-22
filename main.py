import sys
import os
import tempfile
import copy
import json
import re
import shutil
import platform
import contextlib
import logging
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout,
    QPushButton, QLabel, QListWidget, QListWidgetItem, QFileDialog,
    QMessageBox, QGraphicsView, QGraphicsScene, QGraphicsRectItem,
    QSplitter, QGraphicsTextItem, QFrame, QSlider, QToolButton,
    QTextEdit, QGraphicsItemGroup, QGraphicsItem, QTabWidget, QProgressDialog, QCheckBox,
    QDialog, QTableWidget, QTableWidgetItem, QHeaderView,
    QSizePolicy,
    QColorDialog,
    QComboBox,
    QScrollArea
)
from PySide6.QtCore import Qt, QSize, QThread, Signal, QTimer, QPointF, QPoint, QUrl, QLocale
from PySide6.QtGui import (
    QPixmap, QPen, QColor, QFont, QFontMetricsF, QTextOption, QImage, QIcon, QBrush, QAction, QKeySequence, QDesktopServices
)
import cv2
import numpy as np
from image_utils import build_asset_path, imread_any, imwrite_any

logger = logging.getLogger(__name__)

# ==================== 常量定义 ====================
# 图像处理
TARGET_IMAGE_HEIGHT = 1080          # 缩放目标高度

# 字体大小
FONT_SIZE_MIN = 6                   # 最小字号
FONT_SIZE_MAX = 600                 # 最大字号

# 透明度
DEFAULT_BG_ALPHA = 120              # 默认背景透明度
ALPHA_MIN = 0                       # 透明度最小值
ALPHA_MAX = 255                     # 透明度最大值

# 缩放
ZOOM_MIN = 10                       # 缩放最小值 (%)
ZOOM_MAX = 400                      # 缩放最大值 (%)

# IOPaint 参数
BOX_PADDING_DEFAULT = 5             # 默认框内边距
BOX_PADDING_MAX = 200               # 框内边距最大值
CROP_PADDING_DEFAULT = 50           # 默认裁剪边距
CROP_PADDING_MAX = 2000             # 裁剪边距最大值

# 临时文件清理
PREVIEW_FILE_MIN_AGE_SEC = 600      # 预览文件最小保留时间（秒）

# 场景重建延迟
SCENE_REBUILD_DELAY_MS = 80         # 场景重建延迟（毫秒）

# ==================== Windows 控制台抑制 ====================
@contextlib.contextmanager
def suppress_windows_subprocess_console():
    """Temporarily hide console windows spawned via subprocess on Windows.

    Some OCR dependencies may spawn helper processes during import/init; when the app runs as
    a GUI process this can show a brief black console window. We patch subprocess.Popen only
    within the OCR init scope to reduce side effects.
    """
    if platform.system() != "Windows":
        yield
        return
    try:
        import subprocess as _subprocess
    except Exception:
        yield
        return

    old_popen = getattr(_subprocess, "Popen", None)
    if old_popen is None or not isinstance(old_popen, type):
        yield
        return

    create_no_window = getattr(_subprocess, "CREATE_NO_WINDOW", 0x08000000)

    class _PopenNoConsole(old_popen):
        def __init__(self, *args, **kwargs):
            kwargs = dict(kwargs or {})
            try:
                kwargs["creationflags"] = int(kwargs.get("creationflags", 0)) | int(create_no_window)
            except Exception:
                pass
            # Best-effort: hide window via STARTUPINFO
            try:
                si = kwargs.get("startupinfo", None)
                if si is None:
                    si = _subprocess.STARTUPINFO()
                    kwargs["startupinfo"] = si
                si.dwFlags |= _subprocess.STARTF_USESHOWWINDOW
                si.wShowWindow = 0  # SW_HIDE
            except Exception:
                pass
            super().__init__(*args, **kwargs)

    _subprocess.Popen = _PopenNoConsole
    try:
        yield
    finally:
        try:
            _subprocess.Popen = old_popen
        except Exception:
            pass


def _try_use_pythonw_for_multiprocessing():
    """Best-effort: make multiprocessing children use pythonw.exe to avoid console flash on Windows."""
    if platform.system() != "Windows":
        return
    try:
        import multiprocessing as _mp
    except Exception:
        return

    exe = (sys.executable or "").strip()
    if not exe:
        return

    low = exe.lower()
    # If started via python.exe, children will be python.exe too (may flash a console).
    # Switch to pythonw.exe if present.
    try:
        if low.endswith("python.exe"):
            pyw = exe[:-10] + "pythonw.exe"
            if os.path.exists(pyw):
                _mp.set_executable(pyw)
    except Exception:
        pass


def parse_inpaint_api_urls(value):
    """Parse `inpaint_api_url` setting into a list of API endpoints.

    Supports:
    - a single URL string
    - multiple URLs separated by newlines/semicolon/comma
    - a list/tuple of URLs
    """
    if isinstance(value, (list, tuple)):
        out = []
        for v in value:
            s = str(v or "").strip()
            if s:
                out.append(s)
        return out

    s = str(value or "").strip()
    if not s:
        return []

    parts = [p.strip() for p in re.split(r"[;\n,]+", s) if p.strip()]

    # De-dup while preserving order.
    seen = set()
    out = []
    for p in parts:
        key = p.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
    return out

def _t_sys(zh: str, en: str = None) -> str:
    """Translate by system locale (used before app settings/UI language are loaded)."""
    try:
        loc = (QLocale.system().name() or "").lower()  # e.g. "zh_CN"
        if loc.startswith("zh"):
            return zh
    except Exception:
        pass
    return en if en is not None else zh

_UI_LANG = None  # "zh" / "en"; set by PPTCloneApp once settings are loaded.

def _t_global(zh: str, en: str = None) -> str:
    """Translate using the app UI language when available; fallback to system locale."""
    try:
        lang = (str(_UI_LANG or "") or "").strip().lower()
        if not lang or lang == "auto":
            lang = (QLocale.system().name() or "").lower()
        if lang.startswith("zh"):
            return zh
        return en if en is not None else zh
    except Exception:
        return en if en is not None else zh

# === 依赖检查 ===
_missing_qtawesome = False
try:
    import qtawesome as qta
except ImportError:
    _missing_qtawesome = True

# 注意：QApplication 必须在主入口处创建，不能在依赖检查时创建
# 否则会导致后续的 QApplication 实例无法正常使用

# === OCR/PPT 导出导入 ===
# 注意：为了让 PaddleX 缓存目录生效（PADDLE_PDX_CACHE_HOME），OCR 引擎必须在设置环境变量后再 import 初始化。
try:
    from ppt_export import PPTExporter
except ImportError:
    class PPTExporter:
        @classmethod
        def _scale_to_ppt_limit(cls, img_width, img_height):
            return int(img_width or 1), int(img_height or 1), 1.0

        @classmethod
        def presentation_size_for_images(cls, image_paths):
            return 1, 1

        def __init__(self, **kwargs): pass
        def add_image_with_text_boxes(self, *args): pass
        def save(self, path): return True

OCREngine = None  # 延迟 import（见 ensure_ocr_engine）

# ==================== 0. 样式表 (CSS) ====================
PPT_THEME_RED = "#D24726"   # PowerPoint 品牌红
PPT_RIBBON_BG = "#F3F3F3"   # 顶部背景浅灰
RIGHT_PANEL_W = 280         # 右侧属性面板宽度（包含滚动条空间，避免右边缘控件被裁剪）

GLOBAL_STYLE = f"""
    QMainWindow, QWidget {{
        background-color: #F8F8F8;
        font-family: "Microsoft YaHei", "Segoe UI";
        font-size: 12px;
        color: #333;
    }}
    
    /* === 1. 顶部 Tab 样式 === */
    QTabWidget::pane {{ border: none; border-bottom: 1px solid #CCC; }}
    QTabWidget::tab-bar {{ left: 10px; }}
    QTabBar::tab {{
        background: transparent; color: #444; padding: 6px 16px; 
        border: none; font-size: 13px; margin-bottom: 2px;
    }}
    QTabBar::tab:selected {{
        color: {PPT_THEME_RED}; font-weight: bold;
        border: 1px solid #DDD; border-bottom: none; background: {PPT_RIBBON_BG};
    }}

    /* === 2. 按钮样式 (仿 Office 悬停效果) === */
    QToolButton {{ border: 1px solid transparent; border-radius: 2px; background: transparent; }}
    QToolButton:hover {{ background-color: #CDE6F7; border: 1px solid #92C0E0; }}

    /* 分组下方的小标题 (灰色) */
    QLabel[cssClass="groupLabel"] {{
        color: #888; font-size: 11px; padding-bottom: 3px;
    }}

    /* === 3. 关键：银色分割线样式 === */
    QFrame[cssClass="RibbonSeparator"] {{
        background-color: #222; /* 小黑线分隔 */
        border: none;
    }}

    /* === 4. 底部状态栏 (去白底修复) === */
    QFrame#StatusBar {{
        background-color: {PPT_THEME_RED};
        min-height: 26px; max-height: 26px;
        border-top: 1px solid #B03010;
    }}
    
    /* 底部所有按钮默认透明 */
    #StatusBar QPushButton {{
        background-color: transparent; 
        color: white; border: none; border-radius: 0px;
        font-size: 11px; padding: 0 8px; font-weight: 500;
    }}
    #StatusBar QPushButton:hover {{ background-color: rgba(255,255,255,0.2); }}

    /* === 5. 滑块美化 === */
    QSlider {{ background: transparent; min-height: 20px; }}
    QSlider::groove:horizontal {{
        background: rgba(0, 0, 0, 0.25); /* 深红槽 */
        height: 4px; border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        background: white; width: 14px; height: 14px;
        margin: -5px 0; border-radius: 7px;
    }}

    /* === 6. 右侧面板（避免全屏时出现空白条/底部被挡） === */
    QScrollArea#RightPanelScroll {{
        background-color: #F3F3F3;
        border-left: 1px solid #DDD;
    }}
    QScrollArea#RightPanelScroll > QWidget {{
        background-color: #F3F3F3;
        border: none;
    }}
    QScrollArea#RightPanelScroll QWidget {{
        background-color: #F3F3F3;
    }}
    QWidget#RightPanel {{
        background-color: #F3F3F3;
    }}
    QWidget#RightPanel QPushButton {{
        padding: 6px 10px;
        background: white;
        border: 1px solid #CCC;
        border-radius: 4px;
        font-size: 11px;
    }}
    QWidget#RightPanel QPushButton:hover {{ background: #F0F0F0; border-color: #999; }}
    QWidget#RightPanel QCheckBox {{ font-size: 12px; padding: 2px; }}

    /* Ensure ribbon separators are visible even if dynamic-property selectors don't repolish */
    QFrame#RibbonSeparator {{
        background-color: #C9C9C9;
        border: none;
    }}
"""

# ==================== 文字颜色提取工具 ====================

def _imread_any(path: str):
    """Robust cv2 image read that also works for non-ASCII Windows paths."""
    return imread_any(path)


def _imwrite_any(path: str, image, params=None) -> bool:
    """Robust cv2 image write that also works for non-ASCII Windows paths."""
    return imwrite_any(path, image, params=params)


def _srgb_to_linear(c):
    c = float(c)
    if c <= 0.04045:
        return c / 12.92
    return ((c + 0.055) / 1.055) ** 2.4


def _rgb_to_lab(rgb):
    """Convert sRGB [0..255] to CIE Lab (D65). Returns (L,a,b) floats."""
    r, g, b = [max(0.0, min(255.0, float(v))) / 255.0 for v in rgb]
    r = _srgb_to_linear(r)
    g = _srgb_to_linear(g)
    b = _srgb_to_linear(b)

    # sRGB -> XYZ (D65)
    x = r * 0.4124564 + g * 0.3575761 + b * 0.1804375
    y = r * 0.2126729 + g * 0.7151522 + b * 0.0721750
    z = r * 0.0193339 + g * 0.1191920 + b * 0.9503041

    # Normalize by reference white (D65)
    xr = x / 0.95047
    yr = y / 1.00000
    zr = z / 1.08883

    def f(t):
        if t > 0.008856:
            return t ** (1.0 / 3.0)
        return (7.787 * t) + (16.0 / 116.0)

    fx = f(xr)
    fy = f(yr)
    fz = f(zr)

    L = (116.0 * fy) - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return (L, a, b)


def _ciede2000(lab1, lab2):
    """CIEDE2000 color difference for two Lab triples."""
    import math

    L1, a1, b1 = [float(v) for v in lab1]
    L2, a2, b2 = [float(v) for v in lab2]

    avg_L = (L1 + L2) / 2.0
    C1 = math.hypot(a1, b1)
    C2 = math.hypot(a2, b2)
    avg_C = (C1 + C2) / 2.0

    # G factor
    C7 = avg_C ** 7
    G = 0.5 * (1.0 - math.sqrt(C7 / (C7 + (25.0 ** 7))) if (C7 + (25.0 ** 7)) != 0 else 0.0)
    a1p = (1.0 + G) * a1
    a2p = (1.0 + G) * a2
    C1p = math.hypot(a1p, b1)
    C2p = math.hypot(a2p, b2)
    avg_Cp = (C1p + C2p) / 2.0

    def hp(ap, b):
        if ap == 0 and b == 0:
            return 0.0
        h = math.degrees(math.atan2(b, ap))
        return h + 360.0 if h < 0 else h

    h1p = hp(a1p, b1)
    h2p = hp(a2p, b2)

    dLp = L2 - L1
    dCp = C2p - C1p

    dhp = h2p - h1p
    if C1p * C2p == 0:
        dHp = 0.0
    else:
        if dhp > 180.0:
            dhp -= 360.0
        elif dhp < -180.0:
            dhp += 360.0
        dHp = 2.0 * math.sqrt(C1p * C2p) * math.sin(math.radians(dhp / 2.0))

    # Average hue
    if C1p * C2p == 0:
        avg_hp = h1p + h2p
    else:
        if abs(h1p - h2p) > 180.0:
            avg_hp = (h1p + h2p + 360.0) / 2.0 if (h1p + h2p) < 360.0 else (h1p + h2p - 360.0) / 2.0
        else:
            avg_hp = (h1p + h2p) / 2.0

    T = (
        1.0
        - 0.17 * math.cos(math.radians(avg_hp - 30.0))
        + 0.24 * math.cos(math.radians(2.0 * avg_hp))
        + 0.32 * math.cos(math.radians(3.0 * avg_hp + 6.0))
        - 0.20 * math.cos(math.radians(4.0 * avg_hp - 63.0))
    )

    d_ro = 30.0 * math.exp(-(((avg_hp - 275.0) / 25.0) ** 2))
    Rc = 2.0 * math.sqrt((avg_Cp ** 7) / ((avg_Cp ** 7) + (25.0 ** 7))) if avg_Cp != 0 else 0.0
    Sl = 1.0 + (0.015 * ((avg_L - 50.0) ** 2)) / math.sqrt(20.0 + ((avg_L - 50.0) ** 2))
    Sc = 1.0 + 0.045 * avg_Cp
    Sh = 1.0 + 0.015 * avg_Cp * T
    Rt = -math.sin(math.radians(2.0 * d_ro)) * Rc

    # Parametric factors
    Kl = Kc = Kh = 1.0
    dE = math.sqrt(
        (dLp / (Kl * Sl)) ** 2
        + (dCp / (Kc * Sc)) ** 2
        + (dHp / (Kh * Sh)) ** 2
        + Rt * (dCp / (Kc * Sc)) * (dHp / (Kh * Sh))
    )
    return float(dE)


def extract_text_color_from_region(image, rect_xywh, sample_ratio=0.3):
    """
    从图片指定区域提取文字的主要颜色（支持所有颜色）。

    算法思路：
    1. 裁剪文本框区域
    2. 使用 K-Means 聚类找出主要颜色（通常2-3种：背景+文字+可能的边缘）
    3. 背景通常是面积最大的颜色，文字是第二大或与背景对比度最高的颜色
    4. 如果 K-Means 失败，回退到边缘检测 + 颜色采样

    Args:
        image: numpy array (BGR格式) 或图片路径
        rect_xywh: [x, y, w, h] 文本框坐标
        sample_ratio: 采样比例，用于加速处理

    Returns:
        [r, g, b] 文字颜色，如果提取失败返回 None
    """
    try:
        # 如果是路径，读取图片
        if isinstance(image, str):
            image = _imread_any(image)
        if image is None:
            return None

        x, y, w, h = [int(v) for v in rect_xywh]
        H, W = image.shape[:2]

        # 边界检查
        x = max(0, min(x, W - 1))
        y = max(0, min(y, H - 1))
        w = max(1, min(w, W - x))
        h = max(1, min(h, H - y))

        # 裁剪区域
        crop = image[y:y+h, x:x+w]
        if crop.size == 0:
            return None

        # 先用“背景估计 + 前景核心像素”法，优先抓文字主体，避免被抗锯齿边缘/阴影/表格线带偏。
        result = _extract_color_core_contrast(crop)
        if result is not None:
            return result

        # 再用“背景/前景”法（对“白字黑描边/阴影”等也较稳），失败再回退到 K-Means/边缘/OTSU
        result = _extract_color_bg_foreground(crop)
        if result is not None:
            return result

        # 尝试使用 K-Means 方法
        result = _extract_color_kmeans(crop)
        if result is not None:
            return result

        # 回退到边缘检测方法
        result = _extract_color_edge_based(crop)
        if result is not None:
            return result

        # 最后回退到 Otsu 方法
        return _extract_color_otsu(crop)

    except Exception as e:
        logger.debug(f"提取文字颜色失败: {e}")
        return None


def quantize_text_color_basic(rgb):
    """将任意 RGB 颜色量化为常见基础色（用于 PPT 文本颜色），并返回 (rgb, name_zh)。

    使用 CIEDE2000（Lab 感知距离）在固定调色板中选最近颜色，比简单 HSV 区间更稳。
    """
    try:
        if not (isinstance(rgb, (list, tuple)) and len(rgb) == 3):
            return (None, None)
        r, g, b = [int(max(0, min(255, v))) for v in rgb]

        palette = [
            ([0, 0, 0], "黑"),
            ([255, 255, 255], "白"),
            ([255, 0, 0], "红"),
            ([255, 165, 0], "橙"),
            ([255, 255, 0], "黄"),
            ([0, 255, 0], "绿"),
            ([0, 255, 255], "青"),
            ([0, 0, 255], "蓝"),
            ([128, 0, 255], "紫"),
        ]

        lab = _rgb_to_lab([r, g, b])
        best = None
        best_d = None
        for prgb, name in palette:
            d = _ciede2000(lab, _rgb_to_lab(prgb))
            if best_d is None or d < best_d:
                best_d = d
                best = (prgb, name)
        return best if best is not None else (None, None)
    except Exception:
        return (None, None)


def normalize_box_text_color_fields(box):
    """Prefer the raw extracted text color; keep palette color only as metadata."""
    if not isinstance(box, dict):
        return
    raw = box.get("text_color_raw")
    if not (isinstance(raw, (list, tuple)) and len(raw) == 3):
        return
    raw_rgb = [int(max(0, min(255, v))) for v in raw]
    q_rgb, q_name = quantize_text_color_basic(raw_rgb)
    current = box.get("text_color")
    current_rgb = None
    if isinstance(current, (list, tuple)) and len(current) == 3:
        current_rgb = [int(max(0, min(255, v))) for v in current]

    # Migrate older OCR results that stored only the quantized color in text_color.
    if current_rgb is None or current_rgb == raw_rgb or (q_rgb and current_rgb == [int(v) for v in q_rgb]):
        box["text_color"] = list(raw_rgb)

    if q_rgb:
        box.setdefault("text_color_quantized", [int(v) for v in q_rgb])
        box.setdefault("text_color_name", q_name)
        try:
            key_map = {
                "黑": "black",
                "白": "white",
                "红": "red",
                "橙": "orange",
                "黄": "yellow",
                "绿": "green",
                "青": "cyan",
                "蓝": "blue",
                "紫": "purple",
            }
            box.setdefault("text_color_key", key_map.get(str(q_name), None))
        except Exception:
            pass


def should_auto_refresh_text_color(box):
    if not isinstance(box, dict):
        return False
    if "text_color_auto" in box:
        return bool(box.get("text_color_auto", False))
    if bool(box.get("text_color_manual", False)):
        return False
    return any(k in box for k in ("text_color_raw", "text_color_quantized", "text_color_key", "text_color_name"))


def _extract_color_kmeans(crop, n_clusters=3):
    """使用 K-Means 聚类提取文字颜色"""
    try:
        h, w = crop.shape[:2]
        pixels = crop.reshape(-1, 3).astype(np.float32)

        # 采样加速
        if len(pixels) > 5000:
            indices = np.random.choice(len(pixels), 5000, replace=False)
            pixels = pixels[indices]

        # K-Means 聚类
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
        _, labels, centers = cv2.kmeans(
            pixels, n_clusters, None, criteria, 3, cv2.KMEANS_PP_CENTERS
        )

        # 统计每个簇的像素数量
        unique, counts = np.unique(labels, return_counts=True)
        sorted_idx = np.argsort(-counts)  # 按数量降序

        # 背景通常是最大的簇
        bg_idx = sorted_idx[0]
        bg_color = centers[bg_idx]

        # 找与背景对比度最高的颜色作为文字颜色
        best_text_idx = None
        best_contrast = 0

        for idx in sorted_idx[1:]:
            color = centers[idx]
            # 计算颜色对比度（欧氏距离）
            contrast = np.sqrt(np.sum((color - bg_color) ** 2))
            if contrast > best_contrast:
                best_contrast = contrast
                best_text_idx = idx

        if best_text_idx is None or best_contrast < 30:
            return None

        text_color = centers[best_text_idx].astype(int)
        # BGR -> RGB
        return [int(text_color[2]), int(text_color[1]), int(text_color[0])]

    except Exception:
        return None


def _extract_color_edge_based(crop):
    """使用边缘检测提取文字颜色（文字边缘通常有明显的颜色）"""
    try:
        # Canny 边缘检测
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # 膨胀边缘以获取更多文字像素
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)

        # 获取边缘附近的像素颜色
        edge_pixels = crop[dilated > 0]
        if len(edge_pixels) < 10:
            return None

        # 采样
        if len(edge_pixels) > 1000:
            indices = np.random.choice(len(edge_pixels), 1000, replace=False)
            edge_pixels = edge_pixels[indices]

        # 中位数颜色
        median_color = np.median(edge_pixels, axis=0).astype(int)
        return [int(median_color[2]), int(median_color[1]), int(median_color[0])]

    except Exception:
        return None


def _estimate_border_bg_bgr(crop):
    try:
        h, w = crop.shape[:2]
        if h <= 1 or w <= 1:
            return None
        t = max(1, int(round(min(h, w) * 0.08)))
        t = min(t, max(1, min(h, w) // 3))
        top = crop[:t, :, :]
        bot = crop[-t:, :, :]
        lef = crop[:, :t, :]
        rig = crop[:, -t:, :]
        border = np.concatenate(
            [top.reshape(-1, 3), bot.reshape(-1, 3), lef.reshape(-1, 3), rig.reshape(-1, 3)],
            axis=0,
        )
        if border.size == 0:
            return None
        return np.median(border.astype(np.float32), axis=0).astype(np.float32)
    except Exception:
        return None


def _suppress_long_lines(mask_u8):
    try:
        mask = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
        if not mask.any():
            return mask_u8
        h, w = mask.shape[:2]
        if h < 6 or w < 6:
            return (mask * 255).astype(np.uint8)

        line_mask = np.zeros_like(mask, dtype=np.uint8)
        row_fill = mask.mean(axis=1)
        col_fill = mask.mean(axis=0)

        row_idx = np.where(row_fill >= 0.60)[0]
        col_idx = np.where(col_fill >= 0.60)[0]
        if row_idx.size:
            line_mask[row_idx, :] = 1
        if col_idx.size:
            line_mask[:, col_idx] = 1

        if line_mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            line_mask = cv2.dilate(line_mask * 255, kernel, iterations=1)
            cleaned = mask.copy()
            cleaned[line_mask > 0] = 0
            if int(cleaned.sum()) >= max(12, int(mask.sum() * 0.08)):
                return (cleaned * 255).astype(np.uint8)
        return (mask * 255).astype(np.uint8)
    except Exception:
        return mask_u8


def _extract_color_core_contrast(crop):
    """基于背景对比与文字核心像素提取颜色。

    目标：
    - 黑字/白字优先提取到真正的字芯，而不是浅灰抗锯齿边缘
    - 表格页里尽量忽略横线/竖线
    - 仍保留有色文字的识别能力
    """
    try:
        h, w = crop.shape[:2]
        if h <= 2 or w <= 2:
            return None

        bg_bgr = _estimate_border_bg_bgr(crop)
        if bg_bgr is None:
            return None

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.float32)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_bgr_u8 = np.uint8([[np.clip(bg_bgr, 0, 255)]])
        bg_lab = cv2.cvtColor(bg_bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
        dist = np.sqrt(np.sum((lab - bg_lab) * (lab - bg_lab), axis=2))

        dmin = float(np.min(dist))
        dmax = float(np.max(dist))
        if dmax <= dmin + 1e-6:
            return None

        norm = np.uint8(np.clip((dist - dmin) * (255.0 / (dmax - dmin)), 0, 255))
        thr_u8, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thr = dmin + (float(thr_u8) / 255.0) * (dmax - dmin)
        thr = float(max(10.0, min(42.0, thr)))

        fg = (dist >= thr).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8), iterations=1)
        fg = _suppress_long_lines(fg)
        fg_bool = fg > 0
        if int(fg_bool.sum()) < 12:
            return None

        bg_luma = float(0.114 * float(bg_bgr[0]) + 0.587 * float(bg_bgr[1]) + 0.299 * float(bg_bgr[2]))
        delta = gray - bg_luma
        dark_strength = float(np.sum(np.clip(-delta[fg_bool], 0.0, None)))
        light_strength = float(np.sum(np.clip(delta[fg_bool], 0.0, None)))
        prefer_dark = bool(dark_strength >= light_strength)

        fg_vals = gray[fg_bool]
        lum_margin = max(10.0, float(np.std(fg_vals)) * 0.35 + 6.0) if fg_vals.size else 10.0
        if prefer_dark:
            polarity_mask = fg_bool & (gray <= (bg_luma - lum_margin))
            if int(polarity_mask.sum()) < max(8, int(fg_bool.sum() * 0.06)):
                polarity_mask = fg_bool & (gray <= (bg_luma - 4.0))
        else:
            polarity_mask = fg_bool & (gray >= (bg_luma + lum_margin))
            if int(polarity_mask.sum()) < max(8, int(fg_bool.sum() * 0.06)):
                polarity_mask = fg_bool & (gray >= (bg_luma + 4.0))

        chosen_mask = polarity_mask if int(polarity_mask.sum()) >= 8 else fg_bool
        if int(chosen_mask.sum()) < 8:
            return None

        core_mask = chosen_mask
        dt = cv2.distanceTransform((chosen_mask.astype(np.uint8) * 255), cv2.DIST_L2, 3)
        maxd = float(np.max(dt)) if dt is not None else 0.0
        if maxd > 1.0:
            cand_core = dt >= max(1.0, maxd * 0.30)
            if int(cand_core.sum()) >= max(6, int(chosen_mask.sum() * 0.05)):
                core_mask = cand_core

        ys, xs = np.nonzero(core_mask)
        if len(xs) < 6:
            ys, xs = np.nonzero(chosen_mask)
        if len(xs) < 6:
            return None

        pixels = crop[ys, xs].astype(np.float32)
        pix_gray = gray[ys, xs]
        pix_dist = dist[ys, xs]
        if prefer_dark:
            polarity_strength = np.clip(bg_luma - pix_gray, 0.0, None)
        else:
            polarity_strength = np.clip(pix_gray - bg_luma, 0.0, None)
        weights = np.maximum(pix_dist, 1.0) * np.power(np.maximum(polarity_strength, 1.0), 1.15)
        if weights.size >= 8:
            keep_thr = float(np.percentile(weights, 55))
            keep = weights >= keep_thr
            if int(np.count_nonzero(keep)) >= 6:
                pixels = pixels[keep]

        vals = np.median(pixels, axis=0).astype(int)
        return [int(vals[2]), int(vals[1]), int(vals[0])]
    except Exception:
        return None


def _extract_color_bg_foreground(crop):
    """基于“边缘背景估计 -> 前景分割 ->（可选）腐蚀去描边”的文字颜色提取。

    目的：解决“白字黑描边/阴影”场景，避免只取到描边的黑色。
    """
    try:
        h, w = crop.shape[:2]
        if h <= 2 or w <= 2:
            return None

        # 1) 估计背景色：取边缘像素的中位数（更稳，避免被文字覆盖）
        t = max(1, int(round(min(h, w) * 0.08)))  # ~8% border
        t = min(t, max(1, min(h, w) // 3))
        top = crop[:t, :, :]
        bot = crop[-t:, :, :]
        lef = crop[:, :t, :]
        rig = crop[:, -t:, :]
        border = np.concatenate(
            [top.reshape(-1, 3), bot.reshape(-1, 3), lef.reshape(-1, 3), rig.reshape(-1, 3)],
            axis=0,
        )
        if border.size == 0:
            return None
        bg_bgr = np.median(border, axis=0).astype(np.float32)

        # 2) 用 Lab 距离做前景分割（比 RGB 更符合感知）
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_bgr_u8 = np.uint8([[np.clip(bg_bgr, 0, 255)]])
        bg_lab = cv2.cvtColor(bg_bgr_u8, cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
        d = lab - bg_lab  # (h,w,3)
        dist = np.sqrt(np.sum(d * d, axis=2))  # (h,w)

        # 自适应阈值：对不同背景纹理/压缩噪声更稳
        thr = 18.0
        try:
            dmin = float(np.min(dist))
            dmax = float(np.max(dist))
            if dmax > dmin + 1e-6:
                norm = np.uint8(np.clip((dist - dmin) * (255.0 / (dmax - dmin)), 0, 255))
                # Otsu on distances (as 8-bit) -> map back to float threshold.
                thr_u8, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                thr = dmin + (float(thr_u8) / 255.0) * (dmax - dmin)
                thr = float(max(10.0, min(35.0, thr)))
        except Exception:
            thr = 18.0

        fg = (dist >= thr).astype(np.uint8) * 255

        # 3) 轻微形态学清理
        kernel = np.ones((3, 3), np.uint8)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, kernel, iterations=1)

        # 4) 距离变换取“核心像素”：更数学化地去掉描边/阴影/抗锯齿边缘，只保留填充内部
        core = None
        try:
            dt = cv2.distanceTransform(fg, cv2.DIST_L2, 3)
            maxd = float(np.max(dt)) if dt is not None else 0.0
            if maxd > 0:
                core_thr = max(0.8, maxd * 0.55)
                core = np.uint8(dt >= core_thr) * 255
        except Exception:
            core = None

        def _bgr_luma(c):
            try:
                return float(0.114 * float(c[0]) + 0.587 * float(c[1]) + 0.299 * float(c[2]))
            except Exception:
                return 0.0

        def _pick_color_from_mask(mask, prefer_core=False):
            pts = crop[mask > 0]
            if pts is None or len(pts) < 20:
                return None
            if len(pts) > 5000:
                idx = np.random.choice(len(pts), 5000, replace=False)
                pts = pts[idx]

            if prefer_core:
                med = np.median(pts.astype(np.float32), axis=0).astype(int)
                return [int(med[2]), int(med[1]), int(med[0])]

            # 2-cluster to separate fill vs outline/shadow when both remain.
            pixels = pts.astype(np.float32)
            if len(pixels) < 50:
                med = np.median(pixels, axis=0).astype(int)
                return [int(med[2]), int(med[1]), int(med[0])]

            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5)
            _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 3, cv2.KMEANS_PP_CENTERS)
            labels = labels.reshape(-1)
            cnt0 = int(np.sum(labels == 0))
            cnt1 = int(np.sum(labels == 1))

            # Prefer the cluster that is more opposite to the background, while still
            # giving some weight to coverage so tiny noise doesn't win.
            c0 = centers[0]
            c1 = centers[1]
            d0 = float(np.linalg.norm(c0 - bg_bgr))
            d1 = float(np.linalg.norm(c1 - bg_bgr))
            bg_luma = _bgr_luma(bg_bgr)
            l0 = _bgr_luma(c0)
            l1 = _bgr_luma(c1)
            max_cnt = max(1, cnt0, cnt1)

            def _score(dist, cnt, lum):
                score = float(dist) * 1.4 + abs(float(lum) - bg_luma) * 0.45 + (float(cnt) / float(max_cnt)) * 6.0
                if bg_luma >= 170.0:
                    score += max(0.0, bg_luma - float(lum)) * 0.30
                elif bg_luma <= 85.0:
                    score += max(0.0, float(lum) - bg_luma) * 0.30
                return score

            pick = 0 if _score(d0, cnt0, l0) >= _score(d1, cnt1, l1) else 1
            c = centers[pick].astype(int)
            return [int(c[2]), int(c[1]), int(c[0])]

        # Prefer core pixels (best for outlined/shadowed text). If too small, fall back to full foreground.
        col = _pick_color_from_mask(core, prefer_core=True) if core is not None else None
        if col is not None:
            return col
        return _pick_color_from_mask(fg)

    except Exception:
        return None


def _extract_color_otsu(crop):
    """使用 Otsu 阈值分割提取文字颜色（回退方法）"""
    try:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        white_pixels = np.sum(binary == 255)
        black_pixels = np.sum(binary == 0)

        # 文字通常占少数像素
        mask = binary == 255 if white_pixels < black_pixels else binary == 0

        text_pixels = crop[mask]
        if len(text_pixels) == 0:
            return None

        if len(text_pixels) > 1000:
            indices = np.random.choice(len(text_pixels), 1000, replace=False)
            text_pixels = text_pixels[indices]

        median_color = np.median(text_pixels, axis=0).astype(int)
        return [int(median_color[2]), int(median_color[1]), int(median_color[0])]

    except Exception:
        return None


# ==================== 核心组件封装 ====================

class OCRThread(QThread):
    """OCR识别线程"""
    # (image_path, results, roi_used_or_none)；roi 用于“仅更新选区内框”合并逻辑
    finished = Signal(str, list, object)
    progress = Signal(int, int)   # (current, total)
    all_done = Signal()           # emitted once when the run loop ends
    error = Signal(str)           # emitted on first fatal OCR error

    def __init__(self, ocr_engine, images, scaled_images, roi_by_image=None, roi_temp_dir=None):
        super().__init__()
        self.ocr_engine = ocr_engine
        self.images = images
        self.scaled_images = scaled_images
        self.roi_by_image = roi_by_image or {}
        self.roi_temp_dir = roi_temp_dir

    @staticmethod
    def _parse_roi(roi_xywh):
        if not (isinstance(roi_xywh, (list, tuple)) and len(roi_xywh) == 4):
            return None
        try:
            x, y, w, h = [int(v) for v in roi_xywh]
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None
        return [x, y, w, h]

    def run(self):
        import time
        from PIL import Image

        for i, img_path in enumerate(self.images):
            if self.isInterruptionRequested():
                break

            scaled_path = self.scaled_images.get(img_path, img_path) or img_path
            roi_orig = self._parse_roi((self.roi_by_image or {}).get(img_path))

            # 如果设置了 ROI，就只识别选区（在缩放图上裁剪，结果坐标仍在“缩放图坐标系”里，主线程再还原到原图）。
            if roi_orig:
                orig_w = orig_h = None
                try:
                    with Image.open(img_path) as _im:
                        orig_w, orig_h = _im.size
                except Exception:
                    orig_w = orig_h = None

                try:
                    x, y, w, h = [int(v) for v in roi_orig]
                except Exception:
                    x = y = w = h = None

                if orig_w and orig_h and x is not None:
                    # Clamp ROI to original bounds.
                    x = max(0, min(x, orig_w - 1))
                    y = max(0, min(y, orig_h - 1))
                    w = max(1, min(w, orig_w - x))
                    h = max(1, min(h, orig_h - y))
                    roi_orig = [x, y, w, h]

                try:
                    img = _imread_any(scaled_path)
                    if img is None:
                        img = _imread_any(img_path)
                        scaled_path = img_path
                    if img is not None and x is not None:
                        Hs, Ws = img.shape[:2]
                        xs, ys, ws, hs = x, y, w, h
                        if scaled_path != img_path and orig_w and orig_h:
                            # Map ROI from original coords -> scaled coords.
                            sx = float(Ws) / max(1.0, float(orig_w))
                            sy = float(Hs) / max(1.0, float(orig_h))
                            xs = int(round(float(x) * sx))
                            ys = int(round(float(y) * sy))
                            ws = int(round(float(w) * sx))
                            hs = int(round(float(h) * sy))

                        xs = max(0, min(int(xs), Ws - 1))
                        ys = max(0, min(int(ys), Hs - 1))
                        ws = max(1, min(int(ws), Ws - xs))
                        hs = max(1, min(int(hs), Hs - ys))

                        # Too small -> fallback to full image OCR (avoid accidental tiny drags).
                        if ws >= 5 and hs >= 5:
                            crop = img[ys : ys + hs, xs : xs + ws]
                            out_dir = str(self.roi_temp_dir or tempfile.gettempdir())
                            try:
                                os.makedirs(out_dir, exist_ok=True)
                            except Exception:
                                pass
                            ts = int(time.time() * 1000)
                            crop_path = build_asset_path(
                                out_dir,
                                "roi_ocr",
                                img_path,
                                suffix=f"{ts}_{xs}_{ys}_{ws}x{hs}",
                                ext=".png",
                            )
                            if not _imwrite_any(crop_path, crop):
                                raise RuntimeError(_t_global("无法写入 ROI 临时图片", "Failed to write ROI temp image"))

                            results = self.ocr_engine.recognize(crop_path) or []
                            # Offset rects back to (scaled) full-image coordinates.
                            for r in results:
                                if not isinstance(r, dict):
                                    continue
                                rect = r.get("rect")
                                if isinstance(rect, (list, tuple)) and len(rect) == 4:
                                    try:
                                        rx, ry, rw, rh = [int(v) for v in rect]
                                        r["rect"] = [rx + xs, ry + ys, rw, rh]
                                    except Exception:
                                        pass

                            self.finished.emit(img_path, results, roi_orig)
                            self.progress.emit(i + 1, len(self.images))
                            continue
                except Exception:
                    # Fall back to full-image OCR below.
                    pass

            try:
                results = self.ocr_engine.recognize(scaled_path)
            except Exception as e:
                # Avoid "Error calling Python override of QThread::run()" and let the UI show the error.
                try:
                    self.error.emit(f"OCR 识别失败: {os.path.basename(scaled_path)}\n{type(e).__name__}: {e}")
                except Exception:
                    pass
                break
            self.finished.emit(img_path, results, None)
            self.progress.emit(i + 1, len(self.images))
        self.all_done.emit()


class OCRRoiThread(QThread):
    """OCR only within a ROI; results are emitted in original-image coordinates."""
    finished = Signal(str, list)  # (image_path, results)
    error = Signal(str)

    def __init__(self, ocr_engine, image_path: str, roi_xywh, out_dir: str):
        super().__init__()
        self.ocr_engine = ocr_engine
        self.image_path = str(image_path or "")
        self.roi = roi_xywh
        self.out_dir = str(out_dir or "")

    def run(self):
        try:
            if not self.image_path or not os.path.exists(self.image_path):
                raise RuntimeError(_t_global("图片不存在", "Image not found"))
            if not (isinstance(self.roi, (list, tuple)) and len(self.roi) == 4):
                raise RuntimeError(_t_global("未设置选区", "ROI is not set"))
            x, y, w, h = [int(v) for v in self.roi]
            if w <= 0 or h <= 0:
                raise RuntimeError(_t_global("选区无效", "Invalid ROI"))

            img = _imread_any(self.image_path)
            if img is None:
                raise RuntimeError(_t_global("无法读取图片", "Failed to read image"))

            H, W = img.shape[:2]
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            w = max(1, min(w, W - x))
            h = max(1, min(h, H - y))

            crop = img[y : y + h, x : x + w]
            out_path = build_asset_path(
                self.out_dir,
                "roi_ocr",
                self.image_path,
                suffix=f"{x}_{y}_{w}x{h}",
                ext=".png",
            )
            if not _imwrite_any(out_path, crop):
                raise RuntimeError(_t_global("无法写入 ROI 临时图片", "Failed to write ROI temp image"))

            results = self.ocr_engine.recognize(out_path) or []
            # Offset rects back to original coordinates
            for r in results:
                if not isinstance(r, dict):
                    continue
                rect = r.get("rect")
                if isinstance(rect, (list, tuple)) and len(rect) == 4:
                    try:
                        rx, ry, rw, rh = [int(v) for v in rect]
                        r["rect"] = [rx + x, ry + y, rw, rh]
                    except Exception:
                        pass

            self.finished.emit(self.image_path, results)
        except Exception as e:
            self.error.emit(str(e))


class InpaintThread(QThread):
    """Text removal with explicit fill / IOPaint modes and per-box overrides."""
    progress = Signal(int, int)      # (current, total)
    finished_one = Signal(str, str)  # (src_path, out_path)
    error = Signal(str)
    all_done = Signal(bool)          # (canceled)

    STRATEGY_LOCAL_FILL = "local_fill"
    STRATEGY_LOCAL_CV2 = "local_cv2"
    STRATEGY_REMOTE = "remote"
    MODE_AUTO = "auto"
    MODE_FILL = "fill"
    MODE_REMOTE = "remote"
    RUN_SMART = "smart"
    RUN_FILL = "fill"
    RUN_REMOTE = "remote"

    def __init__(
        self,
        images,
        box_data,
        out_dir,
        api_url,
        box_padding=6,
        crop_padding=128,
        input_image_by_src=None,
        roi_by_image=None,
        timeout_sec=120,
        run_mode="smart",
        fill_pad_x=10,
        fill_pad_y=6,
        remote_pad_x=8,
        remote_pad_y=4,
    ):
        super().__init__()
        self.images = list(images or [])
        self.box_data = box_data or {}
        self.out_dir = out_dir
        self.api_urls = parse_inpaint_api_urls(api_url)
        # Keep a single-url field for backward-compatible logs/messages.
        self.api_url = self.api_urls[0] if self.api_urls else ""
        self.box_padding = int(box_padding or 0)
        self.crop_padding = int(crop_padding or 0)
        # If provided, use this mapping to decide which file to inpaint for a given logical slide image.
        # This enables iterative inpaint: run again on the already-inpainted preview image.
        self.input_image_by_src = input_image_by_src or {}
        self.roi_by_image = roi_by_image or {}
        self.timeout_sec = int(timeout_sec or 120)
        self.run_mode = self._normalize_run_mode(run_mode)
        self.fill_pad_x = max(0, int(fill_pad_x or 0))
        self.fill_pad_y = max(0, int(fill_pad_y or 0))
        self.remote_pad_x = max(0, int(remote_pad_x or 0))
        self.remote_pad_y = max(0, int(remote_pad_y or 0))
        self.results = []  # list[(src, out)]

    @classmethod
    def _normalize_box_clean_mode(cls, value):
        mode = str(value or "").strip().lower()
        if mode in {"fill", "solid", "solid_fill", "color_fill"}:
            return cls.MODE_FILL
        if mode in {"remote", "iopaint", "api"}:
            return cls.MODE_REMOTE
        return cls.MODE_AUTO

    @classmethod
    def _normalize_run_mode(cls, value):
        mode = str(value or "").strip().lower()
        if mode in {cls.RUN_FILL, cls.MODE_FILL, "solid", "solid_fill", "color_fill"}:
            return cls.RUN_FILL
        if mode in {cls.RUN_REMOTE, cls.MODE_REMOTE, "iopaint", "api"}:
            return cls.RUN_REMOTE
        return cls.RUN_SMART

    @staticmethod
    def _extract_rect(box):
        if not isinstance(box, dict):
            return None
        rect = box.get("rect")
        if not (isinstance(rect, (list, tuple)) and len(rect) == 4):
            return None
        try:
            x, y, w, h = [int(v) for v in rect]
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None
        return [x, y, w, h]

    @classmethod
    def _extract_polygon(cls, box):
        if not isinstance(box, dict):
            return None
        bbox = box.get("bbox")
        if isinstance(bbox, np.ndarray):
            try:
                bbox = bbox.tolist()
            except Exception:
                bbox = None

        points = []
        if isinstance(bbox, (list, tuple)):
            for pt in bbox:
                if not (isinstance(pt, (list, tuple)) and len(pt) >= 2):
                    points = []
                    break
                try:
                    px = int(round(float(pt[0])))
                    py = int(round(float(pt[1])))
                except Exception:
                    points = []
                    break
                points.append([px, py])

        if len(points) >= 3:
            return points

        rect = cls._extract_rect(box)
        if rect is None:
            return None
        x, y, w, h = rect
        return [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]

    @staticmethod
    def _intersects_roi(rect, roi):
        if roi is None:
            return True
        try:
            x, y, w, h = rect
            rx, ry, rw, rh = roi
        except Exception:
            return True
        return not (x + w <= rx or x >= rx + rw or y + h <= ry or y >= ry + rh)

    @classmethod
    def _create_mask(cls, img_size, boxes, padding, prefer_rect=False, rect_expand_x=0, rect_expand_y=0):
        W, H = img_size
        mask = np.zeros((max(1, int(H)), max(1, int(W))), dtype=np.uint8)
        extra_x = max(0, int(rect_expand_x or 0))
        extra_y = max(0, int(rect_expand_y or 0))
        for b in boxes or []:
            poly = None if prefer_rect else cls._extract_polygon(b)
            if poly and len(poly) >= 3:
                arr = np.array(poly, dtype=np.int32)
                arr[:, 0] = np.clip(arr[:, 0], 0, max(0, W - 1))
                arr[:, 1] = np.clip(arr[:, 1], 0, max(0, H - 1))
                if int(abs(cv2.contourArea(arr))) > 0:
                    cv2.fillPoly(mask, [arr], 255)
                    continue

            rect = cls._extract_rect(b)
            if rect is None:
                continue
            x, y, w, h = rect
            x1 = max(0, x - extra_x)
            y1 = max(0, y - extra_y)
            x2 = min(W, x + w + extra_x)
            y2 = min(H, y + h + extra_y)
            cv2.rectangle(mask, (x1, y1), (max(x1, x2 - 1), max(y1, y2 - 1)), 255, thickness=-1)

        pad = max(0, int(padding or 0))
        if pad > 0 and mask.any():
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
            mask = cv2.dilate(mask, kernel, iterations=1)
        from PIL import Image
        return Image.fromarray(mask, mode="L")

    @staticmethod
    def _limit_mask_to_roi(mask_pil, roi, img_size):
        if not (isinstance(roi, (list, tuple)) and len(roi) == 4):
            return mask_pil
        try:
            rx, ry, rw, rh = [int(v) for v in roi]
        except Exception:
            return mask_pil
        if rw <= 0 or rh <= 0:
            return mask_pil

        W, H = img_size
        rx = max(0, min(rx, W - 1))
        ry = max(0, min(ry, H - 1))
        rw = max(1, min(rw, W - rx))
        rh = max(1, min(rh, H - ry))

        roi_box = (rx, ry, rx + rw, ry + rh)
        from PIL import Image
        m2 = Image.new("L", img_size, 0)
        m2.paste(mask_pil.crop(roi_box), (rx, ry))
        return m2

    @staticmethod
    def _rects_touch(rect_a, rect_b, gap):
        try:
            ax, ay, aw, ah = rect_a
            bx, by, bw, bh = rect_b
        except Exception:
            return False
        g = max(0, int(gap or 0))
        return not (
            (ax + aw + g) < bx
            or (bx + bw + g) < ax
            or (ay + ah + g) < by
            or (by + bh + g) < ay
        )

    @classmethod
    def _cluster_boxes(cls, boxes, merge_gap):
        valid = []
        for b in boxes or []:
            rect = cls._extract_rect(b)
            if rect is None:
                continue
            valid.append((rect, b))
        n = len(valid)
        if n <= 1:
            return [[b] for _, b in valid]

        parent = list(range(n))

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra = find(a)
            rb = find(b)
            if ra != rb:
                parent[rb] = ra

        for i in range(n):
            rect_i, _ = valid[i]
            for j in range(i + 1, n):
                rect_j, _ = valid[j]
                if cls._rects_touch(rect_i, rect_j, merge_gap):
                    union(i, j)

        groups = {}
        for idx, (rect, box) in enumerate(valid):
            root = find(idx)
            groups.setdefault(root, []).append((rect[1], rect[0], box))

        clusters = []
        for items in groups.values():
            items.sort(key=lambda t: (t[0], t[1]))
            clusters.append([box for _, _, box in items])
        clusters.sort(
            key=lambda grp: (
                cls._extract_rect(grp[0])[1] if cls._extract_rect(grp[0]) else 0,
                cls._extract_rect(grp[0])[0] if cls._extract_rect(grp[0]) else 0,
            )
        )
        return clusters

    @staticmethod
    def _crop_from_mask(image_pil, mask_pil, crop_padding):
        bbox = mask_pil.getbbox()
        if not bbox:
            return None

        left, top, right, bottom = bbox
        W, H = image_pil.size
        pad = int(crop_padding or 0)
        x1 = max(0, int(left) - pad)
        y1 = max(0, int(top) - pad)
        x2 = min(W, int(right) + pad)
        y2 = min(H, int(bottom) + pad)
        crop_box = (x1, y1, x2, y2)
        crop_img = image_pil.crop(crop_box)
        crop_mask = mask_pil.crop(crop_box)
        return crop_box, crop_img, crop_mask

    @staticmethod
    def _analysis_ring(mask_u8):
        base = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
        if not base.any():
            return base.astype(bool)
        h, w = base.shape[:2]
        pad = max(4, min(24, int(round(max(h, w) * 0.08))))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pad * 2 + 1, pad * 2 + 1))
        outer = cv2.dilate(base * 255, kernel, iterations=1)
        ring = (outer > 0) & (~base.astype(bool))
        if int(ring.sum()) < 96:
            ring = ~base.astype(bool)
        return ring

    @staticmethod
    def _fill_sample_ring(mask_u8):
        base = (np.asarray(mask_u8, dtype=np.uint8) > 0).astype(np.uint8)
        if not base.any():
            return base.astype(bool)

        h, w = base.shape[:2]
        inner_pad = max(2, min(8, int(round(min(h, w) * 0.06))))
        inner_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (inner_pad * 2 + 1, inner_pad * 2 + 1))
        eroded = cv2.erode(base * 255, inner_kernel, iterations=1)
        inner_ring = (base > 0) & (~(eroded > 0))
        if int(inner_ring.sum()) >= 48:
            return inner_ring

        outer_pad = max(2, min(6, int(round(min(h, w) * 0.04))))
        outer_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (outer_pad * 2 + 1, outer_pad * 2 + 1))
        dilated = cv2.dilate(base * 255, outer_kernel, iterations=1)
        outer_ring = (dilated > 0) & (~(base > 0))
        ring = inner_ring | outer_ring
        if int(ring.sum()) >= 48:
            return ring
        return outer_ring if int(outer_ring.sum()) >= 16 else (~(base > 0))

    @classmethod
    def _fit_plane_residual(cls, img_rgb, sample_mask):
        ys, xs = np.nonzero(sample_mask)
        if len(xs) < 32:
            return None
        if len(xs) > 5000:
            step = max(1, len(xs) // 5000)
            xs = xs[::step]
            ys = ys[::step]
        xs_f = xs.astype(np.float64)
        ys_f = ys.astype(np.float64)
        A = np.stack([xs_f, ys_f, np.ones_like(xs_f, dtype=np.float64)], axis=1)
        residuals = []
        for c in range(3):
            vals = img_rgb[ys, xs, c].astype(np.float64)
            coeff, *_ = np.linalg.lstsq(A, vals, rcond=None)
            if not np.all(np.isfinite(coeff)):
                return None
            pred = coeff[0] * xs_f + coeff[1] * ys_f + coeff[2]
            if not np.all(np.isfinite(pred)):
                return None
            residuals.append(float(np.mean(np.abs(pred - vals))))
        if not residuals:
            return None
        return float(sum(residuals) / len(residuals))

    @classmethod
    def _analyze_crop(cls, crop_img, crop_mask):
        img_rgb = np.asarray(crop_img, dtype=np.uint8)
        mask_u8 = np.asarray(crop_mask, dtype=np.uint8)
        sample_mask = cls._analysis_ring(mask_u8)
        if int(sample_mask.sum()) < 64:
            return {
                "strategy": cls.STRATEGY_REMOTE,
                "edge_density": 1.0,
                "texture": 999.0,
                "color_std": 999.0,
                "plane_residual": 999.0,
            }

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150) > 0
        edge_density = float(edges[sample_mask].mean()) if sample_mask.any() else 1.0

        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        texture = float(np.mean(np.abs(gray.astype(np.float32) - blur.astype(np.float32))[sample_mask]))

        sample_pixels = img_rgb[sample_mask]
        color_std = float(np.mean(np.std(sample_pixels.astype(np.float32), axis=0))) if sample_pixels.size else 999.0
        color_range = 999.0
        if sample_pixels.size:
            try:
                p95 = np.percentile(sample_pixels.astype(np.float32), 95, axis=0)
                p05 = np.percentile(sample_pixels.astype(np.float32), 5, axis=0)
                color_range = float(np.mean(np.abs(p95 - p05)))
            except Exception:
                color_range = 999.0

        plane_residual = cls._fit_plane_residual(img_rgb, sample_mask)
        plane_residual = float(plane_residual) if plane_residual is not None else 999.0

        # "单色覆盖" should only be used on genuinely flat backgrounds.
        # Keep the gate strict so light gradients / shadows / tinted panels fall back to IOPaint.
        if plane_residual <= 2.0 and edge_density <= 0.010 and texture <= 1.8 and color_std <= 6.0 and color_range <= 8.0:
            strategy = cls.STRATEGY_LOCAL_FILL
        else:
            strategy = cls.STRATEGY_REMOTE

        return {
            "strategy": strategy,
            "edge_density": edge_density,
            "texture": texture,
            "color_std": color_std,
            "color_range": color_range,
            "plane_residual": plane_residual,
        }

    @staticmethod
    def _extract_runs_1d(mask_line):
        runs = []
        start = None
        for idx, val in enumerate(mask_line):
            if bool(val):
                if start is None:
                    start = idx
            elif start is not None:
                runs.append((start, idx - 1))
                start = None
        if start is not None:
            runs.append((start, len(mask_line) - 1))
        return runs

    @staticmethod
    def _sample_median_color(img_rgb, sample_mask):
        try:
            pixels = img_rgb[np.asarray(sample_mask, dtype=bool)]
        except Exception:
            return None
        if pixels.size == 0:
            return None
        vals = np.median(pixels.astype(np.float32), axis=0)
        return tuple(int(np.clip(v, 0, 255)) for v in vals[:3])

    @classmethod
    def _estimate_fill_color(cls, img_rgb, sample_mask):
        try:
            pixels = img_rgb[np.asarray(sample_mask, dtype=bool)]
        except Exception:
            pixels = None
        if pixels is None or pixels.size == 0:
            return None
        pixels = pixels.reshape(-1, 3).astype(np.uint8)
        if len(pixels) > 12000:
            step = max(1, len(pixels) // 12000)
            pixels = pixels[::step]
        # Quantize first, then take the median inside the dominant bucket.
        quant = (pixels // 16).astype(np.uint8)
        try:
            uniq, counts = np.unique(quant, axis=0, return_counts=True)
        except Exception:
            uniq = counts = None
        if uniq is None or counts is None or len(uniq) == 0:
            vals = np.median(pixels.astype(np.float32), axis=0)
            return tuple(int(np.clip(v, 0, 255)) for v in vals[:3])
        bucket = uniq[int(np.argmax(counts))]
        chosen = pixels[np.all(quant == bucket, axis=1)]
        if chosen.size == 0:
            chosen = pixels
        vals = np.median(chosen.astype(np.float32), axis=0)
        return tuple(int(np.clip(v, 0, 255)) for v in vals[:3])

    @classmethod
    def _register_overlay(cls, overlays, orientation, pos, start, end, color, thickness):
        if color is None or end <= start:
            return
        pos = int(pos)
        start = int(start)
        end = int(end)
        thickness = max(1, int(thickness or 1))
        for item in overlays:
            if item.get("orientation") != orientation:
                continue
            if abs(int(item.get("pos", 0)) - pos) > max(int(item.get("thickness", 1)), thickness):
                continue
            if end < int(item.get("start", 0)) - 6 or start > int(item.get("end", 0)) + 6:
                continue
            item["pos"] = int(round((int(item.get("pos", pos)) + pos) / 2.0))
            item["start"] = min(int(item.get("start", start)), start)
            item["end"] = max(int(item.get("end", end)), end)
            item["thickness"] = max(int(item.get("thickness", 1)), thickness)
            prev_color = item.get("color") or color
            item["color"] = tuple(int(round((int(prev_color[i]) + int(color[i])) / 2.0)) for i in range(3))
            return
        overlays.append(
            {
                "orientation": orientation,
                "pos": pos,
                "start": start,
                "end": end,
                "color": tuple(int(v) for v in color),
                "thickness": thickness,
            }
        )

    @classmethod
    def _collect_line_overlays(cls, crop_img, crop_mask, analysis):
        img_rgb = np.asarray(crop_img, dtype=np.uint8)
        mask_u8 = np.asarray(crop_mask, dtype=np.uint8)
        mask_bool = mask_u8 > 0
        h, w = mask_bool.shape[:2]
        if h < 8 or w < 8 or int(mask_bool.sum()) < 24:
            return []

        strategy = str((analysis or {}).get("strategy") or cls.STRATEGY_REMOTE)
        edge_density = float((analysis or {}).get("edge_density", 1.0))
        if strategy == cls.STRATEGY_REMOTE and edge_density > 0.16:
            return []

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        sample_mask = cls._analysis_ring(mask_u8)
        sample_vals = gray[sample_mask] if sample_mask.any() else gray.reshape(-1)
        if sample_vals.size < 32:
            return []

        median = float(np.median(sample_vals))
        spread = max(12.0, float(np.std(sample_vals)) * 0.9 + 6.0)
        dark = (gray <= max(0.0, median - spread)).astype(np.uint8) * 255
        light = (gray >= min(255.0, median + spread)).astype(np.uint8) * 255
        dark[mask_bool] = 0
        light[mask_bool] = 0
        bg_color = np.median(img_rgb[sample_mask].astype(np.float32), axis=0) if sample_mask.any() else np.array([255.0, 255.0, 255.0], dtype=np.float32)

        min_h_len = max(14, w // 8)
        min_v_len = max(14, h // 8)
        hk = cv2.getStructuringElement(cv2.MORPH_RECT, (min_h_len, 1))
        vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_v_len))
        horizontal_dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, hk)
        horizontal_light = cv2.morphologyEx(light, cv2.MORPH_OPEN, hk)
        vertical_dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, vk)
        vertical_light = cv2.morphologyEx(light, cv2.MORPH_OPEN, vk)
        horizontal = cv2.bitwise_or(horizontal_dark, horizontal_light)
        vertical = cv2.bitwise_or(vertical_dark, vertical_light)

        overlays = []
        rows = np.where(mask_bool.any(axis=1))[0]
        cols = np.where(mask_bool.any(axis=0))[0]

        for y in rows:
            if not (horizontal[y] > 0).any():
                continue
            for start, end in cls._extract_runs_1d(mask_bool[y]):
                best = None
                for support_src in (horizontal_dark, horizontal_light):
                    support_row = support_src[y] > 0
                    left_idx = np.where(support_row[:start])[0]
                    right_idx = np.where(support_row[end + 1 :])[0]
                    if left_idx.size == 0 or right_idx.size == 0:
                        continue
                    x1 = int(left_idx[-1])
                    x2 = int(end + 1 + right_idx[0])
                    if (x2 - x1) < max(10, min_h_len // 2):
                        continue
                    y1 = max(0, y - 1)
                    y2 = min(h, y + 2)
                    sx1 = max(0, x1 - 12)
                    sx2 = min(w, x2 + 13)
                    support_strip = (support_src[y1:y2, sx1:sx2] > 0) & (~mask_bool[y1:y2, sx1:sx2])
                    color = cls._sample_median_color(img_rgb[y1:y2, sx1:sx2], support_strip)
                    if color is None:
                        continue
                    contrast = float(np.mean(np.abs(np.asarray(color, dtype=np.float32) - bg_color)))
                    score = float(np.count_nonzero(support_strip)) + contrast * 10.0
                    if best is None or score > best["score"]:
                        best = {
                            "x1": x1,
                            "x2": x2,
                            "color": color,
                            "score": score,
                            "thickness": int(np.clip(np.sum(np.any(support_src[max(0, y - 2) : min(h, y + 3), x1 : x2 + 1] > 0, axis=1)), 1, 3)),
                            "contrast": contrast,
                        }
                if best and best["contrast"] >= 8.0:
                    cls._register_overlay(overlays, "h", y, best["x1"], best["x2"], best["color"], best["thickness"])

        for x in cols:
            if not (vertical[:, x] > 0).any():
                continue
            for start, end in cls._extract_runs_1d(mask_bool[:, x]):
                best = None
                for support_src in (vertical_dark, vertical_light):
                    support_col = support_src[:, x] > 0
                    top_idx = np.where(support_col[:start])[0]
                    bottom_idx = np.where(support_col[end + 1 :])[0]
                    if top_idx.size == 0 or bottom_idx.size == 0:
                        continue
                    y1 = int(top_idx[-1])
                    y2 = int(end + 1 + bottom_idx[0])
                    if (y2 - y1) < max(10, min_v_len // 2):
                        continue
                    x1 = max(0, x - 1)
                    x2 = min(w, x + 2)
                    sy1 = max(0, y1 - 12)
                    sy2 = min(h, y2 + 13)
                    support_strip = (support_src[sy1:sy2, x1:x2] > 0) & (~mask_bool[sy1:sy2, x1:x2])
                    color = cls._sample_median_color(img_rgb[sy1:sy2, x1:x2], support_strip)
                    if color is None:
                        continue
                    contrast = float(np.mean(np.abs(np.asarray(color, dtype=np.float32) - bg_color)))
                    score = float(np.count_nonzero(support_strip)) + contrast * 10.0
                    if best is None or score > best["score"]:
                        best = {
                            "y1": y1,
                            "y2": y2,
                            "color": color,
                            "score": score,
                            "thickness": int(np.clip(np.sum(np.any(support_src[y1 : y2 + 1, max(0, x - 2) : min(w, x + 3)] > 0, axis=0)), 1, 3)),
                            "contrast": contrast,
                        }
                if best and best["contrast"] >= 8.0:
                    cls._register_overlay(overlays, "v", x, best["y1"], best["y2"], best["color"], best["thickness"])

        return overlays

    @staticmethod
    def _draw_line_overlays(crop_img, overlays):
        if not overlays:
            return crop_img
        from PIL import Image

        arr = np.asarray(crop_img, dtype=np.uint8).copy()
        h, w = arr.shape[:2]
        for item in overlays:
            try:
                orientation = str(item.get("orientation") or "")
                pos = int(item.get("pos", 0))
                start = int(item.get("start", 0))
                end = int(item.get("end", 0))
                color = tuple(int(v) for v in (item.get("color") or (0, 0, 0)))
                thickness = max(1, int(item.get("thickness", 1)))
            except Exception:
                continue
            if orientation == "h":
                pos = max(0, min(pos, h - 1))
                start = max(0, min(start, w - 1))
                end = max(0, min(end, w - 1))
                if end > start:
                    cv2.line(arr, (start, pos), (end, pos), color, thickness=thickness, lineType=cv2.LINE_AA)
            elif orientation == "v":
                pos = max(0, min(pos, w - 1))
                start = max(0, min(start, h - 1))
                end = max(0, min(end, h - 1))
                if end > start:
                    cv2.line(arr, (pos, start), (pos, end), color, thickness=thickness, lineType=cv2.LINE_AA)
        return Image.fromarray(arr, mode="RGB")

    @classmethod
    def _run_local_fill(cls, crop_img, crop_mask, analysis=None):
        from PIL import Image

        img_rgb = np.asarray(crop_img, dtype=np.uint8)
        mask_u8 = np.asarray(crop_mask, dtype=np.uint8)
        hard_mask = bool((analysis or {}).get("hard_mask", False))
        sample_mask_src = None
        if hard_mask:
            sample_mask_src = (analysis or {}).get("sample_mask")
            try:
                if sample_mask_src is not None:
                    sample_mask_src = np.asarray(sample_mask_src, dtype=np.uint8)
            except Exception:
                sample_mask_src = None
        if sample_mask_src is None:
            sample_mask_src = mask_u8
        sample_mask = cls._fill_sample_ring(sample_mask_src) if hard_mask else cls._analysis_ring(mask_u8)
        fill_color = cls._estimate_fill_color(img_rgb, sample_mask)
        if fill_color is None:
            fallback_mask = ~(mask_u8 > 0)
            fill_color = cls._estimate_fill_color(img_rgb, fallback_mask)
        if fill_color is None:
            return None

        pred = np.empty_like(img_rgb, dtype=np.uint8)
        pred[:, :] = np.asarray(fill_color, dtype=np.uint8)
        mask_f = (mask_u8 > 0).astype(np.float32)
        if not hard_mask:
            mask_f = cv2.GaussianBlur(mask_f, (0, 0), sigmaX=1.0, sigmaY=1.0)
        mask_f = np.clip(mask_f, 0.0, 1.0)[..., None]

        blended = img_rgb.astype(np.float32) * (1.0 - mask_f) + pred.astype(np.float32) * mask_f
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        return Image.fromarray(blended, mode="RGB")

    @classmethod
    def _run_local_cv2(cls, crop_img, crop_mask, analysis):
        from PIL import Image

        img_bgr = cv2.cvtColor(np.asarray(crop_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
        mask_u8 = (np.asarray(crop_mask, dtype=np.uint8) > 0).astype(np.uint8) * 255

        edge_density = float((analysis or {}).get("edge_density", 1.0))
        plane_residual = float((analysis or {}).get("plane_residual", 999.0))
        radius = 3 if edge_density <= 0.05 else 4
        method = cv2.INPAINT_NS if plane_residual <= 10.0 and edge_density <= 0.04 else cv2.INPAINT_TELEA
        out_bgr = cv2.inpaint(img_bgr, mask_u8, radius, method)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(out_rgb, mode="RGB")

    @classmethod
    def _run_local_strategy(cls, crop_img, crop_mask, analysis):
        strategy = str((analysis or {}).get("strategy") or cls.STRATEGY_REMOTE)
        if strategy == cls.STRATEGY_LOCAL_FILL:
            return cls._run_local_fill(crop_img, crop_mask, analysis)
        if strategy == cls.STRATEGY_LOCAL_CV2:
            return cls._run_local_cv2(crop_img, crop_mask, analysis)
        return None

    def _call_api_crop(self, url, image_pil, mask_pil, crop_padding):
        import base64
        from io import BytesIO

        import requests
        from PIL import Image

        payload_crop = self._crop_from_mask(image_pil, mask_pil, crop_padding=crop_padding)
        if not payload_crop:
            return None

        crop_box, crop_img, crop_mask = payload_crop
        buf_img = BytesIO()
        crop_img.save(buf_img, "PNG")
        buf_mask = BytesIO()
        crop_mask.save(buf_mask, "PNG")
        payload = {
            "image": base64.b64encode(buf_img.getvalue()).decode("ascii"),
            "mask": base64.b64encode(buf_mask.getvalue()).decode("ascii"),
            # Reasonable defaults (same spirit as the reference project)
            "ldm_steps": 30,
            "hd_strategy": "Original",
            "sd_sampler": "UniPC",
        }

        resp = requests.post(str(url), json=payload, timeout=self.timeout_sec)
        if resp.status_code != 200:
            raise RuntimeError(
                _t_global(
                    f"IOPaint API返回错误: {resp.status_code} {resp.text[:200]}",
                    f"IOPaint API error: {resp.status_code} {resp.text[:200]}",
                )
            )

        res_crop = Image.open(BytesIO(resp.content)).convert("RGB")
        if tuple(getattr(res_crop, "size", ())) != tuple(getattr(crop_img, "size", ())):
            raise RuntimeError(
                _t_global(
                    f"IOPaint 返回尺寸异常: {getattr(res_crop, 'size', None)} != {getattr(crop_img, 'size', None)}",
                    f"IOPaint returned an unexpected image size: {getattr(res_crop, 'size', None)} != {getattr(crop_img, 'size', None)}",
                )
            )
        return (crop_box, res_crop, crop_mask)

    @staticmethod
    def _apply_crop_result(final_img, crop_res):
        if not crop_res:
            return
        from PIL import Image, ImageFilter

        payload = list(crop_res) + [None, None]
        crop_box, res_crop, crop_mask, overlays, options = payload[:5]
        if overlays:
            try:
                res_crop = InpaintThread._draw_line_overlays(res_crop, overlays)
            except Exception:
                pass
        options = options or {}
        try:
            blur_radius = float(options.get("blur_radius", 3.0))
            if blur_radius <= 0:
                blur_mask = crop_mask
            else:
                blur_mask = crop_mask.filter(ImageFilter.GaussianBlur(blur_radius))
        except Exception:
            blur_mask = crop_mask
        orig_crop_area = final_img.crop(crop_box)
        blended = Image.composite(res_crop, orig_crop_area, blur_mask)
        final_img.paste(blended, (int(crop_box[0]), int(crop_box[1])))

    def _normalize_roi(self, roi, img_size):
        if not (isinstance(roi, (list, tuple)) and len(roi) == 4):
            return None
        try:
            rx, ry, rw, rh = [int(v) for v in roi]
        except Exception:
            return None
        if rw <= 0 or rh <= 0:
            return None

        W, H = img_size
        rx = max(0, min(rx, W - 1))
        ry = max(0, min(ry, H - 1))
        rw = max(1, min(rw, W - rx))
        rh = max(1, min(rh, H - ry))
        return [rx, ry, rw, rh]

    def _build_tasks(self, src, img, boxes_raw, roi):
        buckets = {
            self.MODE_AUTO: [],
            self.MODE_FILL: [],
            self.MODE_REMOTE: [],
        }
        for b in boxes_raw:
            if not isinstance(b, dict):
                continue
            if not bool(b.get("clean_enabled", True)):
                continue
            rect = self._extract_rect(b)
            if rect is None:
                continue
            if not self._intersects_roi(rect, roi):
                continue
            x, y, _, _ = rect
            mode = self._normalize_box_clean_mode(b.get("clean_mode"))
            if self.run_mode == self.RUN_FILL:
                effective_mode = self.MODE_FILL
            elif self.run_mode == self.RUN_REMOTE:
                effective_mode = self.MODE_REMOTE
            else:
                effective_mode = mode
            buckets.setdefault(effective_mode, []).append((y, x, b))
        if not any(buckets.values()):
            return []

        tasks = []
        next_idx = 0
        for requested_mode, sortable in buckets.items():
            if not sortable:
                continue
            sortable.sort(key=lambda t: (t[0], t[1]))
            boxes = [t[2] for t in sortable]
            if requested_mode in (self.MODE_FILL, self.MODE_REMOTE, self.MODE_AUTO):
                # Decide per OCR box. Smart mode also needs per-box analysis so the background
                # complexity is judged on the actual text region instead of a merged paragraph/table row.
                clusters = [[box] for box in boxes]
            else:
                clusters = self._cluster_boxes(boxes, merge_gap=max(8, int(self.box_padding) * 2 + 6))
            for cluster in clusters:
                prefer_rect = bool(requested_mode in (self.MODE_FILL, self.MODE_REMOTE))
                mask_padding = 0 if requested_mode == self.MODE_FILL else max(0, self.box_padding)
                sample_mask_pil = None
                if requested_mode == self.MODE_FILL:
                    sample_mask_pil = self._create_mask(
                        img.size,
                        cluster,
                        padding=0,
                        prefer_rect=True,
                        rect_expand_x=0,
                        rect_expand_y=0,
                    )
                mask_pil = self._create_mask(
                    img.size,
                    cluster,
                    padding=mask_padding,
                    prefer_rect=prefer_rect,
                    rect_expand_x=self.fill_pad_x if requested_mode == self.MODE_FILL else (self.remote_pad_x if requested_mode == self.MODE_REMOTE else 0),
                    rect_expand_y=self.fill_pad_y if requested_mode == self.MODE_FILL else (self.remote_pad_y if requested_mode == self.MODE_REMOTE else 0),
                )
                if roi is not None:
                    mask_pil = self._limit_mask_to_roi(mask_pil, roi, img.size)
                    if sample_mask_pil is not None:
                        sample_mask_pil = self._limit_mask_to_roi(sample_mask_pil, roi, img.size)
                crop_payload = self._crop_from_mask(img, mask_pil, crop_padding=max(0, self.crop_padding))
                if not crop_payload:
                    continue
                crop_box, crop_img, crop_mask = crop_payload
                analysis = self._analyze_crop(crop_img, crop_mask)
                if requested_mode == self.MODE_AUTO and analysis.get("strategy") == self.STRATEGY_REMOTE:
                    # Smart remote should use the same stronger rectangular mask as explicit IOPaint.
                    mask_pil = self._create_mask(
                        img.size,
                        cluster,
                        padding=max(0, self.box_padding),
                        prefer_rect=True,
                        rect_expand_x=self.remote_pad_x,
                        rect_expand_y=self.remote_pad_y,
                    )
                    if roi is not None:
                        mask_pil = self._limit_mask_to_roi(mask_pil, roi, img.size)
                    crop_payload = self._crop_from_mask(img, mask_pil, crop_padding=max(0, self.crop_padding))
                    if not crop_payload:
                        continue
                    crop_box, crop_img, crop_mask = crop_payload
                    analysis = self._analyze_crop(crop_img, crop_mask)
                    analysis["strategy"] = self.STRATEGY_REMOTE
                    analysis["mask_shape"] = "rect"
                elif requested_mode == self.MODE_AUTO and analysis.get("strategy") == self.STRATEGY_LOCAL_FILL:
                    # Keep smart-fill behavior aligned with explicit "单色本页": rect mask + per-axis expansion.
                    sample_mask_pil = self._create_mask(
                        img.size,
                        cluster,
                        padding=0,
                        prefer_rect=True,
                        rect_expand_x=0,
                        rect_expand_y=0,
                    )
                    mask_pil = self._create_mask(
                        img.size,
                        cluster,
                        padding=0,
                        prefer_rect=True,
                        rect_expand_x=self.fill_pad_x,
                        rect_expand_y=self.fill_pad_y,
                    )
                    if roi is not None:
                        mask_pil = self._limit_mask_to_roi(mask_pil, roi, img.size)
                        sample_mask_pil = self._limit_mask_to_roi(sample_mask_pil, roi, img.size)
                    crop_payload = self._crop_from_mask(img, mask_pil, crop_padding=max(0, self.crop_padding))
                    if not crop_payload:
                        continue
                    crop_box, crop_img, crop_mask = crop_payload
                    analysis = self._analyze_crop(crop_img, crop_mask)
                    analysis["strategy"] = self.STRATEGY_LOCAL_FILL
                    analysis["hard_mask"] = True
                    analysis["mask_shape"] = "rect"
                    analysis["fill_single_box"] = True
                    if sample_mask_pil is not None:
                        try:
                            analysis["sample_mask"] = sample_mask_pil.crop(crop_box)
                        except Exception:
                            pass
                if requested_mode == self.MODE_FILL:
                    analysis["strategy"] = self.STRATEGY_LOCAL_FILL
                    analysis["hard_mask"] = True
                    analysis["mask_shape"] = "rect"
                    analysis["fill_single_box"] = True
                    if sample_mask_pil is not None:
                        try:
                            analysis["sample_mask"] = sample_mask_pil.crop(crop_box)
                        except Exception:
                            pass
                elif requested_mode == self.MODE_REMOTE:
                    analysis["strategy"] = self.STRATEGY_REMOTE
                    analysis["mask_shape"] = "rect"
                overlays = [] if analysis.get("strategy") == self.STRATEGY_REMOTE or requested_mode == self.MODE_FILL else self._collect_line_overlays(crop_img, crop_mask, analysis)
                analysis["line_overlays"] = int(len(overlays))
                analysis["requested_mode"] = requested_mode
                logger.debug(
                    "Clean strategy %s for %s cluster %s (requested=%s edge=%.4f texture=%.2f color_std=%.2f plane=%.2f lines=%s)",
                    analysis.get("strategy"),
                    os.path.basename(str(src or "")),
                    next_idx,
                    requested_mode,
                    float(analysis.get("edge_density", 0.0)),
                    float(analysis.get("texture", 0.0)),
                    float(analysis.get("color_std", 0.0)),
                    float(analysis.get("plane_residual", 0.0)),
                    int(analysis.get("line_overlays", 0)),
                )
                tasks.append(
                    {
                        "index": next_idx,
                        "boxes": cluster,
                        "mask": mask_pil,
                        "analysis": analysis,
                        "overlays": overlays,
                        "remote_required": bool(requested_mode == self.MODE_REMOTE or analysis.get("strategy") == self.STRATEGY_REMOTE),
                        "composite_blur": 0.0 if requested_mode == self.MODE_FILL or analysis.get("strategy") == self.STRATEGY_REMOTE else 3.0,
                    }
                )
                next_idx += 1
        if tasks:
            counts = {
                self.STRATEGY_LOCAL_FILL: 0,
                self.STRATEGY_LOCAL_CV2: 0,
                self.STRATEGY_REMOTE: 0,
            }
            total_lines = 0
            for task in tasks:
                strategy = str((task.get("analysis") or {}).get("strategy") or self.STRATEGY_REMOTE)
                counts[strategy] = counts.get(strategy, 0) + 1
                total_lines += int((task.get("analysis") or {}).get("line_overlays", 0) or 0)
            logger.info(
                "Clean summary for %s: fill=%d cv2=%d remote=%d line_overlays=%d run_mode=%s",
                os.path.basename(str(src or "")),
                int(counts.get(self.STRATEGY_LOCAL_FILL, 0)),
                int(counts.get(self.STRATEGY_LOCAL_CV2, 0)),
                int(counts.get(self.STRATEGY_REMOTE, 0)),
                int(total_lines),
                self.run_mode,
            )
        return tasks

    def _run_remote_tasks(self, img, remote_tasks, api_urls):
        from concurrent.futures import ThreadPoolExecutor, as_completed

        if not remote_tasks:
            return []
        strict_required = any(bool(task.get("remote_required", False)) for task in remote_tasks)
        if not api_urls:
            if strict_required:
                raise RuntimeError(_t_global("已指定 IOPaint 去字，但未配置可用的 IOPaint API 地址。", "IOPaint was explicitly selected, but no IOPaint API URL is configured."))
            logger.warning("Complex regions detected but no IOPaint API configured; falling back to local cv2 inpaint.")
            out = []
            for task in remote_tasks:
                payload = self._crop_from_mask(img, task["mask"], crop_padding=max(0, self.crop_padding))
                if not payload:
                    continue
                crop_box, crop_img, crop_mask = payload
                local_img = self._run_local_cv2(crop_img, crop_mask, task.get("analysis") or {})
                if local_img is not None:
                    out.append((task["index"], (crop_box, local_img, crop_mask, task.get("overlays") or [], {"blur_radius": float(task.get("composite_blur", 3.0) or 0.0)})))
            return out

        def worker(task):
            start = int(task["index"]) % len(api_urls)
            last_err = None
            for off in range(len(api_urls)):
                url = api_urls[(start + off) % len(api_urls)]
                try:
                    crop_res = self._call_api_crop(
                        url,
                        img,
                        task["mask"],
                        crop_padding=max(0, self.crop_padding),
                    )
                    if crop_res:
                        crop_box, res_crop, crop_mask = crop_res
                        return task["index"], (crop_box, res_crop, crop_mask, task.get("overlays") or [], {"blur_radius": float(task.get("composite_blur", 3.0) or 0.0)})
                    return task["index"], None
                except Exception as e:
                    last_err = e
                    continue
            if bool(task.get("remote_required", False)):
                raise last_err or RuntimeError(_t_global("IOPaint 请求失败", "IOPaint request failed"))
            logger.warning("IOPaint failed for cluster %s: %s. Falling back to local cv2 inpaint.", task.get("index"), last_err)
            payload = self._crop_from_mask(img, task["mask"], crop_padding=max(0, self.crop_padding))
            if not payload:
                raise last_err or RuntimeError(_t_global("IOPaint 请求失败", "IOPaint request failed"))
            crop_box, crop_img, crop_mask = payload
            local_img = self._run_local_cv2(crop_img, crop_mask, task.get("analysis") or {})
            if local_img is not None:
                return task["index"], (crop_box, local_img, crop_mask, task.get("overlays") or [], {"blur_radius": float(task.get("composite_blur", 3.0) or 0.0)})
            raise last_err or RuntimeError(_t_global("IOPaint 请求失败", "IOPaint request failed"))

        if len(remote_tasks) == 1:
            return [worker(remote_tasks[0])]

        out = []
        with ThreadPoolExecutor(max_workers=min(len(api_urls), len(remote_tasks))) as ex:
            futs = [ex.submit(worker, task) for task in remote_tasks]
            for fut in as_completed(futs):
                if self.isInterruptionRequested():
                    return None
                idx, crop_res = fut.result()
                if crop_res:
                    out.append((idx, crop_res))
        return out

    def _save_result_image(self, src, final_img):
        import time

        out_path = build_asset_path(
            self.out_dir,
            "inpaint",
            src,
            suffix=str(int(time.time() * 1000)),
            ext=".png",
        )
        final_img.save(out_path, "PNG")
        return out_path

    def run(self):
        from PIL import Image

        total = len(self.images)
        done = 0
        canceled = False
        for src in self.images:
            if self.isInterruptionRequested():
                canceled = True
                break
            done += 1
            try:
                boxes_raw = self.box_data.get(src, []) or []
                if not boxes_raw:
                    self.progress.emit(done, total)
                    continue

                in_path = str(src)
                try:
                    cand = (self.input_image_by_src or {}).get(src)
                    if cand and os.path.exists(str(cand)):
                        in_path = str(cand)
                except Exception:
                    pass

                img = Image.open(in_path).convert("RGB")
                roi = self._normalize_roi((self.roi_by_image or {}).get(src), img.size)
                tasks = self._build_tasks(src, img, boxes_raw, roi)
                if not tasks:
                    self.progress.emit(done, total)
                    continue

                local_results = []
                remote_tasks = []
                for task in tasks:
                    strategy = str((task.get("analysis") or {}).get("strategy") or self.STRATEGY_REMOTE)
                    if strategy == self.STRATEGY_REMOTE:
                        remote_tasks.append(task)
                        continue
                    payload = self._crop_from_mask(img, task["mask"], crop_padding=max(0, self.crop_padding))
                    if not payload:
                        continue
                    crop_box, crop_img, crop_mask = payload
                    local_img = self._run_local_strategy(crop_img, crop_mask, task.get("analysis"))
                    if local_img is None:
                        remote_tasks.append(task)
                        continue
                    local_results.append((task["index"], (crop_box, local_img, crop_mask, task.get("overlays") or [], {"blur_radius": float(task.get("composite_blur", 3.0) or 0.0)})))

                remote_results = self._run_remote_tasks(img, remote_tasks, list(self.api_urls or []))
                if remote_results is None:
                    canceled = True
                    break

                final = img.copy()
                for _, crop_res in sorted((local_results or []) + (remote_results or []), key=lambda t: t[0]):
                    self._apply_crop_result(final, crop_res)

                out_path = self._save_result_image(src, final)
                self.results.append((src, out_path))
                self.finished_one.emit(src, out_path)
                self.progress.emit(done, total)
            except Exception as e:
                self.error.emit(str(e))
                break

        self.all_done.emit(bool(canceled))

class RibbonGroup(QFrame):
    """ 功能分组 (例如：剪贴板) """
    def __init__(self, title, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 4, 6, 0) # 内部间距
        layout.setSpacing(0)
        
        # 上半部分：按钮容器
        self.content_layout = QHBoxLayout()
        self.content_layout.setContentsMargins(0, 2, 0, 0)
        self.content_layout.setSpacing(2)
        layout.addLayout(self.content_layout)
        
        layout.addStretch(1) # 弹簧，把按钮顶上去
        
        # 下半部分：标题
        lbl = QLabel(title)
        lbl.setProperty("cssClass", "groupLabel")
        lbl.setAlignment(Qt.AlignCenter)
        lbl.setFixedHeight(18)
        layout.addWidget(lbl)

    def add_widget(self, widget):
        self.content_layout.addWidget(widget)

class RibbonSeparator(QFrame):
    """ 
    [关键修改] 参考图片制作的分割线 
    """
    def __init__(self):
        super().__init__()
        # Make the separator reliably visible regardless of stylesheet repolish quirks.
        self.setObjectName("RibbonSeparator")
        self.setProperty("cssClass", "RibbonSeparator")  # keep for backward-compat QSS
        self.setFrameShape(QFrame.VLine)
        self.setFrameShadow(QFrame.Plain)
        self.setLineWidth(1)
        self.setFixedWidth(1)   # 1px
        self.setFixedHeight(86) # mimic PPT 留白
        # Fallback: paint even if the global stylesheet doesn't apply.
        self.setStyleSheet("background-color: #C9C9C9; border: none;")

class RibbonLargeBtn(QToolButton):
    def __init__(self, text, icon_name, color="#444"):
        super().__init__()
        self.setText(text)
        self.setIcon(qta.icon(icon_name, color=color))
        self.setIconSize(QSize(24, 24)) 
        self.setToolButtonStyle(Qt.ToolButtonTextUnderIcon)
        self.setFixedSize(52, 66) 
        self.setStyleSheet("font-size: 11px; padding-top: 4px;")

class RibbonSmallBtn(QToolButton):
    def __init__(self, text, icon_name):
        super().__init__()
        self.setText(text)
        self.setIcon(qta.icon(icon_name, color="#444"))
        self.setIconSize(QSize(14, 14))
        self.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.setFixedHeight(22)
        self.setFixedWidth(70)
        self.setStyleSheet("font-size: 11px; text-align: left; padding-left: 5px;")

# ==================== 画布与主逻辑 ====================

class ShortcutsDialog(QDialog):
    """A small, scrollable shortcut cheat-sheet."""
    def __init__(self, parent=None):
        super().__init__(parent)
        tr = getattr(parent, "_t", None)
        self._t = tr if callable(tr) else (lambda zh, en=None: zh)
        self.setWindowTitle(self._t("快捷键", "Shortcuts"))
        self.resize(560, 420)

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel(self._t("常用快捷键", "Keyboard shortcuts"))
        title.setStyleSheet("font-weight: bold; font-size: 14px;")
        root.addWidget(title)

        tips = QLabel(self._t("提示：Ctrl+滚轮缩放画布；按住滚轮拖动可平移画布。", "Tip: Ctrl+Wheel to zoom; hold Wheel to pan."))
        tips.setStyleSheet("color: #666;")
        root.addWidget(tips)

        rows = [
            (self._t("Ctrl+滚轮", "Ctrl+Wheel"), self._t("缩放画布", "Zoom canvas")),
            (self._t("滚轮按住拖动", "Wheel drag"), self._t("平移画布", "Pan canvas")),
            ("Ctrl+0", self._t("适应窗口", "Fit to window")),
            ("Ctrl++ / Ctrl+-", self._t("缩放（10%步进）", "Zoom (10% step)")),
            ("Ctrl+V", self._t("粘贴文本框（无选中时：若剪贴板是截图/图片则直接导入）", "Paste box (if none selected: paste image/screenshot as slide)")),
            ("Ctrl+Shift+V", self._t("粘贴图片（从剪贴板导入截图/图片）", "Paste image (from clipboard)")),
            ("Ctrl+O", self._t("导入图片", "Import images")),
            ("Ctrl+Shift+O", self._t("导入PDF", "Import PDF")),
            ("Ctrl+Enter", self._t("OCR本页", "OCR current")),
            ("Ctrl+R", self._t("OCR全部", "OCR all")),
            ("Ctrl+I", self._t("智能去字（本页）", "Smart clean (current)")),
            ("Ctrl+Shift+I", self._t("智能去字（全部）", "Smart clean (all)")),
            ("Ctrl+S", self._t("导出PPT", "Export PPT")),
            ("F5", self._t("预览PPT", "Preview PPT")),
            ("Ctrl+N", self._t("新建空白页", "New blank slide")),
            ("Ctrl+D", self._t("复制当前页", "Duplicate slide")),
            ("Alt+Up / Alt+Down", self._t("上移/下移当前页", "Move slide up/down")),
            ("PageUp / PageDown", self._t("上一页/下一页", "Previous/Next slide")),
            ("Delete", self._t("删除选中文本框", "Delete selected box")),
            ("Ctrl+Z / Ctrl+Y", self._t("撤销 / 重做", "Undo / Redo")),
            ("Ctrl+Alt+L", self._t("显示/隐藏缩略图", "Toggle thumbnails")),
            ("Ctrl+Alt+R", self._t("显示/隐藏右侧面板", "Toggle right panel")),
            ("F1", self._t("打开本快捷键窗口", "Open this window")),
        ]

        table = QTableWidget(len(rows), 2, self)
        table.setHorizontalHeaderLabels([self._t("快捷键", "Shortcut"), self._t("功能", "Action")])
        table.setEditTriggers(QTableWidget.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectRows)
        table.setSelectionMode(QTableWidget.SingleSelection)
        table.setAlternatingRowColors(True)
        table.verticalHeader().setVisible(False)
        table.horizontalHeader().setStretchLastSection(True)
        try:
            table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
            table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
        except Exception:
            pass

        mono = QFont("Consolas")
        if not mono.exactMatch():
            mono = QFont("Courier New")

        for r, (k, desc) in enumerate(rows):
            it_k = QTableWidgetItem(k)
            it_k.setFont(mono)
            it_k.setForeground(QColor("#333"))
            it_d = QTableWidgetItem(desc)
            table.setItem(r, 0, it_k)
            table.setItem(r, 1, it_d)

        root.addWidget(table)

        btn_row = QWidget()
        btn_l = QHBoxLayout(btn_row)
        btn_l.setContentsMargins(0, 0, 0, 0)
        btn_l.addStretch()
        btn_close = QPushButton(self._t("关闭", "Close"))
        btn_close.clicked.connect(self.accept)
        btn_l.addWidget(btn_close)
        root.addWidget(btn_row)

class CustomGraphicsView(QGraphicsView):
    """自定义GraphicsView，支持吸管取色"""
    def __init__(self, scene, parent_win):
        super().__init__(scene)
        self.parent_win = parent_win
        self._panning = False
        self._pan_last = None
        # Match common editor behavior: zoom around cursor.
        try:
            self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
            self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        except Exception:
            pass

    def _evt_pos(self, event):
        """Best-effort event position as QPoint (Qt6 vs bindings differences)."""
        # Prefer Qt6 API: QSinglePointEvent.position() -> QPointF
        try:
            v = getattr(event, "position", None)
            if v is not None:
                p = v() if callable(v) else v
                try:
                    return p.toPoint()  # QPointF -> QPoint
                except Exception:
                    return p
        except Exception as e:
            logger.debug(f"获取事件位置(position)失败: {e}")
        # Fallback: Qt5-style pos()
        try:
            v = getattr(event, "pos", None)
            if v is not None:
                p = v() if callable(v) else v
                return p
        except Exception as e:
            logger.debug(f"获取事件位置(pos)失败: {e}")
        return QPoint(0, 0)

    def mousePressEvent(self, event):
        # Middle-mouse drag to pan
        try:
            if event.button() == Qt.MiddleButton:
                self._panning = True
                self._pan_last = self._evt_pos(event)
                self.setCursor(Qt.ClosedHandCursor)
                event.accept()
                return
        except Exception as e:
            logger.debug(f"处理中键按下事件失败: {e}")

        # ROI selection mode: drag a rectangle to set OCR/inpaint region
        try:
            if getattr(self.parent_win, "roi_select_mode", False) and event.button() == Qt.LeftButton:
                self.parent_win.canvas_roi_press(event)
                event.accept()
                return
        except Exception as e:
            logger.debug(f"处理ROI选择按下事件失败: {e}")

        # 如果是吸管模式，调用取色函数
        if self.parent_win.eyedropper_mode:
            self.parent_win.canvas_mouse_press(event)
        else:
            # 否则正常处理
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        try:
            if self._panning:
                p = self._evt_pos(event)
                if self._pan_last is None:
                    self._pan_last = p
                    event.accept()
                    return
                delta = p - self._pan_last
                self._pan_last = p
                try:
                    self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta.x()))
                    self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta.y()))
                except Exception:
                    pass
                event.accept()
                return
            try:
                if getattr(self.parent_win, "roi_select_mode", False):
                    self.parent_win.canvas_roi_move(event)
                    event.accept()
                    return
            except Exception:
                pass
            super().mouseMoveEvent(event)
        except Exception as e:
            # Prevent PySide6 from spamming "Error calling Python override..." on every move.
            try:
                import time as _time_mod
                now = _time_mod.time()
                last = float(getattr(self, "_last_mousemove_err_ts", 0.0) or 0.0)
                if (now - last) > 1.0:
                    self._last_mousemove_err_ts = now
                    logger.warning(f"mouseMoveEvent 异常: {e}")
            except Exception:
                pass

    def mouseReleaseEvent(self, event):
        try:
            if event.button() == Qt.MiddleButton and self._panning:
                self._panning = False
                self._pan_last = None
                # Restore cursor depending on current tool mode.
                self.setCursor(Qt.CrossCursor if getattr(self.parent_win, "eyedropper_mode", False) else Qt.ArrowCursor)
                event.accept()
                return
        except Exception as e:
            logger.debug(f"处理中键释放事件失败: {e}")
        try:
            if getattr(self.parent_win, "roi_select_mode", False) and event.button() == Qt.LeftButton:
                self.parent_win.canvas_roi_release(event)
                event.accept()
                return
        except Exception as e:
            logger.debug(f"处理ROI选择释放事件失败: {e}")
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        # Ctrl + wheel: zoom in/out
        try:
            if event.modifiers() & Qt.ControlModifier:
                dy = event.angleDelta().y()
                if dy == 0:
                    return
                factor = 1.15 if dy > 0 else (1.0 / 1.15)
                curr = float(self.transform().m11() or 1.0)
                target = curr * factor
                # Keep in sync with the bottom zoom slider (10%..400%)
                min_s, max_s = 0.10, 4.00
                if target < min_s:
                    factor = min_s / max(curr, 1e-6)
                elif target > max_s:
                    factor = max_s / max(curr, 1e-6)
                self.scale(factor, factor)
                try:
                    self.parent_win._update_zoom_label()
                except Exception as e:
                    logger.debug(f"更新缩放标签失败: {e}")
                event.accept()
                return
        except Exception as e:
            logger.debug(f"处理滚轮缩放事件失败: {e}")
        super().wheelEvent(event)

class CanvasTextBox(QGraphicsItemGroup):
    def __init__(self, rect, text, index, parent_win):
        super().__init__()
        self.parent_win = parent_win
        # 绑定 box_data 里的 dict：用于把 UI 改动同步到预览/导出
        self.model = rect if isinstance(rect, dict) and "rect" in rect else None
        self.model_index = index
        if self.model is not None:
            # rect 参数实际是 model dict
            text = self.model.get("text", text)
            rect = self.model.get("rect", rect)
        self.setFlag(QGraphicsItemGroup.ItemIsSelectable)
        self.setFlag(QGraphicsItemGroup.ItemIsMovable)
        self.setFlag(QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setAcceptHoverEvents(True)

        self._resizing = False
        self._resize_handle = None
        self._press_pos_item = None
        self._start_pos = None
        self._start_rect = None

        # 每个文本框自己的背景颜色（None表示使用全局颜色）
        self.custom_bg_color = None
        self.use_custom_bg = False
        # 默认做得更透明一点，避免开启背景色后“整张图像看不见”
        self.bg_alpha = DEFAULT_BG_ALPHA
        if self.model is not None:
            self.use_custom_bg = bool(self.model.get("use_custom_bg", False))
            bg = self.model.get("bg_color")
            if isinstance(bg, (tuple, list)) and len(bg) == 3:
                try:
                    self.custom_bg_color = QColor(int(bg[0]), int(bg[1]), int(bg[2]))
                except Exception:
                    self.custom_bg_color = None
            try:
                self.bg_alpha = int(self.model.get("bg_alpha", 120))
            except Exception:
                self.bg_alpha = DEFAULT_BG_ALPHA

        # 确保rect可以正确解包
        if isinstance(rect, dict):
            x, y, w, h = rect.get('x', 0), rect.get('y', 0), rect.get('w', 100), rect.get('h', 50)
        elif isinstance(rect, (tuple, list)) and len(rect) == 4:
            x, y, w, h = rect
        else:
            logger.warning(f"无效的rect格式: {rect}")
            x, y, w, h = 0, 0, 100, 50

        self.box = QGraphicsRectItem(0, 0, w, h)
        self.box.setPen(QPen(QColor(180, 180, 180), 1, Qt.DashLine))
        # 先设置默认背景，稍后会更新
        self.box.setBrush(QColor(255, 255, 255, 1))
        # 让点击事件落在 group 上，避免用户点到子 item（矩形/文字）导致选中逻辑失效
        self.box.setAcceptedMouseButtons(Qt.NoButton)
        self.addToGroup(self.box)

        self.txt = QGraphicsTextItem(text)
        self.txt.setDefaultTextColor(Qt.black)
        self.txt.setAcceptedMouseButtons(Qt.NoButton)

        # Default font; actual size is set by apply_style_from_model().
        self.txt.setFont(QFont("Microsoft YaHei", 12))
        self.txt.setTextWidth(w)
        # Keep the same "usable" box area as PPT export (which uses tiny margins).
        try:
            self.txt.document().setDocumentMargin(0)
        except Exception:
            pass
        self.addToGroup(self.txt)
        self.setPos(x, y)

        # 最后更新背景色
        self.apply_style_from_model()
        self._sync_model_geometry()
        self._sync_model_bg()

    def _hit_test_handle(self, pos):
        """返回点击位置命中的缩放手柄（tl/tr/bl/br）或 None"""
        r = self.box.rect()
        pts = {
            "tl": r.topLeft(),
            "tr": r.topRight(),
            "bl": r.bottomLeft(),
            "br": r.bottomRight(),
        }
        radius = 6.0
        for name, p in pts.items():
            if (pos - p).manhattanLength() <= radius:
                return name
        return None

    def _sync_model_geometry(self):
        """将当前 item 的几何信息写回 model（用于预览/导出）"""
        if not isinstance(self.model, dict):
            return
        try:
            p = self.pos()
            r = self.box.rect()
            self.model["rect"] = [int(round(p.x())), int(round(p.y())), int(round(r.width())), int(round(r.height()))]
        except Exception as e:
            logger.warning(f"同步文本框位置失败: {e}")

    def _sync_model_bg(self):
        """将当前背景色状态写回 model"""
        if not isinstance(self.model, dict):
            return
        self.model["use_custom_bg"] = bool(self.use_custom_bg)
        if self.custom_bg_color is not None:
            self.model["bg_color"] = [int(self.custom_bg_color.red()), int(self.custom_bg_color.green()), int(self.custom_bg_color.blue())]
        else:
            self.model["bg_color"] = None
        self.model["bg_alpha"] = int(self.bg_alpha)

    def _clean_outline_pen(self):
        color = QColor(180, 180, 180, 180)
        if isinstance(self.model, dict):
            enabled = bool(self.model.get("clean_enabled", True))
            mode = InpaintThread._normalize_box_clean_mode(self.model.get("clean_mode"))
            if not enabled:
                color = QColor(185, 185, 185, 120)
            elif mode == InpaintThread.MODE_FILL:
                color = QColor(72, 140, 96, 200)
            elif mode == InpaintThread.MODE_REMOTE:
                color = QColor(67, 116, 196, 210)
        return QPen(color, 1, Qt.DashLine)

    def refresh_clean_outline(self):
        try:
            self.box.setPen(self._clean_outline_pen())
            self.box.update()
            self.update()
        except Exception:
            pass

    def itemChange(self, change, value):
        if change == QGraphicsItem.ItemPositionHasChanged:
            self._sync_model_geometry()
        return super().itemChange(change, value)

    def apply_style_from_model(self):
        """从 model 读取样式并应用到画布 item（字体/颜色/边框/透明度等）"""
        if not isinstance(self.model, dict):
            self.update_background()
            return

        def _auto_font_pt(text: str, box_w: float, box_h: float, family: str, bold: bool) -> float:
            """Estimate point size so the rendered text height/width matches the OCR box (scene units ~= image px).

            Math idea: measure the same text at a known point size, then scale linearly:
              pt ~= min(avail_w / w_per_pt, avail_h / h_per_pt).
            This is much closer than a fixed `box_h * 0.6` heuristic.
            """
            t = (text or "").strip()
            if not t:
                return 12.0

            # Leave a tiny padding so we don't overflow due to font metric rounding.
            avail_w = max(1.0, float(box_w) - 2.0)
            avail_h = max(1.0, float(box_h) - 2.0)

            sample_pt = 100.0
            f = QFont(str(family))
            f.setBold(bool(bold))
            f.setPointSizeF(sample_pt)
            fm = QFontMetricsF(f)

            # Multi-line: fit the widest line + total line spacing.
            lines = [ln for ln in t.splitlines()] or [t]
            try:
                w100 = max(float(fm.horizontalAdvance(ln or " ")) for ln in lines)
            except Exception:
                w100 = float(fm.horizontalAdvance(t))
            w100 = max(1.0, w100)

            line_h100 = float(fm.lineSpacing() or fm.height() or 1.0)
            h100 = max(1.0, line_h100 * max(1, len(lines)))

            pt_w = avail_w * sample_pt / w100
            pt_h = avail_h * sample_pt / h100
            pt = min(pt_w, pt_h) * 0.98  # small safety factor
            return float(max(6.0, min(pt, 600.0)))

        # 字体/字号/加粗
        family = self.model.get("font_family") or "Microsoft YaHei"
        text_for_measure = str(self.model.get("text") or "")
        fs = self.model.get("font_size")
        try:
            fs = int(fs) if fs is not None else None
        except Exception:
            fs = None

        # Backward compatibility: older builds capped auto-fit at 200pt which makes big titles too small.
        # If the stored value looks like it hit that cap, re-fit with the higher limit.
        if fs is not None and 200 <= fs <= 205:
            try:
                r = self.box.rect()
                h_pt_est = float(r.height()) * 72.0 / 96.0
                if h_pt_est > (float(fs) + 30.0):
                    fs = None
                    self.model["font_size"] = None
            except Exception:
                pass
        if fs is None:
            # Prefer the same font-fitting logic used by PPT export, so canvas preview matches PPT more closely.
            try:
                r = self.box.rect()
                fs = int(self.parent_win.fit_font_size_pt_like_ppt(text_for_measure, r.width(), r.height()))
                self.model["font_size"] = int(fs)
            except Exception:
                r = self.box.rect()
                fs = int(round(_auto_font_pt(text_for_measure, r.width(), r.height(), str(family), bool(self.model.get("bold", False)))))
                self.model["font_size"] = int(fs)

        # PPT uses points (pt) at 96 DPI mapping; for the canvas we set a pixel size derived from pt so it
        # scales with the scene and stays consistent across screens.
        px = max(1, int(round(float(fs) * 96.0 / 72.0)))
        font = QFont(str(family))
        font.setPixelSize(px)
        font.setBold(bool(self.model.get("bold", False)))
        self.txt.setFont(font)

        # 文字颜色
        tc = self.model.get("text_color", [0, 0, 0])
        if isinstance(tc, (list, tuple)) and len(tc) == 3:
            try:
                self.txt.setDefaultTextColor(QColor(int(tc[0]), int(tc[1]), int(tc[2])))
            except Exception:
                self.txt.setDefaultTextColor(Qt.black)
        else:
            self.txt.setDefaultTextColor(Qt.black)

        # 对齐
        try:
            opt = self.txt.document().defaultTextOption()
            a = (self.model.get("align") or "left").lower()
            if a == "center":
                opt.setAlignment(Qt.AlignHCenter)
            elif a == "right":
                opt.setAlignment(Qt.AlignRight)
            else:
                opt.setAlignment(Qt.AlignLeft)
            # Match PPT export (no auto-wrap); explicit '\n' still works.
            opt.setWrapMode(QTextOption.NoWrap)
            self.txt.document().setDefaultTextOption(opt)
        except Exception:
            pass

        # 边框：不提供自定义边框设置，仅保留编辑时虚线轮廓
        self.refresh_clean_outline()

        # 背景透明度（alpha:0-255）
        try:
            self.bg_alpha = int(self.model.get("bg_alpha", self.bg_alpha))
        except Exception:
            pass

        self.update_background()

    def update_background(self):
        """更新文本框背景色"""
        try:
            # 优先使用自定义颜色
            if self.use_custom_bg and self.custom_bg_color:
                color = QColor(self.custom_bg_color.red(), self.custom_bg_color.green(),
                             self.custom_bg_color.blue(), int(self.bg_alpha))
                self.box.setBrush(QBrush(color))
            elif hasattr(self.parent_win, 'use_text_bg') and self.parent_win.use_text_bg:
                # 使用全局背景色
                if hasattr(self.parent_win, 'text_bg_color'):
                    src_color = self.parent_win.text_bg_color
                    alpha = int(getattr(self.parent_win, "text_bg_alpha", 200))
                    color = QColor(src_color.red(), src_color.green(), src_color.blue(), alpha)
                    self.box.setBrush(QBrush(color))
                else:
                    # 默认白色
                    alpha = int(getattr(self.parent_win, "text_bg_alpha", 200))
                    color = QColor(255, 255, 255, alpha)
                    self.box.setBrush(QBrush(color))
            else:
                # 完全透明
                self.box.setBrush(QBrush(QColor(255, 255, 255, 1)))

            # 强制重绘
            self.box.update()
            self.update()
        except Exception as e:
            logger.warning(f"更新背景色失败: {e}")
            self.box.setBrush(QBrush(QColor(255, 255, 255, 1)))
            self.box.update()
            self.update()

    def paint(self, painter, option, widget):
        if self.isSelected():
            painter.setPen(QPen(QColor("#666"), 1, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self.box.rect())
            painter.setBrush(Qt.white); painter.setPen(Qt.black)
            r = self.box.rect()
            for p in [r.topLeft(), r.topRight(), r.bottomLeft(), r.bottomRight()]:
                painter.drawEllipse(p, 3, 3)
        else:
            painter.setPen(self._clean_outline_pen())
            painter.drawRect(self.box.rect())

    def mousePressEvent(self, event):
        # 选中
        self.parent_win.on_item_clicked(self)

        # 缩放：只有选中状态下，点到角上的小圆点才进入缩放
        if self.isSelected():
            handle = self._hit_test_handle(event.pos())
            if handle:
                self._resizing = True
                self._resize_handle = handle
                self._press_pos_item = event.pos()
                self._start_pos = self.pos()
                self._start_rect = self.box.rect()
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing and self._start_rect is not None and self._start_pos is not None and self._press_pos_item is not None:
            dx = event.pos().x() - self._press_pos_item.x()
            dy = event.pos().y() - self._press_pos_item.y()

            min_w, min_h = 30.0, 18.0
            new_pos = self._start_pos
            new_w = self._start_rect.width()
            new_h = self._start_rect.height()

            if self._resize_handle == "br":
                new_w = max(min_w, new_w + dx)
                new_h = max(min_h, new_h + dy)
            elif self._resize_handle == "tl":
                new_w = max(min_w, new_w - dx)
                new_h = max(min_h, new_h - dy)
                new_pos = self._start_pos + QPointF(dx, dy)
            elif self._resize_handle == "tr":
                new_w = max(min_w, new_w + dx)
                new_h = max(min_h, new_h - dy)
                new_pos = self._start_pos + QPointF(0, dy)
            elif self._resize_handle == "bl":
                new_w = max(min_w, new_w - dx)
                new_h = max(min_h, new_h + dy)
                new_pos = self._start_pos + QPointF(dx, 0)

            self.setPos(new_pos)
            self.box.setRect(0, 0, new_w, new_h)
            self.txt.setTextWidth(new_w)

            # 如果字号是自动（None），缩放时跟随高度变化
            if isinstance(self.model, dict) and self.model.get("font_size") is None:
                self.apply_style_from_model()
            else:
                self.update_background()

            self._sync_model_geometry()
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            self._resize_handle = None
            self._press_pos_item = None
            self._start_pos = None
            self._start_rect = None
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def hoverMoveEvent(self, event):
        if self.isSelected():
            handle = self._hit_test_handle(event.pos())
            if handle in ("tl", "br"):
                self.setCursor(Qt.SizeFDiagCursor)
            elif handle in ("tr", "bl"):
                self.setCursor(Qt.SizeBDiagCursor)
            else:
                self.setCursor(Qt.ArrowCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        super().hoverMoveEvent(event)

class PPTCloneApp(QMainWindow):
    def __init__(self):
        super().__init__()
        # Set a safe default title first; final title is set after UI language is resolved.
        self.setWindowTitle("PowerOCR Presentation")
        self.resize(1200, 760)
        self.setStyleSheet(GLOBAL_STYLE)

        self.settings_path = os.path.join(os.path.dirname(__file__), "settings.json")
        self.settings = self.load_settings()
        # UI language (affects visible labels/buttons; stored in settings.json).
        self.ui_lang_setting = str(self.settings.get("ui_lang") or "auto").strip()
        self.ui_lang = self._resolve_ui_lang(self.ui_lang_setting)
        # Make language available to non-UI helpers/threads in this module.
        global _UI_LANG
        _UI_LANG = self.ui_lang
        self.setWindowTitle(self._t("PowerOCR 演示", "PowerOCR Presentation"))

        self.images = []
        self.box_data = {}
        self.current_img = None
        self.selected_box = None
        self._clipboard_box = None  # 复制/剪切用：存 dict（rect/text/bg...）
        self._paste_nudge = 0
        self._format_brush_active = False
        self._format_brush_style = None
        self.undo_stack = []
        self.redo_stack = []
        # 预览生成的临时 PPT：path -> create_ts；定时清理“足够旧且未被占用”的文件
        self._temp_preview_ppts = {}
        self.scaled_images = {}  # 存储缩放后的图片路径
        self.temp_dir = None     # 临时目录（缩放图片）
        # 运行期缓存目录（缩放图片/临时图层/PDF渲染/去字输出等）：默认放到项目目录，避免跑到 C 盘 Temp。
        # 注意：OCR 模型缓存（official_models）不是这里，它由 PADDLE_PDX_CACHE_HOME 控制，默认也在项目目录 model/ 下。
        self.run_cache_dir = None
        try:
            import time as _time_mod
            run_id = f"run_{int(_time_mod.time() * 1000)}_{os.getpid()}"
            root = os.path.join(os.path.dirname(__file__), "_runtime_cache")
            self.run_cache_dir = os.path.join(root, run_id)
            os.makedirs(self.run_cache_dir, exist_ok=True)
        except Exception:
            # 项目目录不可写时，回退到系统临时目录
            self.run_cache_dir = tempfile.mkdtemp(prefix="ocr_runtime_")

        self.slide_assets_dir = os.path.join(self.run_cache_dir, "assets")
        try:
            os.makedirs(self.slide_assets_dir, exist_ok=True)
        except Exception:
            pass
        # 画布背景层（阴影/白底/图片）；用于在透明度频繁变化时强制重建，避免底图“消失”重绘伪影
        self._bg_shadow_item = None
        self._bg_white_item = None
        self._bg_pixmap_item = None
        self._current_pixmap = None
        # 拖动透明度滑块时，Qt 偶发出现“底图不重绘”。用一个短延迟的重建作为兜底（不会频繁重建）。
        self._scene_rebuild_timer = QTimer(self)
        self._scene_rebuild_timer.setSingleShot(True)
        self._scene_rebuild_timer.setInterval(SCENE_REBUILD_DELAY_MS)
        self._scene_rebuild_timer.timeout.connect(self._rebuild_scene_keep_view)

        # OCR 引擎延迟初始化（首次识别才加载）
        self.ocr_engine = None
        self.ocr_loading = False

        # PPT导出配置
        self.use_text_bg = False  # 是否使用文本框背景色（默认关闭）
        self.text_bg_color = QColor(255, 255, 255)  # 默认白色背景
        # 默认透明度降低一些，避免在画布上遮住原图；导出前可在“视图-背景透明度”调高
        self.text_bg_alpha = DEFAULT_BG_ALPHA
        self._user_set_global_bg_alpha = False
        self.eyedropper_mode = False  # 吸管模式
        self._ppt_exporter_metrics = None  # lazy, for fit_font_size used by canvas preview
        self._show_left_panel = True
        self._show_right_panel = True
        # IOPaint variants (non-destructive): original_path -> inpainted_path
        self.inpaint_variants = {}
        self.show_inpaint_preview = False
        # Optional user-defined ROI per image: image_path -> [x, y, w, h]
        self.roi_by_image = {}
        self.roi_select_mode = False
        self._roi_drag_start = None
        self._roi_item = None

        self.init_ui()

        # 定时尝试清理预览产生的临时 PPT（用户关闭 Office 后可自动删除）
        self._preview_cleanup_timer = QTimer(self)
        self._preview_cleanup_timer.setInterval(60_000)
        self._preview_cleanup_timer.timeout.connect(self._cleanup_preview_ppts)
        self._preview_cleanup_timer.start()

    def open_github_repo(self, *args):
        url = "https://github.com/Tansuo2021/OCRPDF-TO-PPT"
        try:
            QDesktopServices.openUrl(QUrl(url))
            return
        except Exception:
            pass
        try:
            import webbrowser

            webbrowser.open(url)
        except Exception as e:
            try:
                QMessageBox.warning(
                    self,
                    self._t("提示", "Info"),
                    self._t(f"无法打开浏览器：{e}\n\n{url}", f"Failed to open browser: {e}\n\n{url}"),
                )
            except Exception:
                pass

    def _resolve_ui_lang(self, pref: str) -> str:
        """Return effective UI language ("zh" or "en") based on user preference."""
        p = (pref or "").strip().lower().replace("_", "-")
        if p in ("zh", "zh-cn", "cn", "chinese"):
            return "zh"
        if p in ("en", "en-us", "english"):
            return "en"

        # auto/system
        try:
            loc = (QLocale.system().name() or "").lower()  # e.g. "zh_CN"
            if loc.startswith("zh"):
                return "zh"
        except Exception:
            pass
        return "en"

    def _t(self, zh: str, en: str = None) -> str:
        """Tiny i18n helper: choose between Chinese / English strings."""
        try:
            if str(getattr(self, "ui_lang", "zh")).lower().startswith("en"):
                return en if en is not None else zh
        except Exception:
            pass
        return zh

    def set_ui_lang(self, pref: str):
        """Persist UI language preference; effective language is resolved immediately."""
        pref = (pref or "auto").strip()
        self.settings["ui_lang"] = pref
        self.ui_lang_setting = pref
        self.ui_lang = self._resolve_ui_lang(pref)
        global _UI_LANG
        _UI_LANG = self.ui_lang
        self.save_settings()

    def _update_lang_button_styles(self):
        """更新语言按钮的选中状态样式"""
        current_lang = getattr(self, "_current_ui_lang", "auto")

        # 选中状态样式（红色背景 + 白色文字）
        selected_style = f"""
            QToolButton {{
                background-color: {PPT_THEME_RED};
                color: white;
                border: 2px solid {PPT_THEME_RED};
                border-radius: 4px;
                padding-top: 4px;
                font-size: 11px;
                font-weight: bold;
            }}
            QToolButton:hover {{
                background-color: #B8381F;
                border: 2px solid #B8381F;
            }}
        """

        # 未选中状态样式（灰色边框 + 黑色文字）
        unselected_style = """
            QToolButton {
                background-color: #F8F8F8;
                color: #333;
                border: 2px solid #CCC;
                border-radius: 4px;
                padding-top: 4px;
                font-size: 11px;
            }
            QToolButton:hover {
                background-color: #E8E8E8;
                border: 2px solid #999;
            }
        """

        # 应用样式
        for btn, lang in [(self.btn_lang_auto, "auto"), (self.btn_lang_zh, "zh"), (self.btn_lang_en, "en")]:
            if lang == current_lang:
                btn.setStyleSheet(selected_style)
            else:
                btn.setStyleSheet(unselected_style)

    def on_ui_lang_btn_clicked(self, lang_code):
        """处理语言按钮点击事件"""
        old = str(self.settings.get("ui_lang") or "auto").strip()
        if lang_code == old:
            return

        self.set_ui_lang(lang_code)

        # 更新当前语言并刷新按钮样式
        self._current_ui_lang = lang_code
        self._update_lang_button_styles()

        QMessageBox.information(
            self,
            self._t("提示", "Info"),
            self._t("界面语言已保存，重启软件后生效。", "UI language saved. Restart the app to apply."),
        )

    def on_ui_lang_changed(self, *_):
        """保留旧的下拉框处理函数以兼容（如果有其他地方调用）"""
        cb = getattr(self, "combo_ui_lang", None)
        if cb is None:
            return
        pref = cb.currentData() or "auto"
        old = str(self.settings.get("ui_lang") or "auto").strip()
        if pref == old:
            return

        self.set_ui_lang(pref)
        QMessageBox.information(
            self,
            self._t("提示", "Info"),
            self._t("界面语言已保存，重启软件后生效。", "UI language saved. Restart the app to apply."),
        )

    def load_settings(self) -> dict:
        defaults = {
            # UI language preference:
            # - "auto": follow system language (zh -> Chinese UI, otherwise English UI)
            # - "zh": force Chinese UI
            # - "en": force English UI
            "ui_lang": "auto",
            "ocr_use_gpu": False,
            # PaddleOCR 3.x (PaddleX) 的缓存根目录。模型会下载到：<PADDLE_PDX_CACHE_HOME>/official_models/...
            # 建议设置为“某个文件夹/.paddlex”，不要直接指向 official_models。
            "ocr_paddlex_home": "",
            # 可选：指定本地检测/识别模型目录（存在才生效）
            "ocr_det_dir": "",
            "ocr_rec_dir": "",
            # IOPaint (inpaint) settings
            "inpaint_enabled": True,
            "inpaint_api_url": "http://127.0.0.1:8080/api/v1/inpaint",
            "inpaint_box_padding": 6,
            "inpaint_crop_padding": 128,
            "inpaint_fill_pad_x": 10,
            "inpaint_fill_pad_y": 6,
            "inpaint_remote_pad_x": 8,
            "inpaint_remote_pad_y": 4,
        }
        try:
            if os.path.exists(self.settings_path):
                with open(self.settings_path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                # 兼容旧 key：ocr_model_cache_dir
                if "ocr_paddlex_home" not in data and "ocr_model_cache_dir" in data:
                    data["ocr_paddlex_home"] = data.get("ocr_model_cache_dir", "")
                defaults.update({k: data.get(k, v) for k, v in defaults.items()})
        except Exception:
            pass
        return defaults

    def save_settings(self):
        try:
            with open(self.settings_path, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"保存设置失败: {e}")

    def _apply_ocr_env(self):
        # Work around a PaddlePaddle OneDNN + PIR limitation that can raise:
        # NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [...]
        # Disabling OneDNN is the most reliable fix on Windows/CPU.
        os.environ["FLAGS_use_mkldnn"] = "0"
        os.environ["FLAGS_use_onednn"] = "0"

        cache_dir = (self.settings.get("ocr_paddlex_home") or "").strip()
        if not cache_dir:
            # 默认放到项目目录，方便拷贝到其他电脑
            proj_dir = os.path.dirname(__file__)
            cache_dir = "model" if os.path.isdir(os.path.join(proj_dir, "model")) else ".paddlex"

        # 相对路径按项目目录解析，方便整体拷贝到其他电脑
        if not os.path.isabs(cache_dir):
            cache_dir = os.path.join(os.path.dirname(__file__), cache_dir)

        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        os.makedirs(cache_dir, exist_ok=True)

        # PaddleOCR 3.x 底层会使用 PaddleX 的缓存目录（默认 ~/.paddlex）。
        # PaddleX 3.x 实际读取的环境变量是 `PADDLE_PDX_CACHE_HOME`（而不是 PADDLEX_HOME）。
        os.environ["PADDLE_PDX_CACHE_HOME"] = cache_dir
        # 兼容旧版本/其他封装里读取的变量名（不一定会用到，但保留无害）
        os.environ["PADDLEX_HOME"] = cache_dir
        os.environ["PADDLE_HOME"] = cache_dir
        os.environ["PADDLEOCR_HOME"] = cache_dir

    def _clear_paddlex_official_models(self, paddlex_home: str):
        """清空 <PADDLE_PDX_CACHE_HOME>/official_models 以强制重新下载"""
        try:
            p = os.path.join(paddlex_home, "official_models")
            if os.path.exists(p):
                shutil.rmtree(p, ignore_errors=True)
        except Exception as e:
            print(f"清空 official_models 失败: {e}")

    def _purge_ocr_modules(self):
        """清理已导入的 paddleocr/paddlex/ocr_engine 模块，确保切换 PADDLEX_HOME 后能生效"""
        try:
            import sys as _sys
            for k in list(_sys.modules.keys()):
                if k == "ocr_engine" or k.startswith("ocr_engine."):
                    _sys.modules.pop(k, None)
                if k == "paddleocr" or k.startswith("paddleocr."):
                    _sys.modules.pop(k, None)
                if k == "paddlex" or k.startswith("paddlex."):
                    _sys.modules.pop(k, None)
        except Exception:
            pass

    def reset_ocr_engine(self, *args):
        """清空 OCR 引擎（下次识别会重新加载/下载模型）"""
        self.ocr_engine = None

    def force_reload_ocr_engine(self, *args):
        """强制重新加载 OCR（会 purge 模块，适合切换模型目录后使用）"""
        global OCREngine
        self.ocr_engine = None
        OCREngine = None
        self._purge_ocr_modules()

    def ensure_ocr_engine(self) -> bool:
        """确保 OCR 引擎已加载；按设置选择目录，并延迟到用户真正识别时初始化"""
        if self.ocr_engine is not None:
            return True
        if self.ocr_loading:
            return False

        self.ocr_loading = True
        progress = QProgressDialog(
            self._t("正在初始化 OCR 引擎（首次可能下载模型，请稍候）...", "Initializing OCR engine (first run may download models)..."),
            "",
            0,
            0,
            self,
        )
        progress.setWindowTitle(self._t("初始化 OCR", "Initializing OCR"))
        progress.setCancelButton(None)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setRange(0, 0)
        progress.show()
        QApplication.processEvents()

        try:
            # OCR 依赖在 import/初始化时可能 spawn subprocess，导致黑色控制台一闪而过（Windows 常见）。
            # 仅在 OCR init 阶段临时抑制该行为，避免影响其它功能（例如预览 PPT）。
            with suppress_windows_subprocess_console():
                _try_use_pythonw_for_multiprocessing()
                self._apply_ocr_env()
                # 延迟 import：确保缓存目录在 import paddleocr/paddlex 前就已设置
                global OCREngine
                if OCREngine is None:
                    from ocr_engine import OCREngine as _OCREngine
                    OCREngine = _OCREngine
                use_gpu = bool(self.settings.get("ocr_use_gpu", False))
                det_dir = (self.settings.get("ocr_det_dir") or "").strip() or None
                rec_dir = (self.settings.get("ocr_rec_dir") or "").strip() or None
                self.ocr_engine = OCREngine(use_gpu=use_gpu, model_det_dir=det_dir, model_rec_dir=rec_dir)
            return True
        except Exception as e:
            self.ocr_engine = None
            QMessageBox.critical(
                self,
                self._t("OCR 初始化失败", "OCR init failed"),
                self._t(
                    f"{e}\n\n你可以在【设置】里选择模型缓存目录（PADDLE_PDX_CACHE_HOME）或手动指定 det/rec 模型目录。",
                    f"{e}\n\nYou can set the model cache folder (PADDLE_PDX_CACHE_HOME) or specify det/rec model dirs in Settings.",
                ),
            )
            return False
        finally:
            self.ocr_loading = False
            progress.close()

    def init_ocr_engine(self):
        """兼容旧调用：现在改为延迟加载"""
        return self.ensure_ocr_engine()

    def open_ocr_settings(self, *args):
        """顶部菜单：OCR 模型/缓存目录设置"""
        from PySide6.QtWidgets import QDialog, QFormLayout, QLineEdit

        dlg = QDialog(self)
        dlg.setWindowTitle(self._t("OCR 设置", "OCR Settings"))
        dlg.setModal(True)
        dlg.setMinimumWidth(520)

        form = QFormLayout(dlg)

        default_rel_home = "model" if os.path.isdir(os.path.join(os.path.dirname(__file__), "model")) else ".paddlex"
        current_default_home = os.path.join(os.path.dirname(__file__), default_rel_home)
        ed_cache = QLineEdit(str(self.settings.get("ocr_paddlex_home", "") or default_rel_home))
        btn_cache = QPushButton(self._t("选择目录", "Browse"))
        btn_cache.clicked.connect(lambda: ed_cache.setText(QFileDialog.getExistingDirectory(self, self._t("选择模型缓存目录（PADDLE_PDX_CACHE_HOME）", "Select model cache folder (PADDLE_PDX_CACHE_HOME)"), ed_cache.text() or current_default_home) or ed_cache.text()))
        btn_use_project = QPushButton(self._t("用项目目录", "Use project dir"))
        btn_use_project.clicked.connect(lambda: ed_cache.setText(default_rel_home))
        row_cache = QWidget()
        hl = QHBoxLayout(row_cache)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(ed_cache)
        hl.addWidget(btn_cache)
        hl.addWidget(btn_use_project)
        form.addRow(self._t("模型缓存目录（PADDLE_PDX_CACHE_HOME，会生成 official_models）", "Model cache dir (PADDLE_PDX_CACHE_HOME; will create official_models/)"), row_cache)

        ed_det = QLineEdit(str(self.settings.get("ocr_det_dir", "")))
        btn_det = QPushButton(self._t("选择目录", "Browse"))
        btn_det.clicked.connect(lambda: ed_det.setText(QFileDialog.getExistingDirectory(self, self._t("选择检测模型目录(det)", "Select detection model dir (det)"), ed_det.text() or os.getcwd()) or ed_det.text()))
        row_det = QWidget()
        hl = QHBoxLayout(row_det)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(ed_det)
        hl.addWidget(btn_det)
        form.addRow(self._t("检测模型目录（可选）", "Detection model dir (optional)"), row_det)

        ed_rec = QLineEdit(str(self.settings.get("ocr_rec_dir", "")))
        btn_rec = QPushButton(self._t("选择目录", "Browse"))
        btn_rec.clicked.connect(lambda: ed_rec.setText(QFileDialog.getExistingDirectory(self, self._t("选择识别模型目录(rec)", "Select recognition model dir (rec)"), ed_rec.text() or os.getcwd()) or ed_rec.text()))
        row_rec = QWidget()
        hl = QHBoxLayout(row_rec)
        hl.setContentsMargins(0, 0, 0, 0)
        hl.addWidget(ed_rec)
        hl.addWidget(btn_rec)
        form.addRow(self._t("识别模型目录（可选）", "Recognition model dir (optional)"), row_rec)

        chk_gpu = QCheckBox(self._t("使用 GPU（若不可用会自动回退 CPU）", "Use GPU (auto fallback to CPU)"))
        chk_gpu.setChecked(bool(self.settings.get("ocr_use_gpu", False)))
        form.addRow(self._t("设备", "Device"), chk_gpu)

        chk_redownload = QCheckBox(self._t("清空目标目录的 official_models（强制重新下载）", "Delete official_models (force re-download)"))
        chk_redownload.setChecked(False)
        form.addRow("", chk_redownload)

        tips = QLabel(self._t(
            "说明：程序启动不再初始化模型；首次点【OCR识别】时才会加载。\n"
            "PaddleOCR 3.x 会把模型下载到：<PADDLE_PDX_CACHE_HOME>/official_models。\n"
            "不要把缓存目录直接指向 official_models。",
            "Notes:\n"
            "- OCR engine is loaded lazily (first OCR run).\n"
            "- PaddleOCR 3.x downloads models to: <PADDLE_PDX_CACHE_HOME>/official_models.\n"
            "- Do NOT set the cache folder to official_models directly.",
        ))
        tips.setStyleSheet("color:#555;")
        form.addRow("", tips)

        btn_row = QWidget()
        hl2 = QHBoxLayout(btn_row)
        hl2.setContentsMargins(0, 0, 0, 0)
        hl2.addStretch()
        btn_reload = QPushButton(self._t("保存并重新加载OCR", "Save & Reload"))
        btn_ok = QPushButton(self._t("保存", "Save"))
        btn_cancel = QPushButton(self._t("取消", "Cancel"))
        hl2.addWidget(btn_reload)
        hl2.addWidget(btn_ok)
        hl2.addWidget(btn_cancel)
        form.addRow("", btn_row)

        def save_and_close(reload_engine: bool):
            cache_text = (ed_cache.text() or "").strip()
            # 防呆：如果用户选到了 official_models，就自动上移一级
            if cache_text.replace("/", "\\").rstrip("\\").lower().endswith("\\official_models"):
                cache_text = os.path.dirname(cache_text)
            # 如果选择的是项目目录下的 .paddlex，则保存相对路径，方便迁移
            try:
                abs_cache = cache_text
                if not os.path.isabs(abs_cache):
                    abs_cache = os.path.join(os.path.dirname(__file__), abs_cache)
                abs_cache = os.path.abspath(os.path.expanduser(abs_cache))
                if os.path.normcase(abs_cache) == os.path.normcase(os.path.abspath(current_default_home)):
                    cache_text = default_rel_home
            except Exception:
                pass
            self.settings["ocr_paddlex_home"] = cache_text
            self.settings["ocr_det_dir"] = ed_det.text().strip()
            self.settings["ocr_rec_dir"] = ed_rec.text().strip()
            self.settings["ocr_use_gpu"] = chk_gpu.isChecked()
            self.save_settings()
            if reload_engine:
                self._apply_ocr_env()
                if chk_redownload.isChecked():
                    home = os.environ.get("PADDLE_PDX_CACHE_HOME") or os.environ.get("PADDLEX_HOME", "")
                    self._clear_paddlex_official_models(home)
                self.force_reload_ocr_engine()
            dlg.accept()

        btn_ok.clicked.connect(lambda: save_and_close(False))
        btn_reload.clicked.connect(lambda: save_and_close(True))
        btn_cancel.clicked.connect(dlg.reject)

        dlg.exec()

    def open_inpaint_settings(self, *args):
        """Settings dialog for hybrid text removal used to clean the background."""
        from PySide6.QtWidgets import QDialog, QFormLayout, QLineEdit, QSpinBox

        dlg = QDialog(self)
        dlg.setWindowTitle(self._t("去字设置", "Clean Background Settings"))
        dlg.setModal(True)
        dlg.setMinimumWidth(520)

        form = QFormLayout(dlg)

        chk_enabled = QCheckBox(self._t("启用去字功能（导出前可生成纯背景底图）", "Enable text clean background generation"))
        chk_enabled.setChecked(bool(self.settings.get("inpaint_enabled", True)))
        form.addRow(self._t("开关", "Enable"), chk_enabled)

        ed_url = QLineEdit(str(self.settings.get("inpaint_api_url", "") or "http://127.0.0.1:8080/api/v1/inpaint"))
        form.addRow(self._t("IOPaint API 地址", "IOPaint API URL"), ed_url)

        sp_box_pad = QSpinBox()
        sp_box_pad.setRange(0, BOX_PADDING_MAX)
        sp_box_pad.setValue(int(self.settings.get("inpaint_box_padding", 6) or 6))
        sp_box_pad.setSuffix(" px")
        form.addRow(self._t("文本框外扩（遮罩）", "Box padding (mask)"), sp_box_pad)

        sp_crop_pad = QSpinBox()
        sp_crop_pad.setRange(0, CROP_PADDING_MAX)
        sp_crop_pad.setValue(int(self.settings.get("inpaint_crop_padding", 128) or 128))
        sp_crop_pad.setSuffix(" px")
        form.addRow(self._t("裁剪外扩（API加速）", "Crop padding (API speed-up)"), sp_crop_pad)

        sp_fill_pad_x = QSpinBox()
        sp_fill_pad_x.setRange(0, BOX_PADDING_MAX)
        sp_fill_pad_x.setValue(int(self.settings.get("inpaint_fill_pad_x", 10) or 10))
        sp_fill_pad_x.setSuffix(" px")
        form.addRow(self._t("单色覆盖横向外扩", "Solid fill extra width"), sp_fill_pad_x)

        sp_fill_pad_y = QSpinBox()
        sp_fill_pad_y.setRange(0, BOX_PADDING_MAX)
        sp_fill_pad_y.setValue(int(self.settings.get("inpaint_fill_pad_y", 6) or 6))
        sp_fill_pad_y.setSuffix(" px")
        form.addRow(self._t("单色覆盖纵向外扩", "Solid fill extra height"), sp_fill_pad_y)

        sp_remote_pad_x = QSpinBox()
        sp_remote_pad_x.setRange(0, BOX_PADDING_MAX)
        sp_remote_pad_x.setValue(int(self.settings.get("inpaint_remote_pad_x", 8) or 8))
        sp_remote_pad_x.setSuffix(" px")
        form.addRow(self._t("IOPaint 横向外扩", "IOPaint extra width"), sp_remote_pad_x)

        sp_remote_pad_y = QSpinBox()
        sp_remote_pad_y.setRange(0, BOX_PADDING_MAX)
        sp_remote_pad_y.setValue(int(self.settings.get("inpaint_remote_pad_y", 4) or 4))
        sp_remote_pad_y.setSuffix(" px")
        form.addRow(self._t("IOPaint 纵向外扩", "IOPaint extra height"), sp_remote_pad_y)

        tips = QLabel(self._t(
            "说明：会根据 OCR 文本框自动生成遮罩（优先使用 polygon），支持三种入口。\n"
            "- 智能去字：简单背景优先单色覆盖，复杂纹理/照片区域自动使用 IOPaint\n"
            "- 单色覆盖去字：直接提取周边主色覆盖文字，适合白底/纯色区域\n"
            "- IOPaint 去字：强制调用 IOPaint，适合复杂纹理或用户明确指定的框\n"
            "右侧可对单个 OCR 框设置“参与去字 / 去字方式”。\n"
            "若单色覆盖仍有边缘残留，可增加“单色覆盖横向外扩 / 纵向外扩”；\n"
            "若 IOPaint 仍有漏字，可增加“IOPaint 横向外扩 / 纵向外扩”。\n"
            "支持多个 API 地址：用分号/逗号分隔（例如：url1;url2），复杂区域可并行加速。\n"
            "建议先在命令行启动服务：\n"
            "  iopaint start --host 127.0.0.1 --port 8080\n",
            "Notes:\n"
            "- OCR boxes are converted to a mask automatically (polygon first).\n"
            "- Smart clean uses solid color fill for simple regions, and IOPaint for complex textured/photo regions.\n"
            "- Solid fill clean samples the nearby dominant background color and covers the text directly.\n"
            "- IOPaint clean forces the remote API and is suitable for complex regions or boxes explicitly set to IOPaint.\n"
            "- Each OCR box can be configured individually from the right-side panel.\n"
            "- If solid fill leaves edge remnants, increase the solid fill extra width / height.\n"
            "- If IOPaint still leaves text behind, increase the IOPaint extra width / height.\n"
            "- Multiple endpoints are supported: separate with ';' or ',' (e.g. url1;url2) for complex-region parallel speed-up.\n"
            "- Start the service first:\n"
            "  iopaint start --host 127.0.0.1 --port 8080\n",
        ))
        tips.setStyleSheet("color:#555;")
        form.addRow("", tips)

        btn_row = QWidget()
        hl2 = QHBoxLayout(btn_row)
        hl2.setContentsMargins(0, 0, 0, 0)
        hl2.addStretch()
        btn_ok = QPushButton(self._t("保存", "Save"))
        btn_cancel = QPushButton(self._t("取消", "Cancel"))
        hl2.addWidget(btn_ok)
        hl2.addWidget(btn_cancel)
        form.addRow("", btn_row)

        def save_and_close():
            self.settings["inpaint_enabled"] = chk_enabled.isChecked()
            self.settings["inpaint_api_url"] = (ed_url.text() or "").strip()
            self.settings["inpaint_box_padding"] = int(sp_box_pad.value())
            self.settings["inpaint_crop_padding"] = int(sp_crop_pad.value())
            self.settings["inpaint_fill_pad_x"] = int(sp_fill_pad_x.value())
            self.settings["inpaint_fill_pad_y"] = int(sp_fill_pad_y.value())
            self.settings["inpaint_remote_pad_x"] = int(sp_remote_pad_x.value())
            self.settings["inpaint_remote_pad_y"] = int(sp_remote_pad_y.value())
            self.save_settings()
            dlg.accept()

        btn_ok.clicked.connect(save_and_close)
        btn_cancel.clicked.connect(dlg.reject)
        dlg.exec()

    def _add_image_item(self, path: str):
        """向左侧缩略图列表添加一项，并同步 images/box_data 的默认结构"""
        self.images.append(path)
        self.box_data.setdefault(path, [])

        item = QListWidgetItem()
        item.setSizeHint(QSize(200, 140))
        self.list_thumb.addItem(item)

        w = QWidget()
        vl = QVBoxLayout(w)
        vl.setContentsMargins(15, 5, 15, 5)
        lb_idx = QLabel(str(len(self.images)))
        lb_idx.setObjectName("ThumbIndex")
        lb_idx.setStyleSheet("font-size: 10px; color: #555;")
        pix = QPixmap(self._get_display_image_path(path)).scaled(180, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        lb_img = QLabel()
        lb_img.setObjectName("ThumbImage")
        lb_img.setPixmap(pix)
        lb_img.setStyleSheet("border: 1px solid #BBB; background: white;")
        vl.addWidget(lb_idx)
        vl.addWidget(lb_img, 0, Qt.AlignCenter)
        self.list_thumb.setItemWidget(item, w)

    def _rebuild_thumb_list(self, select_index=None):
        """重建左侧缩略图列表（用于删除/复制/移动页之后）"""
        if select_index is None:
            select_index = self.list_thumb.currentRow()

        self.list_thumb.blockSignals(True)
        self.list_thumb.clear()
        for idx, p in enumerate(self.images, start=1):
            item = QListWidgetItem()
            item.setSizeHint(QSize(200, 140))
            self.list_thumb.addItem(item)

            w = QWidget()
            vl = QVBoxLayout(w)
            vl.setContentsMargins(15, 5, 15, 5)
            lb_idx = QLabel(str(idx))
            lb_idx.setObjectName("ThumbIndex")
            lb_idx.setStyleSheet("font-size: 10px; color: #555;")
            pix = QPixmap(self._get_display_image_path(p)).scaled(180, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            lb_img = QLabel()
            lb_img.setObjectName("ThumbImage")
            lb_img.setPixmap(pix)
            lb_img.setStyleSheet("border: 1px solid #BBB; background: white;")
            vl.addWidget(lb_idx)
            vl.addWidget(lb_img, 0, Qt.AlignCenter)
            self.list_thumb.setItemWidget(item, w)
        self.list_thumb.blockSignals(False)

        if self.images:
            select_index = max(0, min(select_index, len(self.images) - 1))
            self.list_thumb.setCurrentRow(select_index)
        else:
            self.current_img = None
            self.scene.clear()
        self.update_status()

    def _refresh_thumb_images(self):
        """刷新左侧缩略图（用于原图/去字图预览切换等不改变页数的操作）。"""
        try:
            for idx, p in enumerate(self.images):
                item = self.list_thumb.item(idx)
                if item is None:
                    continue
                w = self.list_thumb.itemWidget(item)
                if w is None:
                    continue
                lb_idx = w.findChild(QLabel, "ThumbIndex")
                if lb_idx is not None:
                    try:
                        lb_idx.setText(str(idx + 1))
                    except Exception:
                        pass
                lb_img = w.findChild(QLabel, "ThumbImage")
                if lb_img is None:
                    continue
                pix = QPixmap(self._get_display_image_path(p)).scaled(180, 100, Qt.KeepAspectRatio, Qt.SmoothTransformation)
                lb_img.setPixmap(pix)
        except Exception:
            pass

    def _get_display_image_path(self, image_path: str) -> str:
        """Return the image path used for UI preview (original vs inpainted variant)."""
        p = str(image_path or "")
        if not p:
            return p
        if bool(getattr(self, "show_inpaint_preview", False)):
            v = (getattr(self, "inpaint_variants", {}) or {}).get(p)
            if v and os.path.exists(v):
                return v
        return p

    def _get_export_image_path(self, image_path: str) -> str:
        """Return the image path used for PPT export (prefer inpainted variant when available)."""
        p = str(image_path or "")
        if not p:
            return p
        v = (getattr(self, "inpaint_variants", {}) or {}).get(p)
        if v and os.path.exists(v):
            return v
        return p

    def _has_any_inpaint_variant(self) -> bool:
        try:
            m = getattr(self, "inpaint_variants", {}) or {}
            for _, dst in (m.items() if isinstance(m, dict) else []):
                if dst and os.path.exists(str(dst)):
                    return True
        except Exception:
            pass
        return False

    def _sync_inpaint_preview_toggle(self):
        """Keep the ribbon toggle button state in sync with show_inpaint_preview."""
        btn = getattr(self, "btn_inpaint_preview", None)
        if btn is None:
            return
        try:
            want = bool(getattr(self, "show_inpaint_preview", False))
            if bool(btn.isChecked()) != want:
                btn.blockSignals(True)
                btn.setChecked(want)
                btn.blockSignals(False)
        except Exception:
            pass

    def set_inpaint_preview(self, enabled: bool):
        """Toggle UI preview between original and inpainted variant (non-destructive)."""
        enabled = bool(enabled)
        if enabled and (not self._has_any_inpaint_variant()):
            # No variants yet: don't enter a confusing "checked but unchanged" state.
            try:
                QMessageBox.information(
                    self,
                    self._t("提示", "Info"),
                    self._t(
                        "还没有生成“去字底图”。\n请先点击【去字-本页/全部】。",
                        'No "clean background" has been generated yet.\nPlease run Clean (Current/All) first.',
                    ),
                )
            except Exception:
                pass
            enabled = False

        if bool(getattr(self, "show_inpaint_preview", False)) == enabled:
            self._sync_inpaint_preview_toggle()
            return

        self.show_inpaint_preview = enabled
        self._sync_inpaint_preview_toggle()
        self._refresh_thumb_images()
        try:
            self._rebuild_scene_keep_view()
        except Exception:
            pass

    def toggle_inpaint_preview(self, *args):
        self.set_inpaint_preview(not bool(getattr(self, "show_inpaint_preview", False)))

    def clear_inpaint_variant_current(self, *args):
        """Restore original for current slide by clearing its inpaint variant mapping."""
        if not self.current_img:
            return
        m = getattr(self, "inpaint_variants", {}) or {}
        cur = str(self.current_img)
        if not (isinstance(m, dict) and cur in m and m.get(cur)):
            try:
                QMessageBox.information(
                    self,
                    self._t("提示", "Info"),
                    self._t("当前页没有“去字底图”。", 'No "clean background" for the current slide.'),
                )
            except Exception:
                pass
            return

        try:
            self.push_undo()
        except Exception:
            pass
        try:
            m.pop(cur, None)
            self.inpaint_variants = m
        except Exception:
            pass

        # If no variants remain, also turn off preview mode.
        if not self._has_any_inpaint_variant():
            self.show_inpaint_preview = False

        self._sync_inpaint_preview_toggle()
        self._refresh_thumb_images()
        try:
            self._rebuild_scene_keep_view()
        except Exception:
            pass

    def _snapshot_state(self):
        try:
            curr_idx = self.images.index(self.current_img) if self.current_img in self.images else self.list_thumb.currentRow()
        except Exception:
            curr_idx = self.list_thumb.currentRow()
        return {
            "kind": "full",
            "images": list(self.images),
            "box_data": copy.deepcopy(self.box_data),
            "inpaint_variants": dict(getattr(self, "inpaint_variants", {}) or {}),
            "show_inpaint_preview": bool(getattr(self, "show_inpaint_preview", False)),
            "roi_by_image": copy.deepcopy(getattr(self, "roi_by_image", {}) or {}),
            "current_index": int(curr_idx) if curr_idx is not None else -1,
        }

    def _snapshot_current_slide_state(self, image_path=None):
        image_path = image_path or self.current_img
        if not image_path:
            return self._snapshot_state()
        try:
            curr_idx = self.images.index(image_path)
        except Exception:
            curr_idx = self.list_thumb.currentRow()
        roi_value = copy.deepcopy((getattr(self, "roi_by_image", {}) or {}).get(image_path))
        return {
            "kind": "slide",
            "image_path": image_path,
            "boxes": copy.deepcopy((self.box_data or {}).get(image_path, []) or []),
            "roi": roi_value,
            "current_index": int(curr_idx) if curr_idx is not None else -1,
        }

    def push_undo(self):
        """保存当前状态到撤销栈（用于 Ctrl+Z）"""
        self.undo_stack.append(self._snapshot_state())
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def push_undo_current_slide(self, image_path=None):
        """仅保存当前页状态，避免频繁编辑时深拷贝整个项目。"""
        self.undo_stack.append(self._snapshot_current_slide_state(image_path=image_path))
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    def _snapshot_for_history_entry(self, entry):
        kind = str((entry or {}).get("kind") or "full")
        if kind == "slide":
            return self._snapshot_current_slide_state(entry.get("image_path"))
        return self._snapshot_state()

    def _restore_state(self, snap):
        kind = str((snap or {}).get("kind") or "full")
        if kind == "slide":
            image_path = snap.get("image_path")
            if image_path in self.images:
                self.box_data[image_path] = copy.deepcopy(snap.get("boxes", []) or [])
                roi_map = getattr(self, "roi_by_image", {}) or {}
                roi_value = copy.deepcopy(snap.get("roi"))
                if roi_value is None:
                    roi_map.pop(image_path, None)
                else:
                    roi_map[image_path] = roi_value
                self.roi_by_image = roi_map
                idx = int(snap.get("current_index", -1))
                if 0 <= idx < len(self.images):
                    self.list_thumb.setCurrentRow(idx)
                    self.switch_slide(idx)
                elif self.current_img == image_path:
                    try:
                        self._rebuild_scene_keep_view()
                    except Exception:
                        self.switch_slide(self.list_thumb.currentRow())
            return
        self.images = list(snap.get("images", []))
        self.box_data = snap.get("box_data", {}) or {}
        self.inpaint_variants = snap.get("inpaint_variants", {}) or {}
        self.show_inpaint_preview = bool(snap.get("show_inpaint_preview", False))
        self.roi_by_image = snap.get("roi_by_image", {}) or {}
        idx = int(snap.get("current_index", -1))
        self._rebuild_thumb_list(select_index=idx if idx >= 0 else 0)
        try:
            self._sync_inpaint_preview_toggle()
        except Exception:
            pass

    def undo(self, *args):
        if not self.undo_stack:
            return
        snap = self.undo_stack.pop()
        self.redo_stack.append(self._snapshot_for_history_entry(snap))
        self._restore_state(snap)

    def redo(self, *args):
        if not self.redo_stack:
            return
        snap = self.redo_stack.pop()
        self.undo_stack.append(self._snapshot_for_history_entry(snap))
        self._restore_state(snap)

    def init_ui(self):
        # 顶部不再使用菜单栏（和“开始/视图”风格统一），但保留快捷键
        try:
            self.menuBar().hide()
        except Exception:
            pass

        self.act_undo = QAction(self._t("撤销", "Undo"), self)
        self.act_undo.setShortcut(QKeySequence.Undo)
        self.act_undo.triggered.connect(self.undo)
        self.addAction(self.act_undo)

        self.act_redo = QAction(self._t("重做", "Redo"), self)
        self.act_redo.setShortcut(QKeySequence.Redo)
        self.act_redo.triggered.connect(self.redo)
        self.addAction(self.act_redo)

        self.act_cut = QAction(self._t("剪切", "Cut"), self)
        self.act_cut.setShortcut(QKeySequence.Cut)
        self.act_cut.triggered.connect(self.cut_selected_box)
        self.addAction(self.act_cut)

        self.act_copy = QAction(self._t("复制", "Copy"), self)
        self.act_copy.setShortcut(QKeySequence.Copy)
        self.act_copy.triggered.connect(self.copy_selected_box)
        self.addAction(self.act_copy)

        self.act_paste = QAction(self._t("粘贴", "Paste"), self)
        self.act_paste.setShortcut(QKeySequence.Paste)
        self.act_paste.triggered.connect(self.paste_box)
        self.addAction(self.act_paste)

        self.act_paste_image = QAction(self._t("粘贴图片", "Paste Image"), self)
        self.act_paste_image.setShortcut(QKeySequence("Ctrl+Shift+V"))
        self.act_paste_image.triggered.connect(self.paste_clipboard_image)
        self.addAction(self.act_paste_image)

        self.act_del_box = QAction(self._t("删除文本框", "Delete Text Box"), self)
        self.act_del_box.setShortcut(QKeySequence.Delete)
        self.act_del_box.triggered.connect(self.delete_box)
        self.addAction(self.act_del_box)

        # Extra shortcuts (PowerPoint-like workflow)
        self.act_import_images = QAction(self._t("导入图片", "Import Images"), self)
        self.act_import_images.setShortcut(QKeySequence("Ctrl+O"))
        self.act_import_images.triggered.connect(self.import_images)
        self.addAction(self.act_import_images)

        self.act_import_pdfs = QAction(self._t("导入PDF", "Import PDF"), self)
        self.act_import_pdfs.setShortcut(QKeySequence("Ctrl+Shift+O"))
        self.act_import_pdfs.triggered.connect(self.import_pdfs)
        self.addAction(self.act_import_pdfs)

        self.act_export_ppt = QAction(self._t("导出PPT", "Export PPT"), self)
        self.act_export_ppt.setShortcut(QKeySequence("Ctrl+S"))
        self.act_export_ppt.triggered.connect(self.export_ppt)
        self.addAction(self.act_export_ppt)

        self.act_preview_ppt = QAction(self._t("预览PPT", "Preview PPT"), self)
        self.act_preview_ppt.setShortcut(QKeySequence("F5"))
        self.act_preview_ppt.triggered.connect(self.preview_ppt)
        self.addAction(self.act_preview_ppt)

        self.act_ocr_current = QAction(self._t("OCR本页", "OCR Current"), self)
        self.act_ocr_current.setShortcuts([QKeySequence("Ctrl+Return"), QKeySequence("Ctrl+Enter")])
        self.act_ocr_current.triggered.connect(self.run_ocr_current_slide)
        self.addAction(self.act_ocr_current)

        self.act_ocr_all = QAction(self._t("OCR全部", "OCR All"), self)
        self.act_ocr_all.setShortcut(QKeySequence("Ctrl+R"))
        self.act_ocr_all.triggered.connect(self.run_ocr_all_images)
        self.addAction(self.act_ocr_all)

        self.act_inpaint_current = QAction(self._t("智能去字本页", "Smart Clean (Current)"), self)
        self.act_inpaint_current.setShortcut(QKeySequence("Ctrl+I"))
        self.act_inpaint_current.triggered.connect(self.inpaint_current_slide)
        self.addAction(self.act_inpaint_current)

        self.act_inpaint_all = QAction(self._t("智能去字全部", "Smart Clean (All)"), self)
        self.act_inpaint_all.setShortcut(QKeySequence("Ctrl+Shift+I"))
        self.act_inpaint_all.triggered.connect(self.inpaint_all_slides)
        self.addAction(self.act_inpaint_all)

        self.act_fill_current = QAction(self._t("单色覆盖去字本页", "Solid Fill Clean (Current)"), self)
        self.act_fill_current.triggered.connect(self.fill_current_slide)
        self.addAction(self.act_fill_current)

        self.act_fill_all = QAction(self._t("单色覆盖去字全部", "Solid Fill Clean (All)"), self)
        self.act_fill_all.triggered.connect(self.fill_all_slides)
        self.addAction(self.act_fill_all)

        self.act_remote_current = QAction(self._t("IOPaint去字本页", "IOPaint Clean (Current)"), self)
        self.act_remote_current.triggered.connect(self.remote_inpaint_current_slide)
        self.addAction(self.act_remote_current)

        self.act_remote_all = QAction(self._t("IOPaint去字全部", "IOPaint Clean (All)"), self)
        self.act_remote_all.triggered.connect(self.remote_inpaint_all_slides)
        self.addAction(self.act_remote_all)

        # Inpaint compare / ROI tools
        self.act_toggle_inpaint_preview = QAction(self._t("切换去字预览（原图/去字）", "Toggle Clean Preview (Orig/Clean)"), self)
        self.act_toggle_inpaint_preview.setShortcut(QKeySequence("Ctrl+Alt+B"))
        self.act_toggle_inpaint_preview.triggered.connect(self.toggle_inpaint_preview)
        self.addAction(self.act_toggle_inpaint_preview)

        self.act_clear_inpaint_current = QAction(self._t("恢复原图（清除本页去字图）", "Restore Original (Clear Clean BG)"), self)
        self.act_clear_inpaint_current.setShortcut(QKeySequence("Ctrl+Alt+Shift+B"))
        self.act_clear_inpaint_current.triggered.connect(self.clear_inpaint_variant_current)
        self.addAction(self.act_clear_inpaint_current)

        self.act_roi_select = QAction(self._t("框选选区（OCR/去字）", "Select ROI (OCR/Clean)"), self)
        self.act_roi_select.setShortcut(QKeySequence("Ctrl+Alt+A"))
        self.act_roi_select.triggered.connect(self.toggle_roi_select_mode)
        self.addAction(self.act_roi_select)

        self.act_roi_clear = QAction(self._t("清除选区", "Clear ROI"), self)
        self.act_roi_clear.setShortcut(QKeySequence("Ctrl+Alt+Shift+A"))
        self.act_roi_clear.triggered.connect(self.clear_roi_current)
        self.addAction(self.act_roi_clear)

        self.act_new_slide = QAction(self._t("新建空白页", "New Blank Slide"), self)
        self.act_new_slide.setShortcut(QKeySequence("Ctrl+N"))
        self.act_new_slide.triggered.connect(self.new_blank_slide)
        self.addAction(self.act_new_slide)

        self.act_dup_slide = QAction(self._t("复制当前页", "Duplicate Slide"), self)
        self.act_dup_slide.setShortcut(QKeySequence("Ctrl+D"))
        self.act_dup_slide.triggered.connect(self.duplicate_slide)
        self.addAction(self.act_dup_slide)

        self.act_del_slide = QAction(self._t("删除当前页", "Delete Slide"), self)
        self.act_del_slide.setShortcut(QKeySequence("Ctrl+Shift+Delete"))
        self.act_del_slide.triggered.connect(self.delete_slide)
        self.addAction(self.act_del_slide)

        self.act_move_slide_up = QAction(self._t("上移当前页", "Move Slide Up"), self)
        self.act_move_slide_up.setShortcut(QKeySequence("Alt+Up"))
        self.act_move_slide_up.triggered.connect(self.move_slide_up)
        self.addAction(self.act_move_slide_up)

        self.act_move_slide_down = QAction(self._t("下移当前页", "Move Slide Down"), self)
        self.act_move_slide_down.setShortcut(QKeySequence("Alt+Down"))
        self.act_move_slide_down.triggered.connect(self.move_slide_down)
        self.addAction(self.act_move_slide_down)

        self.act_prev_slide = QAction(self._t("上一页", "Previous Slide"), self)
        self.act_prev_slide.setShortcuts([QKeySequence("PageUp"), QKeySequence("Alt+Left")])
        self.act_prev_slide.triggered.connect(self.goto_prev_slide)
        self.addAction(self.act_prev_slide)

        self.act_next_slide = QAction(self._t("下一页", "Next Slide"), self)
        self.act_next_slide.setShortcuts([QKeySequence("PageDown"), QKeySequence("Alt+Right")])
        self.act_next_slide.triggered.connect(self.goto_next_slide)
        self.addAction(self.act_next_slide)

        self.act_fit_view = QAction(self._t("适应窗口", "Fit to Window"), self)
        self.act_fit_view.setShortcut(QKeySequence("Ctrl+0"))
        self.act_fit_view.triggered.connect(self.fit_view_to_window)
        self.addAction(self.act_fit_view)

        self.act_zoom_in = QAction(self._t("放大", "Zoom In"), self)
        self.act_zoom_in.setShortcut(QKeySequence.ZoomIn)
        self.act_zoom_in.triggered.connect(self.zoom_in)
        self.addAction(self.act_zoom_in)

        self.act_zoom_out = QAction(self._t("缩小", "Zoom Out"), self)
        self.act_zoom_out.setShortcut(QKeySequence.ZoomOut)
        self.act_zoom_out.triggered.connect(self.zoom_out)
        self.addAction(self.act_zoom_out)

        self.act_toggle_left_panel = QAction(self._t("显示/隐藏缩略图", "Toggle Thumbnails"), self)
        self.act_toggle_left_panel.setShortcut(QKeySequence("Ctrl+Alt+L"))
        self.act_toggle_left_panel.triggered.connect(self.toggle_left_panel)
        self.addAction(self.act_toggle_left_panel)

        self.act_toggle_right_panel = QAction(self._t("显示/隐藏右侧面板", "Toggle Right Panel"), self)
        self.act_toggle_right_panel.setShortcut(QKeySequence("Ctrl+Alt+R"))
        self.act_toggle_right_panel.triggered.connect(self.toggle_right_panel)
        self.addAction(self.act_toggle_right_panel)

        self.act_show_shortcuts = QAction(self._t("快捷键", "Shortcuts"), self)
        self.act_show_shortcuts.setShortcut(QKeySequence("F1"))
        self.act_show_shortcuts.triggered.connect(self.show_shortcuts)
        self.addAction(self.act_show_shortcuts)

        # === 1. TOP RIBBON ===
        self.tabs = QTabWidget()
        self.tabs.setFixedHeight(135)
        self.tabs.setStyleSheet(f"background: {PPT_RIBBON_BG};")

        # Top-right corner: GitHub open-source link (always visible regardless of tab).
        try:
            url = "https://github.com/Tansuo2021/OCRPDF-TO-PPT"
            corner = QWidget()
            cl = QHBoxLayout(corner)
            cl.setContentsMargins(0, 0, 8, 0)
            cl.setSpacing(6)

            btn_gh = QToolButton()
            btn_gh.setAutoRaise(True)
            try:
                btn_gh.setIcon(qta.icon("fa5b.github", color=PPT_THEME_RED))
            except Exception:
                btn_gh.setIcon(qta.icon("fa5s.code-branch", color=PPT_THEME_RED))
            btn_gh.setIconSize(QSize(18, 18))
            btn_gh.setToolTip(url)
            btn_gh.clicked.connect(self.open_github_repo)
            cl.addWidget(btn_gh)

            # Show a short label; keep full URL in tooltip (and in the actual link target).
            lb_gh = QLabel(f'<a href="{url}">{self._t("开源地址", "GitHub")}</a>')
            lb_gh.setOpenExternalLinks(True)
            lb_gh.setToolTip(url)
            lb_gh.setStyleSheet("font-size: 12px; color: #0b57d0;")
            cl.addWidget(lb_gh)

            self.tabs.setCornerWidget(corner, Qt.TopRightCorner)
        except Exception:
            pass
        
        tab_home = QWidget()
        lay_home = QHBoxLayout(tab_home)
        lay_home.setContentsMargins(4, 0, 4, 0)
        lay_home.setSpacing(2) # 组与分割线之间的间距
        
        # --- Group 1: 剪贴板 ---
        grp_clip = RibbonGroup(self._t("剪贴板", "Clipboard"))
        btn_paste = RibbonLargeBtn(self._t("粘贴", "Paste"), "fa5s.paste")
        btn_paste.clicked.connect(self.paste_box)
        w_small = QWidget()
        l_small = QVBoxLayout(w_small)
        l_small.setContentsMargins(0, 4, 0, 0); l_small.setSpacing(0)
        btn_cut = RibbonSmallBtn(self._t("剪切", "Cut"), "fa5s.cut")
        btn_cut.clicked.connect(self.cut_selected_box)
        l_small.addWidget(btn_cut)

        btn_copy = RibbonSmallBtn(self._t("复制", "Copy"), "fa5s.copy")
        btn_copy.clicked.connect(self.copy_selected_box)
        l_small.addWidget(btn_copy)

        btn_brush = RibbonSmallBtn(self._t("格式刷", "Brush"), "fa5s.paint-brush")
        btn_brush.clicked.connect(self.activate_format_brush)
        l_small.addWidget(btn_brush)

        btn_paste_img = RibbonSmallBtn(self._t("粘贴图", "Paste Img"), "fa5s.image")
        btn_paste_img.clicked.connect(self.paste_clipboard_image)
        l_small.addWidget(btn_paste_img)

        l_small.addStretch()
        grp_clip.add_widget(btn_paste)
        grp_clip.add_widget(w_small)
        
        lay_home.addWidget(grp_clip)
        
        # >>>>> 插入分隔线 1 <<<<<
        lay_home.addWidget(RibbonSeparator())
        
        # --- Group 2: 幻灯片 ---
        grp_layer = RibbonGroup(self._t("幻灯片", "Slides"))
        btn_layer = RibbonLargeBtn(self._t("新建空白\n图层", "New Blank\nSlide"), "fa5s.plus-square")
        btn_layer.clicked.connect(self.new_blank_slide)

        w_slide_ops = QWidget()
        l_ops = QVBoxLayout(w_slide_ops)
        l_ops.setContentsMargins(0, 4, 0, 0)
        l_ops.setSpacing(0)
        btn_dup_slide = RibbonSmallBtn(self._t("复制页", "Clone"), "fa5s.clone")
        btn_dup_slide.clicked.connect(self.duplicate_slide)
        l_ops.addWidget(btn_dup_slide)
        btn_del_slide = RibbonSmallBtn(self._t("删除页", "Delete"), "fa5s.trash-alt")
        btn_del_slide.clicked.connect(self.delete_slide)
        l_ops.addWidget(btn_del_slide)
        btn_up = RibbonSmallBtn(self._t("上移", "Up"), "fa5s.arrow-up")
        btn_up.clicked.connect(self.move_slide_up)
        l_ops.addWidget(btn_up)
        btn_down = RibbonSmallBtn(self._t("下移", "Down"), "fa5s.arrow-down")
        btn_down.clicked.connect(self.move_slide_down)
        l_ops.addWidget(btn_down)
        l_ops.addStretch()

        grp_layer.add_widget(btn_layer)
        grp_layer.add_widget(w_slide_ops)
        lay_home.addWidget(grp_layer)

        # >>>>> 插入分隔线 2 <<<<<
        lay_home.addWidget(RibbonSeparator())

        # --- Group 3: 插入 ---
        grp_insert = RibbonGroup(self._t("插入", "Insert"))
        btn_import = RibbonLargeBtn(self._t("导入\n图片", "Import\nImage"), "fa5s.image")
        btn_import.clicked.connect(self.import_images)
        grp_insert.add_widget(btn_import)

        btn_import_pdf = RibbonLargeBtn(self._t("导入\nPDF", "Import\nPDF"), "fa5s.file-pdf")
        btn_import_pdf.clicked.connect(self.import_pdfs)
        grp_insert.add_widget(btn_import_pdf)
        
        btn_text = RibbonLargeBtn(self._t("文本框", "Text\nBox"), "fa5s.font")
        btn_text.clicked.connect(self.insert_text_box)
        grp_insert.add_widget(btn_text)
        lay_home.addWidget(grp_insert)

        # >>>>> 插入分隔线 3 <<<<<
        lay_home.addWidget(RibbonSeparator())
        
        # --- Group 4: 识别与导出 ---
        grp_ai = RibbonGroup(self._t("识别与导出", "OCR/Export"))
        self.btn_ocr_current = RibbonLargeBtn(self._t("OCR\n本页", "OCR\nCurrent"), "fa5s.eye", color=PPT_THEME_RED)
        self.btn_ocr_current.clicked.connect(self.run_ocr_current_slide)

        self.btn_ocr_all = RibbonLargeBtn(self._t("OCR\n全部", "OCR\nAll"), "fa5s.eye", color=PPT_THEME_RED)
        self.btn_ocr_all.clicked.connect(self.run_ocr_all_images)
        
        self.btn_export = RibbonLargeBtn(self._t("导出\nPPT", "Export\nPPT"), "fa5s.file-powerpoint", color=PPT_THEME_RED)
        self.btn_export.clicked.connect(self.export_ppt)
        
        grp_ai.add_widget(self.btn_ocr_current)
        grp_ai.add_widget(self.btn_ocr_all)
        grp_ai.add_widget(self.btn_export)
        lay_home.addWidget(grp_ai)

        lay_home.addWidget(RibbonSeparator())

        # --- Group: 选区（ROI：框选 OCR/去字 区域） ---
        grp_roi = RibbonGroup(self._t("选区", "ROI"))
        self.btn_roi_select = RibbonLargeBtn(self._t("框选\n选区", "Select\nROI"), "fa5s.vector-square", color=PPT_THEME_RED)
        self.btn_roi_select.setToolTip(self._t("拖拽框选 OCR/IOPaint 生效区域 (Ctrl+Alt+A)", "Drag to select ROI for OCR/IOPaint (Ctrl+Alt+A)"))
        self.btn_roi_select.setCheckable(True)
        # Use full selectors; mixing bare declarations + selectors is easy to break QSS parsing.
        self.btn_roi_select.setStyleSheet(
            "QToolButton { font-size: 11px; padding-top: 4px; }"
            "\nQToolButton:checked { background: #D24726; color: white; border-radius: 4px; }"
        )
        self.btn_roi_select.toggled.connect(self.set_roi_select_mode)

        btn_roi_clear = RibbonLargeBtn(self._t("清除\n选区", "Clear\nROI"), "fa5s.times", color=PPT_THEME_RED)
        btn_roi_clear.setToolTip(self._t("清除当前页选区 (Ctrl+Alt+Shift+A)", "Clear ROI (Ctrl+Alt+Shift+A)"))
        btn_roi_clear.clicked.connect(self.clear_roi_current)

        grp_roi.add_widget(self.btn_roi_select)
        grp_roi.add_widget(btn_roi_clear)
        lay_home.addWidget(grp_roi)

        lay_home.addWidget(RibbonSeparator())

        # --- Group: 去字（按模式生成纯背景底图） ---
        grp_inpaint = RibbonGroup(self._t("去字", "Clean"))
        btn_inpaint_cur = RibbonLargeBtn(self._t("智能\n本页", "Smart\nCurrent"), "fa5s.magic", color=PPT_THEME_RED)
        btn_inpaint_cur.clicked.connect(self.inpaint_current_slide)
        btn_inpaint_all = RibbonLargeBtn(self._t("智能\n全部", "Smart\nAll"), "fa5s.magic", color=PPT_THEME_RED)
        btn_inpaint_all.clicked.connect(self.inpaint_all_slides)

        mode_ops = QWidget()
        mode_ops_layout = QGridLayout(mode_ops)
        mode_ops_layout.setContentsMargins(0, 2, 0, 0)
        mode_ops_layout.setHorizontalSpacing(2)
        mode_ops_layout.setVerticalSpacing(2)

        btn_fill_cur = RibbonSmallBtn(self._t("单色本页", "Fill Cur"), "fa5s.paint-brush")
        btn_fill_cur.clicked.connect(self.fill_current_slide)
        mode_ops_layout.addWidget(btn_fill_cur, 0, 0, 1, 1)

        btn_fill_all = RibbonSmallBtn(self._t("单色全部", "Fill All"), "fa5s.paint-brush")
        btn_fill_all.clicked.connect(self.fill_all_slides)
        mode_ops_layout.addWidget(btn_fill_all, 1, 0, 1, 1)

        btn_remote_cur = RibbonSmallBtn(self._t("IOP本页", "IOP Cur"), "fa5s.eraser")
        btn_remote_cur.clicked.connect(self.remote_inpaint_current_slide)
        mode_ops_layout.addWidget(btn_remote_cur, 0, 1, 1, 1)

        btn_remote_all = RibbonSmallBtn(self._t("IOP全部", "IOP All"), "fa5s.eraser")
        btn_remote_all.clicked.connect(self.remote_inpaint_all_slides)
        mode_ops_layout.addWidget(btn_remote_all, 1, 1, 1, 1)

        preview_ops = QWidget()
        preview_ops_layout = QVBoxLayout(preview_ops)
        preview_ops_layout.setContentsMargins(0, 2, 0, 0)
        preview_ops_layout.setSpacing(2)

        self.btn_inpaint_preview = RibbonSmallBtn(self._t("预览切换", "Preview"), "fa5s.adjust")
        self.btn_inpaint_preview.setToolTip(self._t("勾选：显示去字底图；取消：显示原图（便于对比） (Ctrl+Alt+B)", "Checked: show cleaned background; unchecked: original (Ctrl+Alt+B)"))
        self.btn_inpaint_preview.setCheckable(True)
        self.btn_inpaint_preview.setStyleSheet(
            "QToolButton { font-size: 11px; text-align: left; padding-left: 5px; }"
            "\nQToolButton:checked { background: #D24726; color: white; border-radius: 4px; }"
        )
        self.btn_inpaint_preview.toggled.connect(self.set_inpaint_preview)
        preview_ops_layout.addWidget(self.btn_inpaint_preview)

        btn_inpaint_restore = RibbonSmallBtn(self._t("恢复原图", "Restore"), "fa5s.history")
        btn_inpaint_restore.setToolTip(self._t("清除本页去字底图（恢复原图） (Ctrl+Alt+Shift+B)", "Clear cleaned background for this slide (Ctrl+Alt+Shift+B)"))
        btn_inpaint_restore.clicked.connect(self.clear_inpaint_variant_current)
        preview_ops_layout.addWidget(btn_inpaint_restore)
        preview_ops_layout.addStretch()

        grp_inpaint.add_widget(btn_inpaint_cur)
        grp_inpaint.add_widget(btn_inpaint_all)
        grp_inpaint.add_widget(mode_ops)
        grp_inpaint.add_widget(preview_ops)
        lay_home.addWidget(grp_inpaint)

        # >>>>> 插入分隔线 4 (末尾) <<<<<
        lay_home.addWidget(RibbonSeparator())
        
        lay_home.addStretch() # 向左对齐

        self.tabs.addTab(tab_home, self._t("开始", "Home"))

        # --- 视图标签页 ---
        tab_view = QWidget()
        lay_view = QHBoxLayout(tab_view)
        lay_view.setContentsMargins(4, 0, 4, 0)
        lay_view.setSpacing(2)

        # --- Group: PPT导出设置 ---
        grp_ppt_settings = RibbonGroup(self._t("PPT导出设置", "PPT Export"))

        # 创建垂直布局容器（紧凑布局，避免 Tab 高度 135 被撑爆）
        settings_container = QWidget()
        settings_container.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        settings_container.setFixedWidth(230)
        settings_layout = QVBoxLayout(settings_container)
        settings_layout.setContentsMargins(4, 2, 4, 2)
        # Give the View tab a bit more breathing room vertically.
        settings_layout.setSpacing(6)

        # 第一块：启用开关 + 颜色/吸管（拆成两行，避免小窗口挤在一起）
        top_row = QWidget()
        top_v = QVBoxLayout(top_row)
        top_v.setContentsMargins(0, 0, 0, 0)
        top_v.setSpacing(8)

        row1 = QWidget()
        row1_l = QHBoxLayout(row1)
        row1_l.setContentsMargins(0, 0, 0, 0)
        row1_l.setSpacing(6)

        self.chk_text_bg = QCheckBox(self._t("启用文本框背景色", "Text box background"))
        self.chk_text_bg.setStyleSheet("""
            QCheckBox { font-size: 13px; padding: 0px; }
            QCheckBox::indicator { width: 16px; height: 16px; }
        """)
        self.chk_text_bg.setChecked(self.use_text_bg)
        self.chk_text_bg.stateChanged.connect(self.toggle_text_bg)
        row1_l.addWidget(self.chk_text_bg)
        row1_l.addStretch()
        top_v.addWidget(row1)

        row2 = QWidget()
        row2_l = QHBoxLayout(row2)
        row2_l.setContentsMargins(0, 0, 0, 0)
        row2_l.setSpacing(6)
        
        # Color swatch + actions on the second line, left-aligned.
        self.color_preview = QLabel()
        self.color_preview.setFixedSize(22, 18)
        row2_l.addWidget(self.color_preview)
        self.update_color_preview()

        btn_color_picker = QPushButton(self._t("颜色", "Color"))
        btn_color_picker.setIcon(qta.icon("fa5s.palette", color="#666"))
        btn_color_picker.setFixedHeight(20)
        btn_color_picker.setStyleSheet("""
            QPushButton { padding: 1px 6px; background: white; border: 1px solid #CCC; border-radius: 3px; font-size: 11px; }
            QPushButton:hover { background: #F0F0F0; border-color: #999; }
        """)
        btn_color_picker.clicked.connect(self.pick_color)
        row2_l.addWidget(btn_color_picker)

        self.btn_eyedropper = QPushButton(self._t("吸管", "Pick"))
        self.btn_eyedropper.setIcon(qta.icon("fa5s.eye-dropper", color="#666"))
        self.btn_eyedropper.setFixedHeight(20)
        self.btn_eyedropper.setStyleSheet("""
            QPushButton { padding: 1px 6px; background: white; border: 1px solid #CCC; border-radius: 3px; font-size: 11px; }
            QPushButton:hover { background: #F0F0F0; border-color: #999; }
            QPushButton:checked { background: #D24726; color: white; border-color: #D24726; }
        """)
        self.btn_eyedropper.setCheckable(True)
        self.btn_eyedropper.clicked.connect(self.toggle_eyedropper)
        row2_l.addWidget(self.btn_eyedropper)
        row2_l.addStretch()
        top_v.addWidget(row2)

        settings_layout.addWidget(top_row)

        # 第二行：背景透明度（UI 使用“透明度”，内部仍使用 alpha；透明度=255-alpha）
        alpha_row = QWidget()
        alpha_layout = QHBoxLayout(alpha_row)
        alpha_layout.setContentsMargins(0, 0, 0, 0)
        alpha_layout.setSpacing(8)
        alpha_lbl = QLabel(self._t("背景透明度:", "BG transparency:"))
        alpha_lbl.setStyleSheet("font-size: 12px;")
        alpha_layout.addWidget(alpha_lbl)
        self.slider_global_alpha = QSlider(Qt.Horizontal)
        self.slider_global_alpha.setRange(ALPHA_MIN, ALPHA_MAX)
        self.slider_global_alpha.setToolTip(self._t("0=不透明，255=全透明", "0 = opaque, 255 = transparent"))
        try:
            self.slider_global_alpha.setValue(255 - int(getattr(self, "text_bg_alpha", 120)))
        except Exception:
            self.slider_global_alpha.setValue(135)
        self.slider_global_alpha.valueChanged.connect(self.on_global_bg_alpha_changed)
        self.slider_global_alpha.sliderPressed.connect(self.push_undo)
        self.slider_global_alpha.sliderReleased.connect(self._schedule_scene_rebuild)  # 释放时重建场景
        self.slider_global_alpha.setFixedWidth(130)
        alpha_layout.addWidget(self.slider_global_alpha)
        alpha_layout.addStretch()
        settings_layout.addWidget(alpha_row)

        grp_ppt_settings.add_widget(settings_container)
        grp_ppt_settings.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        grp_ppt_settings.setFixedWidth(260)

        lay_view.addWidget(grp_ppt_settings)
        lay_view.addWidget(RibbonSeparator())

        # --- Group: 预览 ---
        grp_preview = RibbonGroup(self._t("预览", "Preview"))
        grp_preview.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Preferred)
        grp_preview.setFixedWidth(90)
        btn_preview_ppt = RibbonLargeBtn(self._t("预览\nPPT", "Preview\nPPT"), "fa5s.eye", color=PPT_THEME_RED)
        btn_preview_ppt.clicked.connect(self.preview_ppt)
        grp_preview.add_widget(btn_preview_ppt)
        lay_view.addWidget(grp_preview)
        lay_view.addWidget(RibbonSeparator())

        lay_view.addStretch()

        self.tabs.addTab(tab_view, self._t("视图", "View"))

        # --- 设置标签页（替代顶部菜单：设置/编辑放到这里，风格与“开始/视图”一致） ---
        tab_settings = QWidget()
        lay_settings = QHBoxLayout(tab_settings)
        lay_settings.setContentsMargins(4, 0, 4, 0)
        lay_settings.setSpacing(2)

        grp_settings = RibbonGroup(self._t("设置", "Settings"))
        btn_ocr_settings = RibbonLargeBtn(self._t("OCR\n设置", "OCR\nSettings"), "fa5s.cog", color=PPT_THEME_RED)
        btn_ocr_settings.clicked.connect(self.open_ocr_settings)
        btn_reload_ocr = RibbonLargeBtn(self._t("重新加载\nOCR", "Reload\nOCR"), "fa5s.sync", color=PPT_THEME_RED)
        btn_reload_ocr.clicked.connect(self.force_reload_ocr_engine)
        btn_inpaint_settings = RibbonLargeBtn(self._t("去字\n设置", "Clean\nSettings"), "fa5s.eraser", color=PPT_THEME_RED)
        btn_inpaint_settings.clicked.connect(self.open_inpaint_settings)
        grp_settings.add_widget(btn_ocr_settings)
        grp_settings.add_widget(btn_reload_ocr)
        grp_settings.add_widget(btn_inpaint_settings)
        lay_settings.addWidget(grp_settings)
        lay_settings.addWidget(RibbonSeparator())

        # --- Language selector (saved to settings.json; restart to apply) ---
        grp_lang = RibbonGroup(self._t("语言", "Language"))

        # 使用大按钮风格，与其他设置保持一致
        self.btn_lang_auto = RibbonLargeBtn(self._t("跟随\n系统", "Auto\n(System)"), "fa5s.globe", color="#666")
        self.btn_lang_zh = RibbonLargeBtn(self._t("中文", "中文"), "fa5s.language", color="#666")
        self.btn_lang_en = RibbonLargeBtn("English", "fa5s.language", color="#666")

        # 设置当前语言
        pref_norm = str(getattr(self, "ui_lang_setting", "auto") or "auto").strip().lower().replace("_", "-")
        if pref_norm in ("zh", "zh-cn", "cn"):
            self._current_ui_lang = "zh"
        elif pref_norm in ("en", "en-us"):
            self._current_ui_lang = "en"
        else:
            self._current_ui_lang = "auto"

        # 连接点击事件
        self.btn_lang_auto.clicked.connect(lambda: self.on_ui_lang_btn_clicked("auto"))
        self.btn_lang_zh.clicked.connect(lambda: self.on_ui_lang_btn_clicked("zh"))
        self.btn_lang_en.clicked.connect(lambda: self.on_ui_lang_btn_clicked("en"))

        # 初始化按钮状态
        self._update_lang_button_styles()

        grp_lang.add_widget(self.btn_lang_auto)
        grp_lang.add_widget(self.btn_lang_zh)
        grp_lang.add_widget(self.btn_lang_en)
        lay_settings.addWidget(grp_lang)
        lay_settings.addWidget(RibbonSeparator())

        grp_edit = RibbonGroup(self._t("编辑", "Edit"))
        btn_undo = RibbonLargeBtn(self._t("撤销", "Undo"), "fa5s.undo")
        btn_undo.clicked.connect(self.undo)
        btn_redo = RibbonLargeBtn(self._t("重做", "Redo"), "fa5s.redo")
        btn_redo.clicked.connect(self.redo)
        grp_edit.add_widget(btn_undo)
        grp_edit.add_widget(btn_redo)

        w_edit_small = QWidget()
        l_edit_small = QVBoxLayout(w_edit_small)
        l_edit_small.setContentsMargins(0, 4, 0, 0)
        l_edit_small.setSpacing(0)
        b_cut = RibbonSmallBtn(self._t("剪切", "Cut"), "fa5s.cut")
        b_cut.clicked.connect(self.cut_selected_box)
        l_edit_small.addWidget(b_cut)
        b_copy = RibbonSmallBtn(self._t("复制", "Copy"), "fa5s.copy")
        b_copy.clicked.connect(self.copy_selected_box)
        l_edit_small.addWidget(b_copy)
        b_paste = RibbonSmallBtn(self._t("粘贴", "Paste"), "fa5s.paste")
        b_paste.clicked.connect(self.paste_box)
        l_edit_small.addWidget(b_paste)
        l_edit_small.addStretch()
        grp_edit.add_widget(w_edit_small)

        lay_settings.addWidget(grp_edit)
        lay_settings.addWidget(RibbonSeparator())
        lay_settings.addStretch()

        self.tabs.addTab(tab_settings, self._t("设置", "Settings"))
        
        # === 2. CENTER AREA ===
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0,0,0,0); main_layout.setSpacing(0)
        
        main_layout.addWidget(self.tabs)
        
        splitter = QSplitter(Qt.Horizontal)
        self.splitter = splitter
        splitter.setHandleWidth(1)
        splitter.setStyleSheet("QSplitter::handle { background: #CCC; }")
        splitter.splitterMoved.connect(lambda *_: self._fix_splitter_sizes())
        splitter.setChildrenCollapsible(False)
        
        self.list_thumb = QListWidget()
        # 固定缩略图栏宽度：用 min/max 避免 QSplitter 拉伸后产生“空白条”
        self.list_thumb.setMinimumWidth(230)
        self.list_thumb.setMaximumWidth(230)
        self.list_thumb.setStyleSheet("background: #F3F3F3; border: none; border-right: 1px solid #DDD;")
        self.list_thumb.currentRowChanged.connect(self.switch_slide)
        splitter.addWidget(self.list_thumb)
        
        self.scene = QGraphicsScene()
        self.view = CustomGraphicsView(self.scene, self)
        self.view.setStyleSheet("background: #E6E6E6; border: none;")
        self.view.setAlignment(Qt.AlignCenter)
        # 避免频繁修改透明度/画刷后出现“底图不刷新/消失”的重绘伪影
        try:
            self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
            self.view.setCacheMode(QGraphicsView.CacheNone)
        except Exception:
            pass
        splitter.addWidget(self.view)
        
        # 右侧属性面板：用 QScrollArea 防止窗口高度不足时被挡住；并避免全屏出现多余空白条
        self.right_panel = QWidget()
        self.right_panel.setObjectName("RightPanel")
        self.setup_right_panel()

        self.right_panel_scroll = QScrollArea()
        self.right_panel_scroll.setObjectName("RightPanelScroll")
        self.right_panel_scroll.setWidgetResizable(True)
        # 某些字体/系统缩放下，右侧控件最小宽度可能略超出，允许水平滚动避免“被挡住”。
        self.right_panel_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.right_panel_scroll.setFrameShape(QFrame.NoFrame)
        # 固定右侧栏宽度（避免全屏后右侧出现“空白条”）
        self.right_panel_scroll.setMinimumWidth(RIGHT_PANEL_W)
        self.right_panel_scroll.setMaximumWidth(RIGHT_PANEL_W)
        self.right_panel_scroll.setWidget(self.right_panel)
        splitter.addWidget(self.right_panel_scroll)

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        # 初始/运行时都强制把剩余空间给中间视图，避免右侧被分到超过 260 导致“多出白框”
        QTimer.singleShot(0, self._fix_splitter_sizes)
        main_layout.addWidget(splitter)
        
        # === 3. BOTTOM BAR (Fixed) ===
        status_bar = QFrame()
        status_bar.setObjectName("StatusBar") # 关联到红色CSS
        sb_layout = QHBoxLayout(status_bar)
        sb_layout.setContentsMargins(10, 0, 15, 0)
        sb_layout.setSpacing(8)
        
        self.lbl_page = QPushButton(self._t("幻灯片 0 / 0", "Slide 0 / 0")) # 样式在 Global CSS 中
        sb_layout.addWidget(self.lbl_page)
        
        btn_lang = QPushButton(self._t("中文", "English"))
        sb_layout.addWidget(btn_lang)
        
        sb_layout.addStretch()
        
        # Bottom quick buttons: bind to the same actions/shortcuts.
        btn_help = QPushButton()
        btn_help.setIcon(qta.icon("fa5s.keyboard", color="white"))
        btn_help.setToolTip(self._t("快捷键 (F1)", "Shortcuts (F1)"))
        btn_help.clicked.connect(self.show_shortcuts)
        sb_layout.addWidget(btn_help)

        btn_toggle_left = QPushButton()
        btn_toggle_left.setIcon(qta.icon("fa5s.th-large", color="white"))
        btn_toggle_left.setToolTip(self._t("显示/隐藏缩略图 (Ctrl+Alt+L)", "Toggle thumbnails (Ctrl+Alt+L)"))
        btn_toggle_left.clicked.connect(self.toggle_left_panel)
        sb_layout.addWidget(btn_toggle_left)

        btn_preview = QPushButton()
        # “电脑/显示器”图标：预览PPT
        btn_preview.setIcon(qta.icon("fa5s.tv", color="white"))
        btn_preview.setToolTip(self._t("预览PPT (F5)", "Preview PPT (F5)"))
        btn_preview.clicked.connect(self.preview_ppt)
        sb_layout.addWidget(btn_preview)
        
        btn_fit = QPushButton()
        btn_fit.setIcon(qta.icon("fa5s.expand-arrows-alt", color="white"))
        btn_fit.setToolTip(self._t("适应窗口 (Ctrl+0)", "Fit to window (Ctrl+0)"))
        btn_fit.clicked.connect(self.fit_view_to_window)
        sb_layout.addWidget(btn_fit)
        
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(ZOOM_MIN, ZOOM_MAX)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setFixedWidth(130)
        self.zoom_slider.setCursor(Qt.PointingHandCursor)
        self.zoom_slider.valueChanged.connect(self.zoom_view)
        sb_layout.addWidget(self.zoom_slider)
        
        self.lbl_zoom_val = QPushButton("100%")
        self.lbl_zoom_val.setFixedWidth(50)
        sb_layout.addWidget(self.lbl_zoom_val)
        
        main_layout.addWidget(status_bar)

    def _fix_splitter_sizes(self):
        """锁定左右侧栏宽度，把剩余空间给中间视图，避免出现多余空白区域。"""
        sp = getattr(self, "splitter", None)
        if sp is None:
            return
        if getattr(self, "_fixing_splitter", False):
            return
        try:
            self._fixing_splitter = True
            sizes = sp.sizes()
            if len(sizes) != 3:
                return
            total = sum(int(x) for x in sizes)
            left = 230 if getattr(self, "_show_left_panel", True) else 0
            right = RIGHT_PANEL_W if getattr(self, "_show_right_panel", True) else 0
            # 窗口太窄时不要强行塞固定宽度，避免右侧/中间被挤压导致显示异常
            if total < (left + right + 200):
                return
            mid = max(200, total - left - right)
            target = [left, mid, right]
            if sizes != target:
                sp.setSizes(target)
        finally:
            self._fixing_splitter = False

    def resizeEvent(self, event):
        super().resizeEvent(event)
        # 窗口尺寸变化时也修正一次，防止全屏/还原后出现空白条
        try:
            self._fix_splitter_sizes()
        except Exception:
            pass

    def setup_right_panel(self):
        l = QVBoxLayout(self.right_panel)
        # 右侧面板宽度固定，减小内边距避免控件被挤到右侧边缘
        l.setContentsMargins(12, 18, 16, 18)
        hdr = QLabel(self._t("设置对象格式", "Format Selection"))
        hdr.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 12px;")
        l.addWidget(hdr)
        l.addWidget(QLabel(self._t("文本内容:", "Text:")))
        self.txt_edit = QTextEdit()
        self.txt_edit.setFixedHeight(100)
        self.txt_edit.setStyleSheet("border: 1px solid #CCC; background: #FAFAFA;")
        self.txt_edit.textChanged.connect(self.sync_text_change)
        l.addWidget(self.txt_edit)
        l.addSpacing(10)
        l.addWidget(QLabel(self._t("字体大小:", "Font size:")))
        self.slider_font = QSlider(Qt.Horizontal)
        self.slider_font.setStyleSheet("""
            QSlider::groove:horizontal { height: 3px; background: #DDD; }
            QSlider::handle:horizontal { background: #D24726; width: 12px; height: 12px; margin: -5px 0; border-radius: 6px; }
        """)
        # PPT title text can easily exceed 72pt (e.g. 1080p/4K slides). Keep the UI range large enough
        # so selecting a big box doesn't clamp the value to the slider max.
        self.slider_font.setRange(FONT_SIZE_MIN, FONT_SIZE_MAX)
        self.slider_font.setValue(12)
        self.slider_font.valueChanged.connect(self.on_font_size_changed)
        self.slider_font.sliderPressed.connect(self.push_undo)
        l.addWidget(self.slider_font)
        l.addSpacing(10)

        # 文字颜色 + 加粗 + 对齐（改为两行布局，避免窄面板时被裁切）
        l.addWidget(QLabel(self._t("文字样式:", "Text style:")))
        text_style_row = QWidget()
        ts = QGridLayout(text_style_row)
        ts.setContentsMargins(0, 0, 0, 0)
        ts.setHorizontalSpacing(6)
        ts.setVerticalSpacing(6)

        self.text_color_preview = QLabel()
        self.text_color_preview.setFixedSize(40, 30)
        self.text_color_preview.setStyleSheet("background: black; border: 1px solid #ccc;")
        ts.addWidget(self.text_color_preview, 0, 0, 1, 1)
        self.btn_text_color = QPushButton(self._t("文字颜色", "Text color"))
        self.btn_text_color.setIcon(qta.icon("fa5s.font", color="#666"))
        self.btn_text_color.clicked.connect(self.choose_text_color)
        self.btn_text_color.setEnabled(False)
        self.btn_text_color.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        ts.addWidget(self.btn_text_color, 0, 1, 1, 2)

        self.chk_bold = QCheckBox(self._t("加粗", "Bold"))
        self.chk_bold.stateChanged.connect(self.toggle_bold)
        self.chk_bold.setEnabled(False)
        ts.addWidget(self.chk_bold, 0, 3, 1, 1)

        self.btn_align_left = QPushButton(self._t("左", "L"))
        self.btn_align_left.clicked.connect(lambda: self.set_align("left"))
        self.btn_align_left.setEnabled(False)
        self.btn_align_left.setFixedWidth(34)
        ts.addWidget(self.btn_align_left, 1, 1, 1, 1)
        self.btn_align_center = QPushButton(self._t("中", "C"))
        self.btn_align_center.clicked.connect(lambda: self.set_align("center"))
        self.btn_align_center.setEnabled(False)
        self.btn_align_center.setFixedWidth(34)
        ts.addWidget(self.btn_align_center, 1, 2, 1, 1)
        self.btn_align_right = QPushButton(self._t("右", "R"))
        self.btn_align_right.clicked.connect(lambda: self.set_align("right"))
        self.btn_align_right.setEnabled(False)
        self.btn_align_right.setFixedWidth(34)
        ts.addWidget(self.btn_align_right, 1, 3, 1, 1)
        ts.setColumnStretch(2, 1)
        l.addWidget(text_style_row)
        l.addSpacing(10)

        # 单独背景色设置
        l.addWidget(QLabel(self._t("文本框背景色:", "Text box background:")))
        bg_container = QWidget()
        bg_layout = QGridLayout(bg_container)
        bg_layout.setContentsMargins(0, 0, 0, 0)
        bg_layout.setHorizontalSpacing(6)
        bg_layout.setVerticalSpacing(6)

        self.chk_custom_bg = QCheckBox(self._t("自定义", "Custom"))
        self.chk_custom_bg.stateChanged.connect(self.toggle_custom_bg)
        bg_layout.addWidget(self.chk_custom_bg, 0, 0, 1, 1)

        self.custom_color_preview = QLabel()
        self.custom_color_preview.setFixedSize(40, 30)
        self.custom_color_preview.setStyleSheet("background-color: white; border: 1px solid #ccc;")
        bg_layout.addWidget(self.custom_color_preview, 0, 1, 1, 1)

        self.btn_choose_custom_color = QPushButton(self._t("选择", "Choose"))
        self.btn_choose_custom_color.setIcon(qta.icon("fa5s.palette", color="#666"))
        self.btn_choose_custom_color.clicked.connect(self.choose_custom_color)
        self.btn_choose_custom_color.setEnabled(False)
        self.btn_choose_custom_color.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bg_layout.addWidget(self.btn_choose_custom_color, 1, 0, 1, 1)

        self.btn_pick_custom_color = QPushButton(self._t("吸管", "Pick"))
        self.btn_pick_custom_color.setIcon(qta.icon("fa5s.eye-dropper", color="#666"))
        self.btn_pick_custom_color.setCheckable(True)
        self.btn_pick_custom_color.setStyleSheet("""
            QPushButton { padding: 4px 10px; background: white; border: 1px solid #CCC; border-radius: 3px; }
            QPushButton:hover { background: #F0F0F0; border-color: #999; }
            QPushButton:checked { background: #D24726; color: white; border-color: #D24726; }
        """)
        self.btn_pick_custom_color.toggled.connect(self.toggle_selected_eyedropper)
        self.btn_pick_custom_color.setEnabled(False)
        self.btn_pick_custom_color.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        bg_layout.addWidget(self.btn_pick_custom_color, 1, 1, 1, 1)
        bg_layout.setColumnStretch(0, 1)
        bg_layout.setColumnStretch(1, 1)
        l.addWidget(bg_container)

        l.addWidget(QLabel(self._t("背景透明度:", "BG transparency:")))
        self.slider_bg_alpha = QSlider(Qt.Horizontal)
        self.slider_bg_alpha.setRange(ALPHA_MIN, ALPHA_MAX)
        self.slider_bg_alpha.setToolTip(self._t("0=不透明，255=全透明", "0 = opaque, 255 = transparent"))
        # UI 透明度：255-alpha
        self.slider_bg_alpha.setValue(135)
        self.slider_bg_alpha.valueChanged.connect(self.on_bg_alpha_changed)
        self.slider_bg_alpha.sliderPressed.connect(self.push_undo)
        self.slider_bg_alpha.sliderReleased.connect(self._schedule_scene_rebuild)  # 释放时重建场景
        self.slider_bg_alpha.setEnabled(False)
        l.addWidget(self.slider_bg_alpha)
        l.addSpacing(10)

        l.addWidget(QLabel(self._t("去字设置:", "Clean mode:")))
        clean_container = QWidget()
        clean_layout = QGridLayout(clean_container)
        clean_layout.setContentsMargins(0, 0, 0, 0)
        clean_layout.setHorizontalSpacing(6)
        clean_layout.setVerticalSpacing(6)

        self.chk_clean_enabled = QCheckBox(self._t("参与去字", "Include in clean"))
        self.chk_clean_enabled.stateChanged.connect(self.toggle_selected_clean_enabled)
        self.chk_clean_enabled.setEnabled(False)
        clean_layout.addWidget(self.chk_clean_enabled, 0, 0, 1, 1)

        self.cmb_clean_mode = QComboBox()
        self.cmb_clean_mode.addItem(self._t("智能判断", "Smart"), InpaintThread.MODE_AUTO)
        self.cmb_clean_mode.addItem(self._t("单色覆盖", "Solid fill"), InpaintThread.MODE_FILL)
        self.cmb_clean_mode.addItem(self._t("IOPaint", "IOPaint"), InpaintThread.MODE_REMOTE)
        self.cmb_clean_mode.currentIndexChanged.connect(self.change_selected_clean_mode)
        self.cmb_clean_mode.setEnabled(False)
        clean_layout.addWidget(self.cmb_clean_mode, 0, 1, 1, 1)
        clean_layout.setColumnStretch(1, 1)
        l.addWidget(clean_container)
        l.addSpacing(10)

        # 右侧面板不再提供“边框”设置（仅保留编辑时的虚线选中框）

        self.btn_apply_style_page = QPushButton(self._t("应用样式到本页全部文本框", "Apply style to all boxes (slide)"))
        self.btn_apply_style_page.clicked.connect(self.apply_style_to_current_slide)
        self.btn_apply_style_page.setEnabled(False)
        l.addWidget(self.btn_apply_style_page)

        l.addStretch()
        btn_del = QPushButton(self._t("删除选中框", "Delete selected box"))
        btn_del.setIcon(qta.icon("fa5s.trash-alt", color="#D24726"))
        btn_del.setStyleSheet("QPushButton { border: 1px solid #D24726; color: #D24726; padding: 6px; background: white; border-radius: 4px; } QPushButton:hover { background: #FFF3F0; }")
        btn_del.clicked.connect(self.delete_box)
        l.addWidget(btn_del)

    # ==================== Logic ====================
    def scale_images_to_1080p(self, images=None):
        """将图片缩放到1080p以优化OCR识别（可传入子集，仅处理指定图片）"""
        images = list(images) if images else list(self.images or [])
        if not images:
            return

        # Drop previous scaling outputs to avoid temp-dir accumulation across multiple OCR runs.
        try:
            if getattr(self, "temp_dir", None) and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass

        # Keep scaled images under run_cache_dir by default (instead of C:\\Temp), and still clean on exit.
        try:
            import time as _time_mod
            base_dir = getattr(self, "run_cache_dir", None) or tempfile.gettempdir()
            ts = int(_time_mod.time() * 1000)
            self.temp_dir = os.path.join(str(base_dir), f"ocr_scaled_{ts}")
            os.makedirs(self.temp_dir, exist_ok=True)
        except Exception:
            self.temp_dir = tempfile.mkdtemp(prefix="ocr_scaled_")
        self.scaled_images = {}

        for original_path in images:
            try:
                img = _imread_any(original_path)
                if img is None:
                    continue

                h, w = img.shape[:2]
                target_h = TARGET_IMAGE_HEIGHT

                # 防止除零：高度为0的图片跳过
                if h <= 0 or w <= 0:
                    self.scaled_images[original_path] = original_path
                    continue

                # 如果图片高度已经接近1080p，不需要缩放
                if abs(h - target_h) < 100:
                    self.scaled_images[original_path] = original_path
                    continue

                # 计算缩放比例
                scale = target_h / max(1, h)
                new_w = int(w * scale)
                new_h = target_h

                # 缩放图片
                scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

                # 保存到临时目录
                scaled_path = build_asset_path(self.temp_dir, "scaled", original_path, ext=".png")
                if not _imwrite_any(scaled_path, scaled_img):
                    raise RuntimeError("无法写入缩放后的临时图片")

                self.scaled_images[original_path] = scaled_path

            except Exception as e:
                logger.warning(f"缩放图片失败 {original_path}: {e}")
                self.scaled_images[original_path] = original_path

    def import_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            None,
            self._t("导入图片", "Import Images"),
            "",
            "Images (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if not paths: return
        for p in paths:
            self._add_image_item(p)
        self.update_status()

    def import_pdfs(self):
        """导入 PDF：把每一页渲染成图片后加入左侧缩略图列表，供 OCR 识别/导出"""
        paths, _ = QFileDialog.getOpenFileNames(None, self._t("导入PDF", "Import PDF"), "", "PDF (*.pdf)")
        if not paths:
            return

        try:
            import fitz  # PyMuPDF
        except Exception:
            QMessageBox.critical(
                self,
                self._t("缺少依赖", "Missing dependency"),
                self._t("导入PDF需要 PyMuPDF。\n请先安装：pip install pymupdf", "PDF import requires PyMuPDF.\nInstall first: pip install pymupdf"),
            )
            return

        # 预先统计总页数用于进度条
        total_pages = 0
        docs = []
        for p in paths:
            try:
                doc = fitz.open(p)
                docs.append((p, doc))
                total_pages += int(getattr(doc, "page_count", 0) or 0)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    self._t("提示", "Info"),
                    self._t(f"无法打开PDF：{p}\n{e}", f"Failed to open PDF: {p}\n{e}"),
                )

        if total_pages <= 0:
            for _, d in docs:
                try:
                    d.close()
                except Exception:
                    pass
            return

        progress = QProgressDialog(self._t("正在导入PDF...", "Importing PDF..."), self._t("取消", "Cancel"), 0, total_pages, self)
        progress.setWindowTitle(self._t("导入PDF", "Import PDF"))
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        imported = 0
        canceled = False
        try:
            import time
            stamp = int(time.time())
            # Render scale: 2x (approx 144 DPI on a 72 DPI base), good balance for OCR.
            zoom = 2.0
            for pdf_path, doc in docs:
                for page_index in range(int(doc.page_count or 0)):
                    if progress.wasCanceled():
                        canceled = True
                        break
                    try:
                        page = doc.load_page(page_index)
                        pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
                        out_path = build_asset_path(
                            self.slide_assets_dir,
                            "pdf",
                            pdf_path,
                            suffix=f"{stamp}_p{page_index+1:04d}",
                            ext=".png",
                        )
                        pix.save(out_path)
                        self._add_image_item(out_path)
                    except Exception as e:
                        QMessageBox.warning(
                            self,
                            self._t("提示", "Info"),
                            self._t(f"PDF渲染失败：{pdf_path}\n第 {page_index+1} 页\n{e}", f"PDF render failed: {pdf_path}\nPage {page_index+1}\n{e}"),
                        )
                    imported += 1
                    progress.setValue(imported)
                if canceled:
                    break
        finally:
            try:
                progress.close()
            except Exception:
                pass
            for _, d in docs:
                try:
                    d.close()
                except Exception:
                    pass

        # Keep behavior consistent with "导入图片" (no auto-select), just refresh status.
        self.update_status()

    def _ensure_inpaint_ready(self) -> bool:
        if not bool(self.settings.get("inpaint_enabled", True)):
            QMessageBox.warning(
                self,
                self._t("提示", "Info"),
                self._t("去字功能已在设置中关闭", "Background cleaning is disabled in Settings."),
            )
            return False
        return True

    def _apply_inpaint_results(self, replacements):
        """Store inpainted variants (non-destructive) and refresh UI."""
        if not replacements:
            return

        for src, dst in replacements:
            if not src or not dst:
                continue
            try:
                if os.path.exists(dst):
                    self.inpaint_variants[str(src)] = str(dst)
            except Exception:
                pass

        # Auto switch to inpaint preview so user sees the result; can toggle back for compare.
        self.show_inpaint_preview = True
        self._sync_inpaint_preview_toggle()
        self._refresh_thumb_images()
        try:
            self._rebuild_scene_keep_view()
        except Exception:
            pass

    def _clean_mode_meta(self, run_mode):
        run_mode = InpaintThread._normalize_run_mode(run_mode)
        if run_mode == InpaintThread.RUN_FILL:
            return {
                "title": self._t("单色覆盖去字", "Solid Fill Clean"),
                "progress": self._t("正在单色覆盖去字...", "Applying solid fill clean..."),
            }
        if run_mode == InpaintThread.RUN_REMOTE:
            return {
                "title": self._t("IOPaint去字", "IOPaint Clean"),
                "progress": self._t("正在调用 IOPaint 去字...", "Cleaning with IOPaint..."),
            }
        return {
            "title": self._t("智能去字", "Smart Clean"),
            "progress": self._t("正在智能去字...", "Cleaning background..."),
        }

    def _collect_clean_targets(self, images_to_run):
        count = 0
        explicit_remote = False
        for img_path in images_to_run or []:
            for box in self.box_data.get(img_path, []) or []:
                if not isinstance(box, dict):
                    continue
                if InpaintThread._extract_rect(box) is None:
                    continue
                if not bool(box.get("clean_enabled", True)):
                    continue
                count += 1
                if InpaintThread._normalize_box_clean_mode(box.get("clean_mode")) == InpaintThread.MODE_REMOTE:
                    explicit_remote = True
        return count, explicit_remote

    def _start_clean_run(self, images_to_run, run_mode):
        if not self.images:
            QMessageBox.warning(self, self._t("提示", "Info"), self._t("请先导入图片", "Please import images first."))
            return
        if not self._ensure_inpaint_ready():
            return

        try:
            images_to_run = [p for p in images_to_run if p]
        except Exception:
            images_to_run = []
        if not images_to_run:
            QMessageBox.warning(self, self._t("提示", "Info"), self._t("没有可处理的图片", "No images to process."))
            return

        cleanable_count, explicit_remote = self._collect_clean_targets(images_to_run)
        if cleanable_count <= 0:
            QMessageBox.warning(
                self,
                self._t("提示", "Info"),
                self._t("没有启用的去字框，请先运行 OCR 或勾选“参与去字”。", "No enabled OCR boxes for cleaning. Run OCR first or enable 'Include in clean'."),
            )
            return

        run_mode = InpaintThread._normalize_run_mode(run_mode)
        strict_remote = bool(run_mode == InpaintThread.RUN_REMOTE or (run_mode == InpaintThread.RUN_SMART and explicit_remote))
        api_urls = parse_inpaint_api_urls(self.settings.get("inpaint_api_url"))
        if strict_remote and not api_urls:
            QMessageBox.warning(
                self,
                self._t("提示", "Info"),
                self._t("当前操作包含强制 IOPaint 去字，但还没有配置可用的 IOPaint API 地址。", "This action requires IOPaint, but no IOPaint API URL is configured."),
            )
            return

        self._run_inpaint(images_to_run, run_mode=run_mode)

    def inpaint_current_slide(self, *args):
        """智能去字：优先单色覆盖，复杂区域或显式指定时调用 IOPaint。"""
        if not self.current_img:
            QMessageBox.warning(self, self._t("提示", "Info"), self._t("请先导入图片", "Please import images first."))
            return
        self._start_clean_run([self.current_img], run_mode=InpaintThread.RUN_SMART)

    def inpaint_all_slides(self, *args):
        """智能去字：批量处理全部图片页。"""
        self._start_clean_run(list(self.images), run_mode=InpaintThread.RUN_SMART)

    def fill_current_slide(self, *args):
        if not self.current_img:
            QMessageBox.warning(self, self._t("提示", "Info"), self._t("请先导入图片", "Please import images first."))
            return
        self._start_clean_run([self.current_img], run_mode=InpaintThread.RUN_FILL)

    def fill_all_slides(self, *args):
        self._start_clean_run(list(self.images), run_mode=InpaintThread.RUN_FILL)

    def remote_inpaint_current_slide(self, *args):
        if not self.current_img:
            QMessageBox.warning(self, self._t("提示", "Info"), self._t("请先导入图片", "Please import images first."))
            return
        self._start_clean_run([self.current_img], run_mode=InpaintThread.RUN_REMOTE)

    def remote_inpaint_all_slides(self, *args):
        self._start_clean_run(list(self.images), run_mode=InpaintThread.RUN_REMOTE)

    def _run_inpaint(self, images_to_run, run_mode=InpaintThread.RUN_SMART):
        try:
            images_to_run = [p for p in images_to_run if p]
        except Exception:
            images_to_run = []
        if not images_to_run:
            return
        run_mode = InpaintThread._normalize_run_mode(run_mode)
        mode_meta = self._clean_mode_meta(run_mode)

        # Avoid concurrent inpaint runs (can confuse UI state / overwrite progress dialogs).
        try:
            th = getattr(self, "inpaint_thread", None)
            if th is not None and th.isRunning():
                QMessageBox.information(
                    self,
                    self._t("提示", "Info"),
                    self._t("去字任务正在运行，请先等待完成或取消。", "A clean task is already running. Please wait or cancel first."),
                )
                return
        except Exception:
            pass

        # Snapshot for undo
        self.push_undo()

        api_urls = [] if run_mode == InpaintThread.RUN_FILL else parse_inpaint_api_urls(self.settings.get("inpaint_api_url"))
        api_url_hint = "\n".join(api_urls)
        box_pad = int(self.settings.get("inpaint_box_padding", 6) or 6)
        crop_pad = int(self.settings.get("inpaint_crop_padding", 128) or 128)
        fill_pad_x = int(self.settings.get("inpaint_fill_pad_x", 10) or 0)
        fill_pad_y = int(self.settings.get("inpaint_fill_pad_y", 6) or 0)
        remote_pad_x = int(self.settings.get("inpaint_remote_pad_x", 8) or 0)
        remote_pad_y = int(self.settings.get("inpaint_remote_pad_y", 4) or 0)

        # Always start from the original slide image. Re-running on an already-cleaned preview
        # compounds artifacts and makes OCR masks drift from the true background.
        input_image_by_src = {}

        progress = QProgressDialog(mode_meta["progress"], self._t("取消", "Cancel"), 0, len(images_to_run), self)
        progress.setWindowTitle(mode_meta["title"])
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        self.inpaint_thread = InpaintThread(
            images=images_to_run,
            box_data=self.box_data,
            out_dir=self.slide_assets_dir,
            api_url=api_urls,
            box_padding=box_pad,
            crop_padding=crop_pad,
            input_image_by_src=input_image_by_src,
            roi_by_image=getattr(self, "roi_by_image", {}) or {},
            timeout_sec=120,
            run_mode=run_mode,
            fill_pad_x=fill_pad_x,
            fill_pad_y=fill_pad_y,
            remote_pad_x=remote_pad_x,
            remote_pad_y=remote_pad_y,
        )
        progress.canceled.connect(self.inpaint_thread.requestInterruption)
        # 显式使用 QueuedConnection 确保跨线程信号在主线程中处理
        self.inpaint_thread.progress.connect(lambda cur, total: progress.setValue(cur), Qt.QueuedConnection)

        # Collect replacements
        self._inpaint_replacements = []
        self.inpaint_thread.finished_one.connect(
            lambda src, dst: self._inpaint_replacements.append((src, dst)),
            Qt.QueuedConnection
        )

        def on_error(msg: str):
            try:
                progress.close()
            except Exception:
                pass
            if run_mode == InpaintThread.RUN_FILL:
                extra = self._t(
                    "请检查 OCR 遮罩框是否正确，以及原图是否可正常读取。",
                    "Please check the OCR masks and make sure the source image can be read correctly.",
                )
            else:
                extra = self._t(
                    "请确认 IOPaint 服务已启动，并且 API 地址正确。",
                    "Please make sure the IOPaint service is running and the API URL is correct.",
                )
                if api_url_hint:
                    extra = self._t(
                        f"{extra}\n{api_url_hint}",
                        f"{extra}\n{api_url_hint}",
                    )
            QMessageBox.critical(
                self,
                self._t("去字失败", "Clean failed"),
                f"{msg}\n\n{extra}",
            )

        def on_done(canceled: bool):
            try:
                progress.close()
            except Exception:
                pass
            reps = getattr(self, "_inpaint_replacements", []) or []
            if reps:
                self._apply_inpaint_results(reps)
            if canceled:
                QMessageBox.information(
                    self,
                    self._t("提示", "Info"),
                    self._t("已取消当前去字任务", "The current clean task was canceled."),
                )

        self.inpaint_thread.error.connect(on_error, Qt.QueuedConnection)
        self.inpaint_thread.all_done.connect(on_done, Qt.QueuedConnection)
        self.inpaint_thread.start()

    def run_ocr_simulation(self, images=None):
        """运行OCR识别（默认全部；传入 images 可只识别单页/子集）"""
        if not self.images:
            QMessageBox.warning(self, self._t("提示", "Info"), self._t("请先导入图片", "Please import images first."))
            return

        images_to_run = list(self.images) if images is None else [p for p in images if p]
        if not images_to_run:
            QMessageBox.warning(self, self._t("提示", "Info"), self._t("没有可识别的图片", "No images to OCR."))
            return

        if not self.ensure_ocr_engine():
            return

        # Avoid concurrent OCR runs (will recreate temp_dir and can race).
        try:
            th = getattr(self, "ocr_thread", None)
            if th is not None and th.isRunning():
                QMessageBox.information(
                    self,
                    self._t("提示", "Info"),
                    self._t("OCR 正在运行，请先等待完成或取消。", "OCR is running. Please wait or cancel first."),
                )
                return
        except Exception:
            pass

        # 先缩放图片（只处理本次识别子集）
        self.scale_images_to_1080p(images_to_run)

        # 创建进度对话框
        progress = QProgressDialog(self._t("正在识别...", "Running OCR..."), self._t("取消", "Cancel"), 0, len(images_to_run), self)
        progress.setWindowTitle(self._t("OCR识别", "OCR"))
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        # 创建OCR线程
        self.ocr_thread = OCRThread(
            self.ocr_engine,
            images_to_run,
            self.scaled_images,
            roi_by_image=getattr(self, "roi_by_image", {}) or {},
            roi_temp_dir=getattr(self, "temp_dir", None),
        )
        progress.canceled.connect(self.ocr_thread.requestInterruption)
        # 显式使用 QueuedConnection 确保跨线程信号在主线程中处理
        self.ocr_thread.finished.connect(
            lambda img, results, roi: self.on_ocr_result(img, results, roi),
            Qt.QueuedConnection
        )
        self.ocr_thread.progress.connect(lambda cur, total: progress.setValue(cur), Qt.QueuedConnection)
        self.ocr_thread.all_done.connect(progress.close, Qt.QueuedConnection)
        self.ocr_thread.error.connect(
            lambda msg: (progress.close(), QMessageBox.critical(self, self._t("OCR错误", "OCR Error"), msg)),
            Qt.QueuedConnection,
        )
        self.ocr_thread.start()

    def run_ocr_current_slide(self):
        """OCR：仅识别当前页"""
        if not self.current_img:
            QMessageBox.warning(self, self._t("提示", "Info"), self._t("请先选择一页图片", "Please select an image/slide first."))
            return
        self.run_ocr_simulation([self.current_img])

    def run_ocr_all_images(self):
        """OCR：识别全部导入图片"""
        self.run_ocr_simulation()

    def new_blank_slide(self, *args):
        """新建一个空白图层（作为普通图片页处理）"""
        try:
            import time
            self.push_undo()
            # 默认用 1920x1080；若已有当前页则复用当前尺寸
            w, h = 1920, 1080
            if self.current_img and os.path.exists(self.current_img):
                pix = QPixmap(self.current_img)
                if not pix.isNull():
                    w, h = max(1, pix.width()), max(1, pix.height())

            import numpy as np
            blank = np.ones((h, w, 3), dtype=np.uint8) * 255
            out_path = build_asset_path(
                self.slide_assets_dir,
                "blank",
                f"blank_{len(self.images)+1}_{w}x{h}_{int(time.time() * 1000)}",
                ext=".png",
            )
            if not _imwrite_any(out_path, blank):
                raise RuntimeError(self._t("无法创建空白页图片", "Failed to create blank slide image"))

            self._add_image_item(out_path)
            self.update_status()
            self.list_thumb.setCurrentRow(len(self.images) - 1)
        except Exception as e:
            QMessageBox.critical(self, self._t("错误", "Error"), self._t(f"新建空白图层失败: {e}", f"Failed to create blank slide: {e}"))

    def delete_slide(self, *args):
        """删除当前页（同时删除该页的文本框数据）"""
        idx = self.list_thumb.currentRow()
        if idx < 0 or idx >= len(self.images):
            return
        if QMessageBox.question(self, self._t("确认删除", "Confirm"), self._t("确定删除当前页吗？", "Delete current slide?")) != QMessageBox.Yes:
            return

        self.push_undo()
        path = self.images[idx]
        self.images.pop(idx)
        try:
            self.box_data.pop(path, None)
        except Exception:
            pass

        # 如果是临时生成页资源，尽量删除文件
        try:
            if path and os.path.exists(path) and os.path.commonpath([self.slide_assets_dir, path]) == self.slide_assets_dir:
                os.remove(path)
        except Exception:
            pass

        self._rebuild_thumb_list(select_index=min(idx, len(self.images) - 1))

    def duplicate_slide(self, *args):
        """复制当前页（复制图片文件，避免与原页共用 key）"""
        idx = self.list_thumb.currentRow()
        if idx < 0 or idx >= len(self.images):
            return
        src = self.images[idx]
        if not src or not os.path.exists(src):
            return

        self.push_undo()
        import time

        _, ext = os.path.splitext(src)
        dst = build_asset_path(
            self.slide_assets_dir,
            "dup",
            src,
            suffix=str(int(time.time() * 1000)),
            ext=ext or ".png",
        )
        try:
            shutil.copyfile(src, dst)
        except Exception as e:
            QMessageBox.critical(self, self._t("错误", "Error"), self._t(f"复制页面图片失败: {e}", f"Failed to duplicate slide image: {e}"))
            return

        # 深拷贝文本框数据
        boxes = copy.deepcopy(self.box_data.get(src, []))
        self.images.insert(idx + 1, dst)
        self.box_data[dst] = boxes
        self._rebuild_thumb_list(select_index=idx + 1)

    def move_slide_up(self, *args):
        idx = self.list_thumb.currentRow()
        if idx <= 0 or idx >= len(self.images):
            return
        self.push_undo()
        self.images[idx - 1], self.images[idx] = self.images[idx], self.images[idx - 1]
        self._rebuild_thumb_list(select_index=idx - 1)

    def move_slide_down(self, *args):
        idx = self.list_thumb.currentRow()
        if idx < 0 or idx >= len(self.images) - 1:
            return
        self.push_undo()
        self.images[idx + 1], self.images[idx] = self.images[idx], self.images[idx + 1]
        self._rebuild_thumb_list(select_index=idx + 1)

    def insert_text_box(self, *args):
        """在当前页插入一个默认文本框"""
        if not self.current_img:
            # 没有页时，先创建空白页
            self.new_blank_slide()
            if not self.current_img:
                return

        pix = QPixmap(self.current_img)
        if pix.isNull():
            return

        w, h = pix.width(), pix.height()
        box_w, box_h = max(120, w // 4), max(50, h // 12)
        x = max(0, (w - box_w) // 2)
        y = max(0, (h - box_h) // 2)

        self.push_undo_current_slide()
        model = {
            "rect": [int(x), int(y), int(box_w), int(box_h)],
            "text": self._t("文本框", "Text Box"),
            "confidence": 1.0,
            "clean_enabled": True,
            "clean_mode": InpaintThread.MODE_AUTO,
            "text_color_auto": False,
            "text_color_manual": True,
            "use_custom_bg": False,
            "bg_color": None,
            "bg_alpha": 120,
            "font_family": "Microsoft YaHei",
            "font_size": None,
            "bold": False,
            "align": "left",
            "text_color": [0, 0, 0],
        }
        self.box_data.setdefault(self.current_img, []).append(model)

        item = CanvasTextBox(model, "", len(self.box_data[self.current_img]) - 1, self)
        self.scene.addItem(item)
        self.on_item_clicked(item)
        self.view.viewport().update()

    def copy_selected_box(self, *args):
        if not (self.selected_box and isinstance(self.selected_box, CanvasTextBox) and isinstance(self.selected_box.model, dict)):
            return
        self._clipboard_box = copy.deepcopy(self.selected_box.model)
        self._paste_nudge = 0

    def cut_selected_box(self, *args):
        self.copy_selected_box()
        self.delete_box()

    def _paste_clipboard_image(self, show_message: bool) -> bool:
        """Paste an image (or image file URLs) from the system clipboard into the slide list."""
        cb = QApplication.clipboard()
        md = cb.mimeData()
        if md is None:
            return False

        # 1) File URLs (e.g. pasted from Explorer)
        try:
            if md.hasUrls():
                paths = []
                for u in md.urls() or []:
                    try:
                        if u.isLocalFile():
                            p = u.toLocalFile()
                        else:
                            p = ""
                    except Exception:
                        p = ""
                    if not p:
                        continue
                    ext = os.path.splitext(p)[1].lower()
                    if ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"):
                        paths.append(p)
                if paths:
                    for p in paths:
                        self._add_image_item(p)
                    self.update_status()
                    self.list_thumb.setCurrentRow(len(self.images) - 1)
                    return True
        except Exception:
            pass

        # 2) Raw image data (e.g. screenshot)
        if not md.hasImage():
            if show_message:
                QMessageBox.information(self, self._t("提示", "Info"), self._t("剪贴板里没有图片", "No image in clipboard."))
            return False

        img = cb.image()
        if img is None or img.isNull():
            try:
                pix = cb.pixmap()
                img = pix.toImage() if pix and not pix.isNull() else None
            except Exception:
                img = None
        if img is None or img.isNull():
            if show_message:
                QMessageBox.information(self, self._t("提示", "Info"), self._t("剪贴板里没有有效图片", "No valid image in clipboard."))
            return False

        # Save into project temp assets so it behaves like imported images (and gets cleaned up on exit).
        try:
            import time
            stamp = int(time.time())
            out_path = build_asset_path(
                self.slide_assets_dir,
                "clipboard",
                f"clipboard_{stamp}",
                ext=".png",
            )
            ok = img.save(out_path, "PNG")
            if not ok:
                out_path = build_asset_path(
                    self.slide_assets_dir,
                    "clipboard",
                    f"clipboard_{stamp}",
                    ext=".jpg",
                )
                ok = img.save(out_path, "JPG")
            if not ok or not os.path.exists(out_path):
                QMessageBox.warning(self, self._t("提示", "Info"), self._t("从剪贴板保存图片失败", "Failed to save clipboard image."))
                return False
        except Exception as e:
            QMessageBox.warning(
                self,
                self._t("提示", "Info"),
                self._t(f"从剪贴板导入图片失败：{e}", f"Failed to import image from clipboard: {e}"),
            )
            return False

        self._add_image_item(out_path)
        self.update_status()
        self.list_thumb.setCurrentRow(len(self.images) - 1)
        return True

    def paste_clipboard_image(self, *args):
        """UI action: paste a screenshot/image from clipboard as a new page."""
        self._paste_clipboard_image(show_message=True)

    def paste_box(self, *args):
        """将复制/剪切的文本框粘贴到当前页（偏移一点避免重叠）"""
        # If nothing was copied inside the app, allow Ctrl+V to paste a screenshot/image from clipboard.
        # Also: if no box is selected, prioritize clipboard image to support the "截图后直接粘贴识别" workflow.
        if (not self._clipboard_box) or (self._clipboard_box and self.selected_box is None):
            try:
                if self._paste_clipboard_image(show_message=False):
                    return
            except Exception:
                pass
        if not self._clipboard_box:
            return
        if not self.current_img:
            self.new_blank_slide()
            if not self.current_img:
                return

        pix = QPixmap(self.current_img)
        if pix.isNull():
            return

        self.push_undo_current_slide()
        model = copy.deepcopy(self._clipboard_box)
        rect = model.get("rect", [0, 0, 120, 50])
        if not (isinstance(rect, (list, tuple)) and len(rect) == 4):
            rect = [0, 0, 120, 50]

        dx = 10 + (self._paste_nudge % 10) * 6
        dy = 10 + (self._paste_nudge % 10) * 6
        self._paste_nudge += 1

        x, y, bw, bh = [int(v) for v in rect]
        x = min(max(0, x + dx), max(0, pix.width() - bw))
        y = min(max(0, y + dy), max(0, pix.height() - bh))
        model["rect"] = [x, y, bw, bh]

        model.setdefault("clean_enabled", True)
        model.setdefault("clean_mode", InpaintThread.MODE_AUTO)
        model.setdefault("use_custom_bg", False)
        model.setdefault("bg_color", None)

        self.box_data.setdefault(self.current_img, []).append(model)
        item = CanvasTextBox(model, "", len(self.box_data[self.current_img]) - 1, self)
        self.scene.addItem(item)
        self.on_item_clicked(item)
        self.view.viewport().update()

    def activate_format_brush(self, *args):
        """格式刷：复制当前选中框的样式，下一次点击其他框时应用（一次性）"""
        if not (self.selected_box and isinstance(self.selected_box, CanvasTextBox) and isinstance(self.selected_box.model, dict)):
            return
        self._format_brush_style = {
            "use_custom_bg": bool(self.selected_box.use_custom_bg),
            "bg_color": (
                [int(self.selected_box.custom_bg_color.red()), int(self.selected_box.custom_bg_color.green()), int(self.selected_box.custom_bg_color.blue())]
                if self.selected_box.custom_bg_color is not None else None
            ),
        }
        self._format_brush_active = True
        self.view.setCursor(Qt.CrossCursor)

    def on_ocr_result(self, image_path, results, roi_used=None):
        """处理OCR识别结果"""
        results = results or []
        # 计算缩放比例，将坐标还原到原图尺寸
        scaled_path = self.scaled_images.get(image_path, image_path)

        if scaled_path != image_path:
            # 读取原图和缩放图的尺寸
            original_img = _imread_any(image_path)
            scaled_img = _imread_any(scaled_path)

            if original_img is not None and scaled_img is not None:
                orig_h, orig_w = original_img.shape[:2]
                scaled_h, scaled_w = scaled_img.shape[:2]

                # 计算缩放比例（防止除零）
                scale_x = orig_w / max(1, scaled_w)
                scale_y = orig_h / max(1, scaled_h)

                # 还原坐标
                for result in results:
                    if 'rect' in result:
                        rect = result['rect']
                        # 确保rect是元组或列表
                        if isinstance(rect, (tuple, list)) and len(rect) == 4:
                            x, y, w, h = rect
                            result['rect'] = [
                                int(x * scale_x),
                                int(y * scale_y),
                                int(w * scale_x),
                                int(h * scale_y)
                            ]

        # 初始化每个文本框的可编辑字段（用于后续：移动/改字/自定义背景色/删除）
        # 读取原图用于提取文字颜色（用更稳的读取方式，兼容 Windows 非 ASCII 路径）
        original_img = None
        try:
            original_img = _imread_any(image_path)
        except Exception as e:
            logger.debug(f"读取原图用于颜色提取失败: {e}")

        for r in results:
            if isinstance(r, dict):
                r.setdefault("clean_enabled", True)
                r.setdefault("clean_mode", InpaintThread.MODE_AUTO)
                r.setdefault("text_color_auto", True)
                r.setdefault("text_color_manual", False)
                r.setdefault("use_custom_bg", False)
                r.setdefault("bg_color", None)  # [r,g,b] or None
                r.setdefault("bg_alpha", 120)
                r.setdefault("font_family", "Microsoft YaHei")
                r.setdefault("font_size", None)  # pt（None=自动）
                r.setdefault("bold", False)
                r.setdefault("align", "left")  # left/center/right

                # 自动提取文字颜色
                extracted_color = None
                if original_img is not None and "rect" in r:
                    extracted_color = extract_text_color_from_region(original_img, r["rect"])
                if extracted_color:
                    # Keep a record of the raw color for debugging/tuning.
                    r.setdefault("text_color_raw", extracted_color)
                    normalize_box_text_color_fields(r)
                else:
                    r.setdefault("text_color", [0, 0, 0])

        # ROI OCR：只更新选区内的文本框，保留选区外用户已调整过的框。
        merged = results
        if isinstance(roi_used, (list, tuple)) and len(roi_used) == 4:
            try:
                rx, ry, rw, rh = [int(v) for v in roi_used]
            except Exception:
                rx = ry = rw = rh = None

            if rx is not None:
                def intersects(rect_xywh):
                    if not (isinstance(rect_xywh, (list, tuple)) and len(rect_xywh) == 4):
                        return False
                    try:
                        x, y, w, h = [int(v) for v in rect_xywh]
                    except Exception:
                        return False
                    if w <= 0 or h <= 0 or rw <= 0 or rh <= 0:
                        return False
                    ax1, ay1, ax2, ay2 = x, y, x + w, y + h
                    bx1, by1, bx2, by2 = rx, ry, rx + rw, ry + rh
                    return (ax1 < bx2) and (ax2 > bx1) and (ay1 < by2) and (ay2 > by1)

                existing = self.box_data.get(image_path, []) or []
                if results:
                    kept = []
                    for b in existing:
                        if not isinstance(b, dict):
                            continue
                        r0 = b.get("rect")
                        if not intersects(r0):
                            kept.append(b)
                    merged = kept + results
                else:
                    # OCR 失败/无结果时不要把选区内旧框清空，避免误伤；用户可重试或手动删除。
                    merged = existing

        self.box_data[image_path] = merged

        # 如果是当前显示的图片，刷新显示
        if self.current_img == image_path:
            self.switch_slide(self.list_thumb.currentRow())

    def switch_slide(self, row):
        if row < 0 or row >= len(self.images): return
        self.current_img = self.images[row]
        self.selected_box = None
        if hasattr(self, "txt_edit"):
            self.txt_edit.blockSignals(True)
            self.txt_edit.clear()
            self.txt_edit.blockSignals(False)
        # 右侧面板状态重置（避免切换页后仍显示上一个框的自定义设置）
        self._reset_right_panel_state()
        self.scene.clear()
        try:
            self._refresh_auto_text_colors_for_image(self.current_img)
        except Exception:
            pass
        pix = QPixmap(self._get_display_image_path(self.current_img))
        self._current_pixmap = pix
        self._build_scene_background(pix)
        for i, b in enumerate(self.box_data.get(self.current_img, [])):
            try:
                normalize_box_text_color_fields(b)
            except Exception:
                pass
            self.scene.addItem(CanvasTextBox(b, "", i, self))
        self._draw_roi_overlay()
        self.scene.setSceneRect(-50, -50, pix.width()+100, pix.height()+100)
        self.fit_view_to_window(); self.update_status()

    def _reset_right_panel_state(self):
        """重置右侧面板控件状态（取消选中、禁用等）"""
        # 自定义背景复选框
        if hasattr(self, "chk_custom_bg") and self.chk_custom_bg is not None:
            self.chk_custom_bg.blockSignals(True)
            self.chk_custom_bg.setChecked(False)
            self.chk_custom_bg.blockSignals(False)
        if hasattr(self, "chk_clean_enabled") and self.chk_clean_enabled is not None:
            self.chk_clean_enabled.blockSignals(True)
            self.chk_clean_enabled.setChecked(True)
            self.chk_clean_enabled.blockSignals(False)
        if hasattr(self, "cmb_clean_mode") and self.cmb_clean_mode is not None:
            idx = self.cmb_clean_mode.findData(InpaintThread.MODE_AUTO)
            self.cmb_clean_mode.blockSignals(True)
            self.cmb_clean_mode.setCurrentIndex(0 if idx < 0 else idx)
            self.cmb_clean_mode.blockSignals(False)
        # 需要禁用的按钮/控件列表
        self._enable_right_panel_widgets(False)

    def _refresh_auto_text_colors_for_image(self, image_path):
        if not image_path:
            return
        boxes = self.box_data.get(image_path, []) or []
        candidates = [b for b in boxes if isinstance(b, dict) and should_auto_refresh_text_color(b)]
        if not candidates:
            return
        original_img = _imread_any(image_path)
        if original_img is None:
            return
        for box in candidates:
            rect = box.get("rect")
            if not (isinstance(rect, (list, tuple)) and len(rect) == 4):
                continue
            try:
                extracted_color = extract_text_color_from_region(original_img, rect)
            except Exception:
                extracted_color = None
            if extracted_color:
                box["text_color_raw"] = [int(v) for v in extracted_color]
                normalize_box_text_color_fields(box)

    def _enable_right_panel_widgets(self, enabled: bool):
        """批量启用/禁用右侧面板中与文本框属性相关的控件"""
        widgets_to_toggle = [
            "btn_choose_custom_color",
            "btn_pick_custom_color",
            "btn_text_color",
            "chk_bold",
            "btn_align_left",
            "btn_align_center",
            "btn_align_right",
            "slider_bg_alpha",
            "chk_clean_enabled",
            "cmb_clean_mode",
            "btn_apply_style_page",
        ]
        for name in widgets_to_toggle:
            widget = getattr(self, name, None)
            if widget is not None:
                try:
                    widget.setEnabled(enabled)
                except Exception:
                    pass

    def _build_scene_background(self, pix: QPixmap):
        """创建/重建画布背景层（阴影/白底/图片），并固定 z 值，避免重绘时底图层级异常。"""
        try:
            if pix is None or pix.isNull():
                # 图片加载失败时，至少保证背景层引用清空，避免后续误操作
                self._bg_shadow_item = None
                self._bg_white_item = None
                self._bg_pixmap_item = None
                return

            # 若之前有旧的背景 item（例如透明度拖动导致重绘伪影），先移除，避免重复叠加
            for attr in ("_bg_shadow_item", "_bg_white_item", "_bg_pixmap_item"):
                it = getattr(self, attr, None)
                try:
                    if it is not None and hasattr(it, "scene") and it.scene() is self.scene:
                        self.scene.removeItem(it)
                except Exception:
                    pass

            w, h = pix.width(), pix.height()
            sd = QGraphicsRectItem(4, 4, w, h)
            sd.setBrush(QColor(0, 0, 0, 40))
            sd.setPen(Qt.NoPen)
            sd.setZValue(-30)
            self.scene.addItem(sd)

            bg = QGraphicsRectItem(0, 0, w, h)
            bg.setBrush(Qt.white)
            bg.setPen(Qt.NoPen)
            bg.setZValue(-20)
            self.scene.addItem(bg)

            pm = self.scene.addPixmap(pix)
            pm.setZValue(-10)
            try:
                pm.setCacheMode(QGraphicsItem.NoCache)
            except Exception:
                pass

            self._bg_shadow_item = sd
            self._bg_white_item = bg
            self._bg_pixmap_item = pm
        except Exception as e:
            logger.warning(f"重建背景层失败: {e}")

    def _ensure_scene_background(self):
        """确保背景图片层存在且有效；用于拖动透明度时修复 Qt 偶发的“底图不见”重绘问题。"""
        try:
            if not getattr(self, "scene", None) or not getattr(self, "current_img", None):
                return

            pix = getattr(self, "_current_pixmap", None)
            if pix is None or pix.isNull():
                pix = QPixmap(self.current_img)
                self._current_pixmap = pix

            if pix is None or pix.isNull():
                return

            pm = getattr(self, "_bg_pixmap_item", None)
            if pm is None or (hasattr(pm, "scene") and pm.scene() is not self.scene):
                self._build_scene_background(pix)
                return

            # 重新 setPixmap 会触发底层刷新；能修复“拖动后不重绘”的情况
            try:
                pm.setPixmap(pix)
                pm.update()
            except Exception:
                self._build_scene_background(pix)
        except Exception:
            pass

    def fit_view_to_window(self):
        if self.scene.itemsBoundingRect().width() > 0:
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self.view.scale(0.9, 0.9)
            self._update_zoom_label()

    def _nudge_zoom(self, delta_percent: int):
        try:
            if not hasattr(self, "zoom_slider") or self.zoom_slider is None:
                return
            curr = int(self.zoom_slider.value())
            nxt = max(int(self.zoom_slider.minimum()), min(int(self.zoom_slider.maximum()), curr + int(delta_percent)))
            self.zoom_slider.setValue(nxt)
        except Exception:
            pass

    def zoom_in(self):
        self._nudge_zoom(10)

    def zoom_out(self):
        self._nudge_zoom(-10)

    def goto_prev_slide(self):
        try:
            if not self.images:
                return
            idx = int(self.list_thumb.currentRow())
            if idx <= 0:
                return
            self.list_thumb.setCurrentRow(idx - 1)
        except Exception:
            pass

    def goto_next_slide(self):
        try:
            if not self.images:
                return
            idx = int(self.list_thumb.currentRow())
            if idx < 0:
                idx = 0
            if idx >= len(self.images) - 1:
                return
            self.list_thumb.setCurrentRow(idx + 1)
        except Exception:
            pass

    def toggle_left_panel(self):
        """Show/hide the left thumbnail list."""
        self._show_left_panel = not bool(getattr(self, "_show_left_panel", True))
        try:
            if self._show_left_panel:
                self.list_thumb.setMinimumWidth(230)
                self.list_thumb.setMaximumWidth(230)
            else:
                self.list_thumb.setMinimumWidth(0)
                self.list_thumb.setMaximumWidth(0)
        except Exception:
            pass
        try:
            self._fix_splitter_sizes()
        except Exception:
            pass

    def toggle_right_panel(self):
        """Show/hide the right properties panel."""
        self._show_right_panel = not bool(getattr(self, "_show_right_panel", True))
        try:
            if self._show_right_panel:
                self.right_panel_scroll.setMinimumWidth(RIGHT_PANEL_W)
                self.right_panel_scroll.setMaximumWidth(RIGHT_PANEL_W)
            else:
                self.right_panel_scroll.setMinimumWidth(0)
                self.right_panel_scroll.setMaximumWidth(0)
        except Exception:
            pass
        try:
            self._fix_splitter_sizes()
        except Exception:
            pass

    def show_shortcuts(self):
        dlg = ShortcutsDialog(self)
        dlg.exec()
    def zoom_view(self):
        val = self.zoom_slider.value() / 100.0
        self.view.resetTransform(); self.view.scale(val, val)
        self.lbl_zoom_val.setText(f"{self.zoom_slider.value()}%")
    def _update_zoom_label(self):
        curr = int(self.view.transform().m11() * 100)
        self.zoom_slider.blockSignals(True); self.zoom_slider.setValue(curr); self.zoom_slider.blockSignals(False)
        self.lbl_zoom_val.setText(f"{curr}%")
    def on_item_clicked(self, item):
        # 格式刷：先应用样式（一次性）
        if self._format_brush_active and self._format_brush_style and isinstance(item, CanvasTextBox):
            try:
                st = self._format_brush_style
                item.use_custom_bg = bool(st.get("use_custom_bg", False))
                bg = st.get("bg_color")
                if isinstance(bg, (list, tuple)) and len(bg) == 3:
                    item.custom_bg_color = QColor(int(bg[0]), int(bg[1]), int(bg[2]))
                else:
                    item.custom_bg_color = None
                item._sync_model_bg()
                item.update_background()
            finally:
                self._format_brush_active = False
                self._format_brush_style = None
                self.view.setCursor(Qt.ArrowCursor)

        self.selected_box = item
        for i in self.scene.items():
            if isinstance(i, CanvasTextBox): i.setSelected(i==item)
        self.txt_edit.blockSignals(True)
        for c in item.childItems():
            if isinstance(c, QGraphicsTextItem):
                self.txt_edit.setText(c.toPlainText())
        self.txt_edit.blockSignals(False)

        # 更新右侧栏的自定义颜色状态
        if hasattr(item, 'use_custom_bg') and hasattr(item, 'custom_bg_color'):
            try:
                self.chk_custom_bg.blockSignals(True)
                self.chk_custom_bg.setChecked(item.use_custom_bg)
            finally:
                self.chk_custom_bg.blockSignals(False)
            if item.custom_bg_color:
                self.update_custom_color_preview(item.custom_bg_color)
            else:
                self.update_custom_color_preview(QColor(255, 255, 255))

        # 只要选中了文本框，就允许点"选择/吸管"；是否启用背景由"自定义"勾选控制
        self._enable_right_panel_widgets(True)

        self.refresh_right_panel_from_selected()

    def _get_selected_model(self):
        if self.selected_box and isinstance(self.selected_box, CanvasTextBox) and isinstance(self.selected_box.model, dict):
            return self.selected_box.model
        return None

    def refresh_right_panel_from_selected(self):
        """将右侧 UI 与当前选中文本框同步"""
        m = self._get_selected_model()
        if not m:
            return

        # 字体大小
        try:
            fs = m.get("font_size")
            if fs is None:
                fs = self.selected_box.txt.font().pointSize() or 12
            self.slider_font.blockSignals(True)
            self.slider_font.setValue(int(fs))
            self.slider_font.blockSignals(False)
        except Exception:
            pass

        # 文字颜色
        tc = m.get("text_color", [0, 0, 0])
        if isinstance(tc, (list, tuple)) and len(tc) == 3:
            try:
                self.text_color_preview.setStyleSheet(
                    f"background-color: rgb({int(tc[0])},{int(tc[1])},{int(tc[2])}); border: 1px solid #ccc;"
                )
            except Exception:
                pass

        # 加粗
        try:
            self.chk_bold.blockSignals(True)
            self.chk_bold.setChecked(bool(m.get("bold", False)))
            self.chk_bold.blockSignals(False)
        except Exception:
            pass

        # 对齐（按钮不做选中态，只同步 model）
        # 背景透明度
        try:
            self.slider_bg_alpha.blockSignals(True)
            # UI 使用"透明度"，内部存 alpha
            self.slider_bg_alpha.setValue(255 - int(m.get("bg_alpha", 120)))
            self.slider_bg_alpha.blockSignals(False)
        except Exception:
            pass

        try:
            self.chk_clean_enabled.blockSignals(True)
            self.chk_clean_enabled.setChecked(bool(m.get("clean_enabled", True)))
            self.chk_clean_enabled.blockSignals(False)
        except Exception:
            pass

        try:
            mode = InpaintThread._normalize_box_clean_mode(m.get("clean_mode"))
            idx = self.cmb_clean_mode.findData(mode)
            self.cmb_clean_mode.blockSignals(True)
            self.cmb_clean_mode.setCurrentIndex(0 if idx < 0 else idx)
            self.cmb_clean_mode.blockSignals(False)
        except Exception:
            pass

        # 注意：边框相关控件已移除，不再同步边框设置

    def _apply_selected_style(self):
        if self.selected_box and isinstance(self.selected_box, CanvasTextBox) and isinstance(self.selected_box.model, dict):
            self.selected_box.apply_style_from_model()
            self.selected_box._sync_model_bg()
            self._force_canvas_redraw()

    def on_font_size_changed(self, val):
        m = self._get_selected_model()
        if not m:
            return
        m["font_size"] = int(val)
        self._apply_selected_style()

    def choose_text_color(self, *args):
        m = self._get_selected_model()
        if not m:
            return
        tc = m.get("text_color", [0, 0, 0])
        try:
            initial = QColor(int(tc[0]), int(tc[1]), int(tc[2]))
        except Exception:
            initial = QColor(0, 0, 0)
        color = QColorDialog.getColor(initial, self, self._t("选择文字颜色", "Choose text color"))
        if color.isValid():
            self.push_undo_current_slide()
            m["text_color"] = [color.red(), color.green(), color.blue()]
            m["text_color_auto"] = False
            m["text_color_manual"] = True
            self.refresh_right_panel_from_selected()
            self._apply_selected_style()

    def toggle_bold(self, state):
        m = self._get_selected_model()
        if not m:
            return
        self.push_undo_current_slide()
        m["bold"] = bool(state)
        self._apply_selected_style()

    def set_align(self, align: str):
        m = self._get_selected_model()
        if not m:
            return
        self.push_undo_current_slide()
        m["align"] = align
        self._apply_selected_style()

    def on_bg_alpha_changed(self, val):
        """单个文本框透明度 - 拖动时轻量更新，释放时重建"""
        m = self._get_selected_model()
        if not m:
            return
        # UI val=透明度(0=不透明,255=全透明) -> alpha(0=全透明,255=不透明)
        try:
            m["bg_alpha"] = max(0, min(255, 255 - int(val)))
        except Exception:
            m["bg_alpha"] = DEFAULT_BG_ALPHA
        self._apply_selected_style()

        # 只有在非拖动状态时才触发场景重建
        if not (hasattr(self, "slider_bg_alpha") and
                self.slider_bg_alpha and
                self.slider_bg_alpha.isSliderDown()):
            self._schedule_scene_rebuild()

    def toggle_selected_clean_enabled(self, state):
        m = self._get_selected_model()
        if not m:
            return
        enabled = bool(state)
        if bool(m.get("clean_enabled", True)) == enabled:
            return
        self.push_undo_current_slide()
        m["clean_enabled"] = enabled
        if self.selected_box and isinstance(self.selected_box, CanvasTextBox):
            self.selected_box.refresh_clean_outline()

    def change_selected_clean_mode(self, *_args):
        m = self._get_selected_model()
        if not m:
            return
        mode = InpaintThread._normalize_box_clean_mode(self.cmb_clean_mode.currentData())
        if InpaintThread._normalize_box_clean_mode(m.get("clean_mode")) == mode:
            return
        self.push_undo_current_slide()
        m["clean_mode"] = mode
        if self.selected_box and isinstance(self.selected_box, CanvasTextBox):
            self.selected_box.refresh_clean_outline()

    # 注意：边框相关方法 (toggle_border, choose_border_color, on_border_width_changed) 已移除
    # 因为右侧面板中没有创建对应的边框控件

    def apply_style_to_current_slide(self, *args):
        """把当前选中文本框的样式批量应用到本页其他文本框（不改位置/文字）"""
        if not self.current_img:
            return
        src = self._get_selected_model()
        if not src:
            return
        self.push_undo_current_slide()
        keys = [
            "use_custom_bg", "bg_color", "bg_alpha",
            "font_family", "font_size", "bold", "align", "text_color",
        ]
        boxes = self.box_data.get(self.current_img, [])
        for b in boxes:
            if not isinstance(b, dict):
                continue
            if b is src:
                continue
            for k in keys:
                b[k] = copy.deepcopy(src.get(k))

        # 画布上同步刷新
        for it in self.scene.items():
            if isinstance(it, CanvasTextBox) and isinstance(it.model, dict):
                it.apply_style_from_model()
        self.view.viewport().update()
    def sync_text_change(self):
        if self.selected_box:
            for c in self.selected_box.childItems():
                if isinstance(c, QGraphicsTextItem):
                    new_text = self.txt_edit.toPlainText()
                    c.setPlainText(new_text)
                    if isinstance(self.selected_box, CanvasTextBox) and isinstance(self.selected_box.model, dict):
                        self.selected_box.model["text"] = new_text
    def delete_box(self, *args):
        """删除选中文本框（同时更新 box_data，保证预览/导出一致）"""
        if not (self.current_img and self.selected_box and isinstance(self.selected_box, CanvasTextBox)):
            return

        self.push_undo_current_slide()
        # 从 model 中删除：用“对象身份”匹配，避免 dict 内含 numpy array 时触发 == 比较报错
        try:
            boxes = self.box_data.get(self.current_img, [])
            removed = False
            # 优先按 index 删除（更快），但要校验身份一致
            try:
                idx = int(getattr(self.selected_box, "model_index", -1))
            except Exception:
                idx = -1
            if 0 <= idx < len(boxes) and boxes[idx] is self.selected_box.model:
                del boxes[idx]
                removed = True
            else:
                for i, b in enumerate(list(boxes)):
                    if b is self.selected_box.model:
                        del boxes[i]
                        removed = True
                        break
            if not removed:
                logger.warning("删除文本框同步数据失败: 未在 box_data 中找到对应对象（可能已被删除/替换）")
        except Exception as e:
            logger.warning(f"删除文本框同步数据失败: {e}")

        # 从场景中删除
        self.scene.removeItem(self.selected_box)
        self.selected_box = None
        try:
            self.view.viewport().update()
        except Exception:
            pass

        self.txt_edit.blockSignals(True)
        self.txt_edit.clear()
        self.txt_edit.blockSignals(False)

        # 重置右侧面板状态
        self._reset_right_panel_state()

    def toggle_custom_bg(self, state):
        """切换自定义背景色"""
        enabled = bool(state)
        has_sel = bool(self.selected_box and isinstance(self.selected_box, CanvasTextBox))
        # “选择/吸管”只跟是否选中了框有关；即使未启用，也可以先选颜色再启用
        self.btn_choose_custom_color.setEnabled(has_sel)
        self.btn_pick_custom_color.setEnabled(has_sel)

        if self.selected_box and isinstance(self.selected_box, CanvasTextBox):
            self.push_undo_current_slide()
            self.selected_box.use_custom_bg = enabled
            self.selected_box._sync_model_bg()
            self.selected_box.update_background()

    def choose_custom_color(self, *args):
        """为选中的文本框选择自定义颜色"""
        if not self.selected_box or not isinstance(self.selected_box, CanvasTextBox):
            return

        # 用户点了“选择”，默认就是要启用自定义背景
        if not self.chk_custom_bg.isChecked():
            self.chk_custom_bg.setChecked(True)

        initial_color = self.selected_box.custom_bg_color if self.selected_box.custom_bg_color else QColor(255, 255, 255)
        color = QColorDialog.getColor(initial_color, self, self._t("选择文本框背景色", "Choose background color"))

        if color.isValid():
            self.push_undo_current_slide()
            self.selected_box.custom_bg_color = color
            self.selected_box.use_custom_bg = True
            self.selected_box._sync_model_bg()
            self.chk_custom_bg.setChecked(True)
            self.update_custom_color_preview(color)
            self.selected_box.update_background()

    def pick_custom_color(self, *args):
        """兼容旧调用：从画布上吸取颜色用于选中的文本框"""
        if hasattr(self, "btn_pick_custom_color"):
            self.btn_pick_custom_color.setChecked(True)

    def toggle_selected_eyedropper(self, checked: bool):
        """右侧面板吸管：为选中的文本框取色"""
        if checked:
            if not (self.current_img and self.selected_box and isinstance(self.selected_box, CanvasTextBox)):
                # 没有选中框就不进入吸管模式
                self.btn_pick_custom_color.blockSignals(True)
                self.btn_pick_custom_color.setChecked(False)
                self.btn_pick_custom_color.blockSignals(False)
                return
            if not self.chk_custom_bg.isChecked():
                self.chk_custom_bg.setChecked(True)
            self.eyedropper_mode = True
            self.picking_for_selected = True

            # 同步顶部“视图”里的吸管按钮状态（如果存在），但不弹两次提示
            if hasattr(self, "btn_eyedropper"):
                self.btn_eyedropper.blockSignals(True)
                self.btn_eyedropper.setChecked(True)
                self.btn_eyedropper.blockSignals(False)

            self.view.setCursor(Qt.CrossCursor)
            QMessageBox.information(
                self,
                self._t("吸管工具", "Eyedropper"),
                self._t("点击画布上的任意位置取色（用于当前选中文本框）", "Click on the canvas to pick a color (for the selected box)."),
            )
        else:
            self.picking_for_selected = False
            self.eyedropper_mode = False
            if hasattr(self, "btn_eyedropper"):
                self.btn_eyedropper.blockSignals(True)
                self.btn_eyedropper.setChecked(False)
                self.btn_eyedropper.blockSignals(False)
            self.view.setCursor(Qt.ArrowCursor)

    def update_custom_color_preview(self, color):
        """更新自定义颜色预览"""
        self.custom_color_preview.setStyleSheet(
            f"background-color: rgb({color.red()}, {color.green()}, {color.blue()}); "
            f"border: 1px solid #ccc;"
        )

    def fit_font_size_pt_like_ppt(self, text: str, box_w_px, box_h_px) -> int:
        """Return a font size in pt using the same fitting routine as PPT export.

        This keeps the canvas preview closer to the final PPT appearance.
        支持大尺寸图片（如 5504x3072 / 4K 截图），字体大小上限提高到 600pt。
        """
        MAX_PT = FONT_SIZE_MAX
        try:
            w = max(1, int(round(float(box_w_px or 0))))
            h = max(1, int(round(float(box_h_px or 0))))
        except Exception:
            w, h = 1, 1

        # Lazy init to avoid paying pptx/PIL cost unless we actually need font fitting.
        if self._ppt_exporter_metrics is None:
            self._ppt_exporter_metrics = PPTExporter(text_bg_color=None)

        try:
            pt = self._ppt_exporter_metrics.fit_font_size(
                str(text or ""),
                w,
                h,
                dpi=96,
                padding_x=2,
                padding_y=2,
                max_pt=MAX_PT,
            )
            return int(max(6, min(MAX_PT, int(round(float(pt))))))
        except Exception:
            # Fallback: simple height-based estimate (still in pt at 96 DPI mapping).
            est = int(round(max(6.0, min(float(MAX_PT), (h * 72.0 / 96.0) * 0.8))))
            return int(est)

    def _prepare_boxes_for_ppt_export(self, image_path: str, boxes):
        """Prepare a non-destructive copy of boxes for PPT export.

        - Ensure auto font sizes are filled so slides not opened in the canvas still export correctly.
        - Account for PPT's slide size cap (image may be scaled down when max dimension > 5000px).
        """
        src_boxes = boxes or []
        if not isinstance(src_boxes, (list, tuple)) or not src_boxes:
            return list(src_boxes) if isinstance(src_boxes, (list, tuple)) else []

        ppt_scale = 1.0
        try:
            from PIL import Image

            p = self._get_export_image_path(image_path)
            with Image.open(p) as img:
                _, _, ppt_scale = PPTExporter._scale_to_ppt_limit(*img.size)
        except Exception:
            ppt_scale = 1.0

        out = []
        for item in src_boxes:
            if not isinstance(item, dict):
                out.append(item)
                continue

            m = copy.deepcopy(item)
            rect = m.get("rect")
            if not (isinstance(rect, (list, tuple)) and len(rect) == 4):
                out.append(m)
                continue

            try:
                _, _, w, h = [float(v) for v in rect]
            except Exception:
                out.append(m)
                continue

            # Compute in the same "scaled slide" coordinate system as PPTExporter.
            w_s = max(1, int(round(float(w) * float(ppt_scale))))
            h_s = max(1, int(round(float(h) * float(ppt_scale))))

            # If font_size is missing/invalid, or looks like it hit the old 200pt cap on a tall box,
            # re-fit with the higher cap.
            fs_raw = m.get("font_size")
            try:
                fs = int(fs_raw) if fs_raw is not None else None
            except Exception:
                fs = None

            need_fit = fs is None
            try:
                # Heuristic: existing 200pt often means "capped" for big titles; only override when
                # the box height clearly allows a much larger size.
                if (not need_fit) and fs is not None and fs >= 200 and fs <= 205:
                    h_pt_est = float(h_s) * 72.0 / 96.0
                    if h_pt_est > (float(fs) + 30.0):
                        need_fit = True
            except Exception:
                pass

            if need_fit:
                try:
                    m["font_size"] = int(self.fit_font_size_pt_like_ppt(str(m.get("text") or ""), w_s, h_s))
                except Exception:
                    pass

            out.append(m)

        return out

    def _collect_ppt_export_items(self):
        items = []
        for img_path in self.images:
            export_path = self._get_export_image_path(img_path)
            boxes = self._prepare_boxes_for_ppt_export(img_path, self.box_data.get(img_path, []))
            items.append((img_path, export_path, boxes))
        return items

    def _build_ppt_exporter(self, export_items):
        slide_size_px = PPTExporter.presentation_size_for_images([item[1] for item in export_items])
        kwargs = {"slide_size_px": slide_size_px}
        if self.use_text_bg:
            color = self.text_bg_color
            kwargs["text_bg_color"] = (color.red(), color.green(), color.blue())
            kwargs["text_bg_alpha"] = int(getattr(self, "text_bg_alpha", 200))
        else:
            kwargs["text_bg_color"] = None
        return PPTExporter(**kwargs)

    def _export_ppt_to_path(self, output_path: str) -> bool:
        export_items = self._collect_ppt_export_items()
        exporter = self._build_ppt_exporter(export_items)
        for _, export_path, boxes in export_items:
            exporter.add_image_with_text_boxes(export_path, boxes)
        return exporter.save(output_path)

    def export_ppt(self):
        """导出PPT"""
        if not self.images:
            QMessageBox.warning(self, self._t("提示", "Info"), self._t("请先导入图片", "Please import images first."))
            return

        if not any(self.box_data.values()):
            QMessageBox.warning(self, self._t("提示", "Info"), self._t("请先运行OCR识别", "Please run OCR first."))
            return

        save_path, _ = QFileDialog.getSaveFileName(self, self._t("保存PPT", "Save PPT"), "", "PowerPoint (*.pptx)")
        if not save_path:
            return

        try:
            if self._export_ppt_to_path(save_path):
                QMessageBox.information(self, self._t("成功", "Success"), self._t(f"PPT已导出到:\n{save_path}", f"PPT exported to:\n{save_path}"))
            else:
                QMessageBox.warning(self, self._t("失败", "Failed"), self._t("PPT导出失败", "Failed to export PPT."))

        except Exception as e:
            QMessageBox.critical(self, self._t("错误", "Error"), self._t(f"导出失败: {str(e)}", f"Export failed: {str(e)}"))

    def toggle_text_bg(self, state):
        """切换文本框背景色设置"""
        # PySide6: stateChanged 传 int(0/2)，不要和 Qt.Checked(enum) 直接比较
        prev = bool(getattr(self, "use_text_bg", False))
        self.use_text_bg = bool(state)

        # 体验优化：第一次开启全局背景时，默认 alpha 太大会把底图“盖住”，自动调低一次（用户手动调过就不干预）
        if (not prev) and self.use_text_bg and (not getattr(self, "_user_set_global_bg_alpha", False)):
            try:
                if int(getattr(self, "text_bg_alpha", 120)) > 160:
                    self.text_bg_alpha = DEFAULT_BG_ALPHA
                    if hasattr(self, "slider_global_alpha"):
                        self.slider_global_alpha.blockSignals(True)
                        # UI 透明度：255-alpha
                        self.slider_global_alpha.setValue(135)
                        self.slider_global_alpha.blockSignals(False)
            except Exception:
                pass
        logger.debug(f"文本框背景色: {'开启' if self.use_text_bg else '关闭'}")
        # 更新画布上所有文本框的背景色
        self.update_all_text_boxes_background()

    def on_global_bg_alpha_changed(self, val):
        """全局背景透明度（0-255）- 拖动时轻量更新，释放时重建"""
        try:
            # UI val=透明度(0=不透明,255=全透明) -> alpha(0=全透明,255=不透明)
            self.text_bg_alpha = max(0, min(255, 255 - int(val)))
        except Exception:
            self.text_bg_alpha = 200
        self._user_set_global_bg_alpha = True

        # 拖动中：只更新文本框背景，不重建场景（避免卡顿）
        self.update_all_text_boxes_background()

        # 只有在非拖动状态时才触发场景重建
        if not (hasattr(self, "slider_global_alpha") and
                self.slider_global_alpha and
                self.slider_global_alpha.isSliderDown()):
            self._schedule_scene_rebuild()

    def _schedule_scene_rebuild(self):
        """短延迟兜底：重建当前页场景，修复透明度拖动时偶发的底图消失/不刷新。"""
        try:
            if hasattr(self, "_scene_rebuild_timer") and self._scene_rebuild_timer:
                self._scene_rebuild_timer.start()
        except Exception:
            pass

    def _rebuild_scene_keep_view(self):
        """重建当前页，但尽量保留视图缩放/中心点/选中项（用于重绘兜底）。"""
        try:
            # 用户仍在拖动滑块时不要重建，避免拖动过程中闪烁/卡顿
            try:
                if hasattr(self, "slider_global_alpha") and self.slider_global_alpha and self.slider_global_alpha.isSliderDown():
                    self._schedule_scene_rebuild()
                    return
                if hasattr(self, "slider_bg_alpha") and self.slider_bg_alpha and self.slider_bg_alpha.isSliderDown():
                    self._schedule_scene_rebuild()
                    return
            except Exception:
                pass

            row = self.list_thumb.currentRow() if hasattr(self, "list_thumb") else -1
            if row < 0 or row >= len(self.images):
                return

            # 记录视图状态
            try:
                center = self.view.mapToScene(self.view.viewport().rect().center())
            except Exception:
                center = None
            try:
                tf = self.view.transform()
            except Exception:
                tf = None

            sel_idx = None
            try:
                if self.selected_box and isinstance(self.selected_box, CanvasTextBox):
                    sel_idx = int(getattr(self.selected_box, "model_index", -1))
            except Exception:
                sel_idx = None

            # 重建
            self.switch_slide(row)

            # 恢复视图
            try:
                if tf is not None:
                    self.view.setTransform(tf)
                    self._update_zoom_label()
                if center is not None:
                    self.view.centerOn(center)
            except Exception:
                pass

            # 恢复选中
            if sel_idx is not None and sel_idx >= 0:
                try:
                    for it in self.scene.items():
                        if isinstance(it, CanvasTextBox) and int(getattr(it, "model_index", -1)) == sel_idx:
                            self.on_item_clicked(it)
                            break
                except Exception:
                    pass
        except Exception:
            pass

    def update_color_preview(self):
        """更新颜色预览框"""
        color = self.text_bg_color
        self.color_preview.setStyleSheet(f"""
            QLabel {{
                background-color: rgb({color.red()}, {color.green()}, {color.blue()});
                border: 2px solid #999;
                border-radius: 4px;
            }}
        """)

    def pick_color(self):
        """打开颜色选择器"""
        color = QColorDialog.getColor(self.text_bg_color, self, self._t("选择文本框背景色", "Choose background color"))
        if color.isValid():
            self.text_bg_color = color
            self.update_color_preview()
            logger.debug(f"选择颜色: RGB({color.red()}, {color.green()}, {color.blue()})")
            # 更新画布上所有文本框的背景色
            self.update_all_text_boxes_background()

    def toggle_eyedropper(self, checked):
        """切换吸管模式"""
        self.eyedropper_mode = checked
        # 顶部“视图”吸管默认作用于全局背景色；避免残留为“单框取色”
        self.picking_for_selected = False
        if checked and hasattr(self, "btn_pick_custom_color"):
            # 用户明确点了全局吸管，则取消右侧单框吸管状态
            try:
                self.btn_pick_custom_color.blockSignals(True)
                self.btn_pick_custom_color.setChecked(False)
                self.btn_pick_custom_color.blockSignals(False)
            except Exception:
                pass
        if checked:
            self.view.setCursor(Qt.CrossCursor)
            QMessageBox.information(
                self,
                self._t("吸管工具", "Eyedropper"),
                self._t("点击画布上的任意位置取色", "Click on the canvas to pick a color."),
            )
        else:
            self.view.setCursor(Qt.ArrowCursor)

    def update_all_text_boxes_background(self):
        """更新画布上所有文本框的背景色"""
        if hasattr(self, 'scene') and self.scene:
            # QGraphicsScene.items() 可能返回 group 子项而不是 group 本身；这里向上找 parentItem
            # 以确保切换全局背景色时所有文本框都能被刷新。
            count = 0
            seen = set()
            for item in self.scene.items():
                cur = item
                while cur is not None:
                    if isinstance(cur, CanvasTextBox):
                        key = id(cur)
                        if key not in seen:
                            cur.update_background()
                            seen.add(key)
                            count += 1
                        break
                    try:
                        cur = cur.parentItem()
                    except Exception:
                        cur = None
        logger.debug(f"已更新 {count} 个文本框的背景色")
        self._force_canvas_redraw()

    def _force_canvas_redraw(self):
        """强制整个画布重绘（用于解决透明度调整后偶发的“底图未刷新/消失”问题）"""
        try:
            # 透明度频繁变化时，Qt 偶发不重绘底图；先确保背景层还在
            self._ensure_scene_background()
            if hasattr(self, "scene") and self.scene:
                try:
                    # invalidate 会让 Qt 重新绘制被透明项“揭露出来”的底图区域
                    self.scene.invalidate(self.scene.sceneRect(), QGraphicsScene.AllLayers)
                except Exception:
                    self.scene.update(self.scene.sceneRect())
            if hasattr(self, "view") and self.view:
                try:
                    self.view.invalidateScene(self.scene.sceneRect(), QGraphicsScene.AllLayers)
                except Exception:
                    pass
                # update() 无参会刷新整个 viewport，避免透明度变化后露出的底图没被重绘
                self.view.viewport().update()
                self.view.viewport().repaint()
        except Exception:
            pass

    # ==================== ROI selection (for OCR / IOPaint) ====================
    def _draw_roi_overlay(self):
        """Draw ROI overlay rectangle for the current slide (if any)."""
        try:
            if self._roi_item is not None:
                self.scene.removeItem(self._roi_item)
        except Exception:
            pass
        self._roi_item = None

        if not self.current_img:
            return
        roi = (getattr(self, "roi_by_image", {}) or {}).get(self.current_img)
        if not (isinstance(roi, (list, tuple)) and len(roi) == 4):
            return
        try:
            x, y, w, h = [int(v) for v in roi]
        except Exception:
            return
        if w <= 0 or h <= 0:
            return
        try:
            pen = QPen(QColor(210, 50, 38), 2, Qt.DashLine)
            pen.setCosmetic(True)
            item = QGraphicsRectItem(x, y, w, h)
            item.setPen(pen)
            item.setBrush(Qt.NoBrush)
            item.setZValue(10_000)
            item.setAcceptedMouseButtons(Qt.NoButton)
            self.scene.addItem(item)
            self._roi_item = item
        except Exception:
            self._roi_item = None

    def set_roi_select_mode(self, enabled: bool):
        """Enable/disable ROI select mode. Drag on canvas to set a region."""
        enabled = bool(enabled)
        self.roi_select_mode = enabled
        try:
            self._roi_drag_start = None
        except Exception:
            pass

        # Sync ribbon toggle button state (if present).
        btn = getattr(self, "btn_roi_select", None)
        if btn is not None:
            try:
                if bool(btn.isChecked()) != enabled:
                    btn.blockSignals(True)
                    btn.setChecked(enabled)
                    btn.blockSignals(False)
            except Exception:
                pass

        # Cursor feedback.
        try:
            if enabled:
                self.view.setCursor(Qt.CrossCursor)
            else:
                self.view.setCursor(Qt.CrossCursor if getattr(self, "eyedropper_mode", False) else Qt.ArrowCursor)
        except Exception:
            pass

    def toggle_roi_select_mode(self, *args):
        self.set_roi_select_mode(not bool(getattr(self, "roi_select_mode", False)))

    def clear_roi_current(self, *args):
        if not self.current_img:
            return
        try:
            self.push_undo_current_slide()
        except Exception:
            pass
        try:
            self.roi_by_image.pop(self.current_img, None)
        except Exception:
            pass
        self._draw_roi_overlay()
        try:
            self.view.viewport().update()
        except Exception:
            pass

    def canvas_roi_press(self, event):
        if not self.current_img:
            return
        try:
            pos = self.view.mapToScene(event.position().toPoint() if hasattr(event, "position") else event.pos())
            self._roi_drag_start = QPointF(pos.x(), pos.y())
        except Exception:
            self._roi_drag_start = None

    def canvas_roi_move(self, event):
        if self._roi_drag_start is None or not self.current_img:
            return
        try:
            pos = self.view.mapToScene(event.position().toPoint() if hasattr(event, "position") else event.pos())
            x1 = float(self._roi_drag_start.x())
            y1 = float(self._roi_drag_start.y())
            x2 = float(pos.x())
            y2 = float(pos.y())
            x = int(round(min(x1, x2)))
            y = int(round(min(y1, y2)))
            w = int(round(abs(x2 - x1)))
            h = int(round(abs(y2 - y1)))
            # draw temp overlay (do not commit)
            try:
                if self._roi_item is not None:
                    self.scene.removeItem(self._roi_item)
            except Exception:
                pass
            pen = QPen(QColor(210, 50, 38), 2, Qt.DashLine)
            pen.setCosmetic(True)
            item = QGraphicsRectItem(x, y, w, h)
            item.setPen(pen)
            item.setBrush(Qt.NoBrush)
            item.setZValue(10_000)
            item.setAcceptedMouseButtons(Qt.NoButton)
            self.scene.addItem(item)
            self._roi_item = item
        except Exception:
            pass

    def canvas_roi_release(self, event):
        if self._roi_drag_start is None or not self.current_img:
            self.set_roi_select_mode(False)
            return
        try:
            pos = self.view.mapToScene(event.position().toPoint() if hasattr(event, "position") else event.pos())
            x1 = float(self._roi_drag_start.x())
            y1 = float(self._roi_drag_start.y())
            x2 = float(pos.x())
            y2 = float(pos.y())
            x = int(round(min(x1, x2)))
            y = int(round(min(y1, y2)))
            w = int(round(abs(x2 - x1)))
            h = int(round(abs(y2 - y1)))

            # Clamp to image bounds
            pix = QPixmap(self._get_display_image_path(self.current_img))
            if not pix.isNull():
                x = max(0, min(x, pix.width() - 1))
                y = max(0, min(y, pix.height() - 1))
                w = max(1, min(w, pix.width() - x))
                h = max(1, min(h, pix.height() - y))

            if w < 5 or h < 5:
                # Too small -> clear
                self.roi_by_image.pop(self.current_img, None)
            else:
                self.push_undo_current_slide()
                self.roi_by_image[self.current_img] = [x, y, w, h]
        except Exception:
            pass
        finally:
            self._roi_drag_start = None
            self.set_roi_select_mode(False)
            self._draw_roi_overlay()

    def _get_current_roi(self):
        if not self.current_img:
            return None
        roi = (getattr(self, "roi_by_image", {}) or {}).get(self.current_img)
        if not (isinstance(roi, (list, tuple)) and len(roi) == 4):
            return None
        try:
            x, y, w, h = [int(v) for v in roi]
        except Exception:
            return None
        if w <= 0 or h <= 0:
            return None
        return [x, y, w, h]

    def canvas_mouse_press(self, event):
        """画布鼠标点击事件（用于吸管取色）"""
        if self.eyedropper_mode and self.current_img is not None:
            # 获取点击位置
            pos = self.view.mapToScene(event.pos())
            x, y = int(pos.x()), int(pos.y())

            # 从当前图片获取颜色
            try:
                # Sample from what user currently sees (original vs inpainted preview).
                img_path = self._get_display_image_path(self.current_img)
                img = _imread_any(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    if 0 <= x < w and 0 <= y < h:
                        # OpenCV使用BGR格式
                        b, g, r = img[y, x]
                        picked_color = QColor(int(r), int(g), int(b))

                        # 判断是为选中框吸取颜色还是全局颜色
                        if hasattr(self, 'picking_for_selected') and self.picking_for_selected:
                            # 为选中的文本框设置颜色
                            if self.selected_box and isinstance(self.selected_box, CanvasTextBox):
                                self.push_undo_current_slide()
                                self.selected_box.custom_bg_color = picked_color
                                self.selected_box.use_custom_bg = True
                                self.selected_box._sync_model_bg()
                                self.chk_custom_bg.setChecked(True)
                                self.update_custom_color_preview(picked_color)
                                self.selected_box.update_background()
                                logger.debug(f"为选中框吸管取色: RGB({r}, {g}, {b})")
                            self.picking_for_selected = False
                        else:
                            # 全局颜色
                            self.text_bg_color = picked_color
                            self.update_color_preview()
                            logger.debug(f"全局吸管取色: RGB({r}, {g}, {b})")

                            # 自动启用背景色
                            if not self.use_text_bg:
                                self.chk_text_bg.setChecked(True)

                            # 更新所有文本框
                            self.update_all_text_boxes_background()

                        # 退出吸管模式（全局/单框都要退出）
                        try:
                            if hasattr(self, "btn_eyedropper"):
                                self.btn_eyedropper.blockSignals(True)
                                self.btn_eyedropper.setChecked(False)
                                self.btn_eyedropper.blockSignals(False)
                            if hasattr(self, "btn_pick_custom_color"):
                                self.btn_pick_custom_color.blockSignals(True)
                                self.btn_pick_custom_color.setChecked(False)
                                self.btn_pick_custom_color.blockSignals(False)
                        except Exception:
                            pass
                        self.picking_for_selected = False
                        self.eyedropper_mode = False
                        self.view.setCursor(Qt.ArrowCursor)
            except Exception as e:
                logger.warning(f"吸管取色失败: {e}")

    def _build_preview_ppt_path(self) -> str:
        import tempfile
        import time

        base_dir = getattr(self, "slide_assets_dir", None) or tempfile.gettempdir()
        os.makedirs(base_dir, exist_ok=True)
        return os.path.abspath(
            build_asset_path(
                base_dir,
                "preview",
                f"preview_{int(time.time() * 1000)}",
                ext=".pptx",
            )
        )

    def _wait_until_file_ready(self, path: str, timeout_sec: float = 4.0) -> None:
        import time

        last_size = -1
        stable = 0
        checks = max(1, int(timeout_sec / 0.05))
        for _ in range(checks):
            try:
                if os.path.exists(path):
                    sz = os.path.getsize(path)
                    if sz > 0 and sz == last_size:
                        stable += 1
                    else:
                        stable = 0
                        last_size = sz
                    if stable >= 3:
                        with open(path, "rb") as _f:
                            _f.read(1)
                        return
            except Exception:
                pass
            time.sleep(0.05)

    def _open_path_with_default_app(self, path: str) -> bool:
        import subprocess

        system = platform.system()
        if system == "Windows":
            for _ in range(3):
                try:
                    os.startfile(path)
                    return True
                except FileNotFoundError:
                    import time

                    time.sleep(0.1)
                except Exception:
                    return False
            return False
        if system == "Darwin":
            return subprocess.run(["open", path], check=False).returncode == 0
        return subprocess.run(["xdg-open", path], check=False).returncode == 0

    def preview_ppt(self):
        """预览导出PPT"""
        if not self.images:
            QMessageBox.warning(self, self._t("提示", "Info"), self._t("请先导入图片", "Please import images first."))
            return

        if not any(self.box_data.values()):
            QMessageBox.warning(self, self._t("提示", "Info"), self._t("请先运行OCR识别", "Please run OCR first."))
            return

        try:
            import time
            temp_path = self._build_preview_ppt_path()

            if self._export_ppt_to_path(temp_path):
                # 记录创建时间，避免被过早清理导致“文件不存在”
                self._temp_preview_ppts[temp_path] = time.time()
                self._wait_until_file_ready(temp_path)
                opened = self._open_path_with_default_app(temp_path)

                if opened:
                    QMessageBox.information(
                        self,
                        self._t("成功", "Success"),
                        self._t(
                            "PPT预览已打开\n\n注意：这是临时文件，本程序退出时会尝试删除（若被 Office 占用可能无法立即删除）",
                            "Preview opened.\n\nNote: this is a temporary file. The app will try to delete it on exit (it may fail if Office is still using it).",
                        ),
                    )
                else:
                    QMessageBox.warning(
                        self,
                        self._t("提示", "Info"),
                        self._t(
                            f"PPT 已生成，但打开失败：{temp_path}\n你可以手动打开该文件。",
                            f"PPT was generated but failed to open: {temp_path}\nYou can open it manually.",
                        ),
                    )
            else:
                QMessageBox.warning(self, self._t("失败", "Failed"), self._t("PPT预览生成失败", "Failed to generate preview PPT."))

        except Exception as e:
            QMessageBox.critical(self, self._t("错误", "Error"), self._t(f"预览失败: {str(e)}", f"Preview failed: {str(e)}"))

    def update_status(self):
        cnt = len(self.images); row = (self.list_thumb.currentRow() + 1) if cnt > 0 else 0
        self.lbl_page.setText(self._t(f"幻灯片 {row} / {cnt}", f"Slide {row} / {cnt}"))

    def _cleanup_preview_ppts(self):
        """尝试删除已经不再被 Office 占用的预览临时 PPT 文件"""
        import time
        # 先保留一段时间，避免 Office 还没来得及打开就被删掉（会导致“文件不存在”）
        MIN_AGE_SEC = PREVIEW_FILE_MIN_AGE_SEC

        items = getattr(self, "_temp_preview_ppts", {}) or {}
        if not isinstance(items, dict):
            # 兼容旧数据结构
            items = {p: 0 for p in list(items)}

        keep = {}
        now = time.time()
        for p, ts in list(items.items()):
            try:
                ts = float(ts or 0)
            except Exception:
                ts = 0
            age = now - ts if ts else 0

            # 太新：不删
            if ts and age < MIN_AGE_SEC:
                keep[p] = ts
                continue

            try:
                if p and os.path.exists(p):
                    os.remove(p)
                    continue
            except Exception:
                # 仍被占用：保留，等下次再试
                keep[p] = ts or now

        self._temp_preview_ppts = keep

    def closeEvent(self, event):
        # 停止正在运行的线程，避免访问已销毁的对象
        for th_name in ("ocr_thread", "inpaint_thread"):
            th = getattr(self, th_name, None)
            if th is not None and th.isRunning():
                logger.debug(f"正在停止线程: {th_name}")
                th.requestInterruption()
                if not th.wait(3000):  # 等待最多3秒
                    logger.warning(f"线程 {th_name} 未能在3秒内停止")

        # 尽量清理预览产生的临时文件（无法保证在 Office 仍占用时删除成功）
        try:
            self._cleanup_preview_ppts()
        except Exception:
            pass
        try:
            if hasattr(self, "_preview_cleanup_timer") and self._preview_cleanup_timer:
                self._preview_cleanup_timer.stop()
        except Exception:
            pass
        try:
            if hasattr(self, "slide_assets_dir") and self.slide_assets_dir and os.path.exists(self.slide_assets_dir):
                for name in os.listdir(self.slide_assets_dir):
                    try:
                        os.remove(os.path.join(self.slide_assets_dir, name))
                    except Exception:
                        pass
                try:
                    os.rmdir(self.slide_assets_dir)
                except Exception:
                    pass
        except Exception:
            pass
        # 清理 OCR 缩放图片临时目录
        try:
            if getattr(self, "temp_dir", None) and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir, ignore_errors=True)
        except Exception:
            pass
        # 清理本次运行缓存目录（默认在项目目录 _runtime_cache 下；也可能回退到系统 temp）
        try:
            run_dir = getattr(self, "run_cache_dir", None)
            if run_dir and os.path.exists(run_dir):
                shutil.rmtree(run_dir, ignore_errors=True)
                # 如果父目录是 _runtime_cache 且为空，顺便删掉，保持项目干净
                parent = os.path.dirname(run_dir)
                if parent and os.path.basename(parent) == "_runtime_cache":
                    try:
                        if os.path.isdir(parent) and not os.listdir(parent):
                            os.rmdir(parent)
                    except Exception:
                        pass
        except Exception:
            pass
        super().closeEvent(event)

if __name__ == "__main__":
    # 配置日志格式
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

    app = QApplication(sys.argv)

    # 检查依赖
    if _missing_qtawesome:
        QMessageBox.critical(
            None,
            _t_sys("缺少库", "Missing dependency"),
            _t_sys("请运行 pip install qtawesome", "Please run: pip install qtawesome"),
        )
        sys.exit(1)

    win = PPTCloneApp()
    win.show()
    sys.exit(app.exec())
