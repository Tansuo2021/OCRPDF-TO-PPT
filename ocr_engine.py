"""
OCR引擎 - 简化版，完全兼容 PaddleOCR 2.x 和 3.x
参考优秀项目实现
"""
import os
import warnings
import tempfile
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import cv2
import numpy as np

# 静默 Paddle 输出
os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("FLAGS_minloglevel", "2")

# Some PaddleOCR 3.x models use PIR. On Windows/CPU, enabling OneDNN can hit a
# PaddlePaddle limitation and crash with:
# NotImplementedError: ConvertPirAttribute2RuntimeAttribute not support [...]
os.environ.setdefault("FLAGS_use_mkldnn", "0")
os.environ.setdefault("FLAGS_use_onednn", "0")
warnings.filterwarnings("ignore")

try:
    from paddleocr import PaddleOCR
    HAS_PADDLEOCR = True
except:
    HAS_PADDLEOCR = False
    PaddleOCR = None


def get_paddleocr_version() -> Optional[int]:
    """获取 PaddleOCR 主版本号"""
    try:
        import paddleocr
        v = getattr(paddleocr, "__version__", "")
        if v:
            return int(str(v).split(".", 1)[0])
    except:
        pass
    return None


def check_gpu_available() -> bool:
    """检查GPU是否可用"""
    try:
        import paddle
        if paddle.is_compiled_with_cuda():
            return paddle.device.cuda.device_count() > 0
    except:
        pass
    return False


class OCREngine:
    """OCR引擎"""

    def __init__(self, use_gpu=False, model_det_dir=None, model_rec_dir=None):
        """
        初始化OCR引擎

        Args:
            use_gpu: 是否使用GPU
            model_det_dir: 检测模型路径（可选）
            model_rec_dir: 识别模型路径（可选）
        """
        self.ocr = None
        self.use_gpu = use_gpu
        self.model_det_dir = model_det_dir
        self.model_rec_dir = model_rec_dir
        self.version = 3  # 默认版本

        if not HAS_PADDLEOCR:
            raise RuntimeError("PaddleOCR 未安装。请运行: pip install paddleocr")

        self._init_ocr()

    def _init_ocr(self):
        """初始化 PaddleOCR"""
        print("正在初始化 OCR 引擎...")

        device = "gpu" if self.use_gpu and check_gpu_available() else "cpu"
        self.version = get_paddleocr_version() or 3

        print(f"- PaddleOCR 版本: {self.version}.x")
        print(f"- 设备: {device.upper()}")

        params_list = []

        # PaddleOCR 3.x
        if self.version >= 3:
            # 优先使用自定义模型
            if self.model_det_dir and self.model_rec_dir:
                if os.path.exists(self.model_det_dir) and os.path.exists(self.model_rec_dir):
                    print(f"- 检测模型: {self.model_det_dir}")
                    print(f"- 识别模型: {self.model_rec_dir}")
                    params_list.append({
                        "text_detection_model_dir": self.model_det_dir,
                        "text_recognition_model_dir": self.model_rec_dir,
                        "device": device,
                        "use_doc_orientation_classify": False,
                        "use_doc_unwarping": False,
                        "use_textline_orientation": False,
                    })

            # 回退到默认模型
            params_list.append({
                "lang": "ch",
                "device": device,
                "use_doc_orientation_classify": False,
                "use_doc_unwarping": False,
                "use_textline_orientation": False,
            })

        # PaddleOCR 2.x
        else:
            params_list.append({
                "lang": "ch",
                "use_gpu": (device == "gpu"),
                "show_log": False,
            })

        # 尝试初始化
        last_error = None
        for params in params_list:
            try:
                self.ocr = PaddleOCR(**params)
                print("[OK] OCR 引擎初始化成功！\n")
                return
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"OCR 初始化失败: {last_error}")

    def recognize(self, image_path: str) -> List[Dict]:
        """
        识别图片中的文字

        Args:
            image_path: 图片路径

        Returns:
            识别结果列表，每项包含:
            - bbox: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            - text: 文字内容
            - confidence: 置信度
            - rect: (x, y, w, h) 矩形框
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片不存在: {image_path}")

        print(f"识别图片: {os.path.basename(image_path)}")

        # PaddleOCR 3.x 使用 predict()，2.x 使用 ocr()
        if self.version >= 3:
            result = self.ocr.predict(image_path)
        else:
            result = self.ocr.ocr(image_path, cls=False)

        # 解析结果
        text_boxes = []

        # PaddleOCR 3.x 返回字典格式
        if self.version >= 3:
            if not result or len(result) == 0:
                print("未检测到文字")
                return []

            page_result = result[0]
            dt_polys = page_result.get("dt_polys", [])
            rec_texts = page_result.get("rec_texts", [])
            rec_scores = page_result.get("rec_scores", [])

            for idx, (poly, text) in enumerate(zip(dt_polys, rec_texts)):
                try:
                    # poly 是四点坐标列表
                    points = np.array(poly)
                    x = int(np.min(points[:, 0]))
                    y = int(np.min(points[:, 1]))
                    w = int(np.max(points[:, 0]) - x)
                    h = int(np.max(points[:, 1]) - y)

                    confidence = rec_scores[idx] if idx < len(rec_scores) else 1.0

                    text_boxes.append({
                        'bbox': poly,
                        'text': text,
                        'confidence': float(confidence),
                        'rect': (x, y, w, h)
                    })

                    print(f"  [{idx+1}] {text} ({confidence:.2f})")

                except Exception as e:
                    print(f"  [!] 解析第 {idx+1} 个文本框失败: {e}")
                    continue

        # PaddleOCR 2.x 返回列表格式
        else:
            if not result or not result[0]:
                print("未检测到文字")
                return []

            for idx, line in enumerate(result[0]):
                try:
                    bbox = line[0]  # 四点坐标
                    text_info = line[1]  # (文本, 置信度)

                    text = str(text_info[0])
                    confidence = float(text_info[1])

                    # 计算矩形框
                    points = np.array(bbox)
                    x = int(np.min(points[:, 0]))
                    y = int(np.min(points[:, 1]))
                    w = int(np.max(points[:, 0]) - x)
                    h = int(np.max(points[:, 1]) - y)

                    text_boxes.append({
                        'bbox': bbox,
                        'text': text,
                        'confidence': confidence,
                        'rect': (x, y, w, h)
                    })

                    print(f"  [{idx+1}] {text} ({confidence:.2f})")

                except Exception as e:
                    print(f"  [!] 解析第 {idx+1} 个文本框失败: {e}")
                    continue

        print(f"[OK] 识别完成，共 {len(text_boxes)} 个文本框\n")
        return text_boxes


# 测试代码
if __name__ == "__main__":
    # 测试OCR引擎
    engine = OCREngine(use_gpu=False)

    # 创建测试图片
    test_img = np.ones((200, 400, 3), dtype=np.uint8) * 255
    cv2.putText(test_img, "Test OCR", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    test_path = "test_ocr.png"
    cv2.imwrite(test_path, test_img)

    # 识别
    results = engine.recognize(test_path)
    print(f"识别结果: {results}")

    # 清理
    if os.path.exists(test_path):
        os.remove(test_path)
