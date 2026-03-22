import hashlib
import os
import re


def _load_cv2_numpy():
    import cv2
    import numpy as np

    return cv2, np


def sanitize_stem(path_or_name: str, default: str = "file") -> str:
    stem = os.path.splitext(os.path.basename(str(path_or_name or "")))[0].strip()
    if not stem:
        stem = default
    stem = re.sub(r"[^\w\-.]", "_", stem).strip("._")
    return stem[:80] or default


def path_token(path: str, length: int = 12) -> str:
    key = os.path.abspath(os.path.expanduser(str(path or "")))
    digest = hashlib.sha1(key.encode("utf-8", "ignore")).hexdigest()
    return digest[: max(4, int(length or 12))]


def build_asset_path(
    directory: str,
    prefix: str,
    source_path: str,
    suffix: str = "",
    ext: str = ".png",
) -> str:
    ext = str(ext or ".png").strip() or ".png"
    if not ext.startswith("."):
        ext = f".{ext}"
    stem = sanitize_stem(source_path, default=str(prefix or "file"))
    token = path_token(source_path)
    suffix_part = f"_{suffix}" if suffix else ""
    filename = f"{prefix}_{stem}_{token}{suffix_part}{ext}"
    return os.path.join(str(directory or ""), filename)


def imread_any(path: str, flags=None):
    cv2, np = _load_cv2_numpy()
    if flags is None:
        flags = cv2.IMREAD_COLOR
    try:
        p = str(path or "")
        if not p:
            return None
        try:
            data = np.fromfile(p, dtype=np.uint8)
            if data is not None and data.size > 0:
                img = cv2.imdecode(data, int(flags))
                if img is not None:
                    return img
        except Exception:
            pass
        return cv2.imread(p, flags)
    except Exception:
        return None


def imwrite_any(path: str, image, params=None) -> bool:
    cv2, _ = _load_cv2_numpy()
    try:
        p = str(path or "")
        if not p:
            return False
        parent = os.path.dirname(p)
        if parent:
            os.makedirs(parent, exist_ok=True)
        ext = os.path.splitext(p)[1].lower() or ".png"
        encode_ext = ext if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"} else ".png"
        encode_params = list(params or [])
        ok, buf = cv2.imencode(encode_ext, image, encode_params)
        if ok:
            buf.tofile(p)
            return True
    except Exception:
        pass

    try:
        return bool(cv2.imwrite(str(path or ""), image, list(params or [])))
    except Exception:
        return False
