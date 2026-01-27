English | [中文](README.zh-CN.md)

# OCRPDF-TO-PPT (PowerOCR Presentation)
<img width="4085" height="2070" alt="Clip_2026-01-27_21-17-54" src="https://github.com/user-attachments/assets/0dbba796-fd55-4128-8087-975e6bb1bfad" />
A lightweight desktop tool that turns images / PDF pages into an **editable PowerPoint (.pptx)** with OCR text boxes.

Core idea:
- Each slide uses the original image (or an inpainted "clean background" image) as the background.
- OCR results are converted to **editable PowerPoint text boxes** positioned on top of the background.

## Features

- Import: images (PNG/JPG/...) and PDFs (render each page to an image)
- OCR: PaddleOCR 2.x / 3.x (auto-detect), CPU/GPU (auto fallback to CPU)
- Batch workflow: thumbnails, add/duplicate/reorder slides, undo/redo
- Edit boxes on canvas: move/resize, edit text, copy/cut/paste, format brush
- ROI (Region of Interest): OCR / inpaint only within a selected rectangle
- Text box background: global and per-box background color + alpha, eyedropper tool
- Export PPTX + Preview (F5)
- Optional: call **IOPaint** API to remove the original text on background before exporting (avoid "double text")

## Requirements

- Python 3.8+ (64-bit recommended)
- OS: Windows is best-tested; macOS/Linux should also work if PaddlePaddle wheels are available

Notes about PaddlePaddle:
- PaddlePaddle wheels availability depends on your OS/Python version.
- For GPU, install the correct PaddlePaddle GPU build according to the official guide.

## Install

1) Clone the repo

```bash
git clone https://github.com/Tansuo2021/OCRPDF-TO-PPT.git
cd OCRPDF-TO-PPT
```

2) Create & activate a virtual environment

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS / Linux:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

3) Install dependencies

```bash
pip install -U pip
pip install -r requirements.txt
pip install qtawesome
```

If you hit `paddlepaddle` install errors, install PaddlePaddle first using the official instructions, then retry:
- PaddlePaddle install guide: https://www.paddlepaddle.org.cn/install/quick

## Run

```bash
python main.py
```

On the first OCR run, PaddleOCR/PaddleX may download models. By default the cache goes to `model/official_models/` (configurable in the app).

## Usage (Step-by-step)

### 1) Import images / PDFs

- Images: click `导入图片` (or press `Ctrl+O`)
- PDFs: click `导入PDF` (or press `Ctrl+Shift+O`)
  - Each PDF page will be rendered into an image and added as a slide thumbnail.

### 2) (Optional) Select ROI (only OCR/Inpaint within a region)

- Click `框选选区` (or press `Ctrl+Alt+A`)
- Drag on the canvas to select a rectangle
- Click `清除选区` (or press `Ctrl+Alt+Shift+A`) to reset

### 3) Run OCR

- Current slide: `OCR本页` (`Ctrl+Enter`)
- All slides: `OCR全部` (`Ctrl+R`)

You can edit OCR results after recognition:
- Drag to move boxes; drag handles to resize
- Double-click a box to edit its text
- `Delete` deletes the selected box
- `Ctrl+C / Ctrl+X / Ctrl+V` copy/cut/paste boxes
- `格式刷` copies style from one box and applies to the next clicked box

### 4) (Recommended) Remove background text with IOPaint (avoid "double text")

Why: scanned images already contain the original text; exporting OCR boxes on top can look like duplicated text.

1) Start IOPaint service (example):
```bash
iopaint start --host 127.0.0.1 --port 8080
```

2) In the app: `设置` tab -> `IOPaint 设置`
- Enable IOPaint
- Set API URL (default): `http://127.0.0.1:8080/api/v1/inpaint`
- Adjust paddings if needed

3) Run inpaint:
- Current slide: `去字本页` (`Ctrl+I`)
- All slides: `去字全部` (`Ctrl+Shift+I`)

Tips:
- `去字预览` toggles original / inpainted background for comparison (`Ctrl+Alt+B`)
- `恢复原图` removes the inpainted variant for the current slide (`Ctrl+Alt+Shift+B`)

### 5) Export / Preview PPT

- Export: `导出PPT` (`Ctrl+S`)
- Preview: `预览PPT` (`F5`)

Export behavior:
- If an inpainted variant exists for a slide, export will prefer it as the background.
- Text boxes remain editable in PowerPoint.

### 6) Export appearance settings (View tab)

In `视图` tab -> `PPT导出设置`:
- Enable/disable text box background
- Pick global background color / alpha
- Use eyedropper to pick a color from the current image
- Per-box background: select a box, then adjust its custom background in the right panel

## Keyboard shortcuts

Inside the app, press `F1` to open the shortcut cheat-sheet.

## Configuration

Settings are stored in `settings.json` (created/updated by the app).

### OCR model cache

In `设置` tab -> `OCR 设置`:
- `模型缓存目录 (PADDLE_PDX_CACHE_HOME)`: where PaddleOCR/PaddleX downloads models (`official_models/` will be created under it)
- Optional: set custom `det/rec` model directories
- Enable GPU (auto fallback if GPU is not available)

## Troubleshooting

- `No module named qtawesome`: run `pip install qtawesome`
- OCR init fails / model download fails:
  - Check network access (first run needs to download models)
  - Change model cache dir in `OCR 设置` to a writable folder
  - For GPU: confirm the correct PaddlePaddle GPU wheel + CUDA runtime
- PDF import fails:
  - Ensure `PyMuPDF` is installed (`pip install pymupdf`)
- IOPaint fails:
  - Make sure the IOPaint service is running
  - Verify API URL in `IOPaint 设置` (default: `http://127.0.0.1:8080/api/v1/inpaint`)

## Project structure

- `main.py`: GUI + workflow (import/OCR/edit/export)
- `ocr_engine.py`: PaddleOCR wrapper (2.x/3.x compatible)
- `ppt_export.py`: PPTX generation with editable text boxes

## Credits

- PaddleOCR / PaddlePaddle
- PySide6 (Qt)
- python-pptx
- PyMuPDF (PDF import)
- IOPaint (optional inpaint backend)
- QtAwesome (icons)

## Star / Watch

If this project helps you, please consider starring or watching it on GitHub.

[![GitHub stars](https://img.shields.io/github/stars/Tansuo2021/OCRPDF-TO-PPT?style=social)](https://github.com/Tansuo2021/OCRPDF-TO-PPT/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/Tansuo2021/OCRPDF-TO-PPT?style=social)](https://github.com/Tansuo2021/OCRPDF-TO-PPT/watchers)

[![Star History Chart](https://api.star-history.com/svg?repos=Tansuo2021/OCRPDF-TO-PPT&type=Date)](https://star-history.com/#Tansuo2021/OCRPDF-TO-PPT&Date)

Image/badge providers:
- https://shields.io/ (badges)
- https://star-history.com/ (star history images)
