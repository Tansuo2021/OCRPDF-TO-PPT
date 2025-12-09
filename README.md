# AI PPT Restorer | AI生成PPT图片还原工具

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey.svg)

**将AI生成的精美PPT图片还原为可编辑的PPT文档**

[English](#english) | [中文文档](#chinese)

</div>

---

<a name="chinese"></a>

## 📖 项目简介

使用Google Nano Banana Pro等AI工具生成的PPT虽然视觉效果惊艳，但输出的是**不可编辑的图片格式**。当需要修改文字内容时，只能重新生成，极其不便。

**AI PPT Restorer** 是一个开源工具，通过 **OCR文字识别 + AI智能修复** 技术，将AI生成的PPT图片还原为完全可编辑的PPT文档，实现：

✅ 自动识别图片中的文字位置和内容
✅ AI智能去除文字区域，生成干净背景
✅ 还原为可编辑的文本框图层
✅ 支持手动涂抹补充和迭代修复
✅ 全局撤销/重做，操作零风险

---

## 🎯 核心功能

### 1️⃣ OCR智能识别
- 基于PaddleOCR自动检测文字位置
- 识别文字内容和字体大小
- 支持单页和批量处理
- 支持GPU加速（速度提升5-10倍）

### 2️⃣ AI背景生成
- 集成IOPaint API智能修复
- 自动去除文字区域
- 生成无痕干净背景
- 支持本地/云端API部署

### 3️⃣ 自定义涂抹工具
- 🖌️ 笔刷工具：涂抹式标记
- ⬜ 框选工具：矩形框选
- 可调节笔刷大小（5-100px）
- 实时视觉反馈

### 4️⃣ 迭代修复模式 ⭐
- 在已生成的背景上继续编辑
- 智能检测背景图自动切换模式
- 多次迭代直到完美

### 5️⃣ 全局撤销/重做 ⭐
- 支持文本框编辑撤销
- 支持涂抹操作撤销
- 支持背景生成撤销
- 可撤销到原图状态
- 快捷键：`Ctrl+Z` / `Ctrl+Y`
- 最多50步历史记录

### 6️⃣ 完整编辑功能
- 文本框拖拽、缩放、旋转
- 字体、字号、颜色调整
- 对齐和分布工具
- 批量操作和复制粘贴

### 7️⃣ 多格式支持
- 导入：PNG、JPG、PDF
- 导出：PPTX、PDF

---

## 🚀 快速开始

### 环境要求

- Python 3.8+
- Windows / Linux / macOS
- IOPaint服务（用于AI背景生成）

### 安装步骤

#### 1. 克隆项目

```bash
git clone https://github.com/Tansuo2021/OCRPDF-TO-PPT.git
cd OCRPDF-TO-PPT
```

#### 2. 安装依赖

```bash
pip install -r requirements.txt
```

依赖包括：
- `python-pptx` - PPT生成
- `Pillow` - 图像处理
- `opencv-python` - 图像操作
- `numpy` - 数值计算
- `paddleocr` - OCR识别
- `paddlepaddle` - 深度学习框架
- `requests` - API调用
- `PyMuPDF` - PDF处理

#### 3. 安装并启动IOPaint服务 参考如下

https://www.iopaint.com/install/windows_1click_installer

> 💡 **提示**：IOPaint支持CPU和GPU模式，GPU可大幅提升处理速度

#### 4. 运行程序

```bash
python modern_ppt_editor_full_enhanced.py
```

---

## 📚 使用教程

### 方式一：OCR自动识别 + 背景生成

**适用场景**：标准文档、文字清晰的PPT图片

```
1. 点击"导入图片"或"导入PDF"
2. 点击"检测 - 当前页"（OCR自动识别）
3. 点击"背景生成 - 当前页"
4. 等待5-30秒
5. 切换到"PPT效果"查看结果
6. 编辑文本框内容、位置、样式
7. 点击"生成PPT"导出
```

### 方式二：手动涂抹 + 背景生成

**适用场景**：水印去除、Logo去除、OCR识别不全

```
1. 导入图片
2. 点击"进入涂抹"
3. 选择工具：
   - ✏️ 笔刷：涂抹不规则区域
   - ⬜ 框选：框选矩形区域
4. 标记需要去除的区域
5. 点击"🎨生成背景"
6. 等待完成
7. 编辑文本框并导出
```

### 方式三：组合使用（推荐）⭐

**适用场景**：复杂文档、需要精确控制

```
1. 先用OCR自动识别（快速覆盖大部分文字）
2. 进入涂抹模式补充遗漏区域
3. 生成背景（OCR框 + 涂抹区域会叠加处理）
4. 如效果不满意，再次进入涂抹模式（迭代修复）
5. 编辑完成后导出
```

---

## 🎨 功能演示

### OCR识别 + 背景生成

```
原图（AI生成的PPT图片）
    ↓ OCR检测
识别出文字框
    ↓ 生成背景
干净背景 + 可编辑文本框
    ↓ 编辑导出
可编辑的PPTX文件
```

### 手动涂抹工具

| 工具 | 使用方式 | 适用场景 |
|------|----------|----------|
| 🖌️ 笔刷 | 按住拖动涂抹 | 不规则区域、精细操作 |
| ⬜ 框选 | 拉框选择矩形 | 规则文本框、快速标记 |

### 迭代修复流程

```
原图 → 背景图1（OCR生成）
         ↓
    发现遗漏区域
         ↓
    进入涂抹模式（自动基于背景图1）
         ↓
    涂抹遗漏 → 生成背景图2（迭代修复）
         ↓
    继续修复 → 背景图3...
```

### 全局撤销/重做

```
操作历史：
原图 → OCR检测 → 生成背景1 → 涂抹标记 → 生成背景2

Ctrl+Z 撤销：
背景2 → 背景1（含涂抹标记）→ 背景1（清空涂抹）→ 原图

Ctrl+Y 重做：
原图 → 背景1 → 涂抹标记 → 背景2
```

---

## ⚙️ 配置说明

### IOPaint API配置

首次使用需配置API地址：

1. 点击"⚙️设置"按钮
2. 找到"IOPaint API配置"
3. 输入API地址（默认：`http://127.0.0.1:8080/api/v1/inpaint`）
4. 点击"测试连接"验证
5. 保存配置

### 配置文件

配置保存在 `ppt_editor_config.json`：

```json
{
  "model_dir": ".paddlex/official_models",
  "inpaint_api_url": "http://127.0.0.1:8080/api/v1/inpaint",
  "inpaint_enabled": true,
  "ocr_device": "cpu"
}
```

### GPU加速设置

如果你有NVIDIA显卡，可以开启GPU加速：

1. 安装GPU版本PaddlePaddle：
   ```bash
   pip uninstall paddlepaddle
   pip install paddlepaddle-gpu
   ```
2. 打开程序，点击右上角 ⚙ 设置
3. 选择"GPU - 速度快，需要NVIDIA显卡"
4. 点击"保存并加载OCR"

详细说明见 [GPU加速使用说明.md](GPU加速使用说明.md)

### 高级参数调整

修改 `call_inpaint_api()` 方法中的参数（约第5101行）：

- `ldm_steps`: 修复步数（20-50，默认30）
- `hd_strategy`: 高清策略（"Original" / "Resize" / "Crop"）
- `crop_padding`: 裁切边距（默认128像素）

---

## 🛠️ 技术架构

### 核心技术栈

| 技术 | 用途 |
|------|------|
| **PaddleOCR** | 文字检测与识别 |
| **IOPaint** | AI图像修复 |
| **PIL/Pillow** | 图像处理 |
| **python-pptx** | PPT生成 |
| **PyMuPDF** | PDF处理 |
| **Tkinter** | GUI界面 |

### 工作原理

#### 1. OCR文字识别

```python
# 使用PaddleOCR检测文字位置
result = ocr.ocr(image, cls=True)
# 提取文字框坐标和内容
text_boxes = [{"position": box, "text": text}]
```

#### 2. 蒙版创建

```python
# 创建L模式图像（黑白蒙版）
mask = Image.new("L", image.size, 0)
# 白色=需要修复的区域，黑色=保留区域
draw.rectangle(box, fill=255)
```

#### 3. AI智能修复

```python
# 智能裁切（仅处理有蒙版的区域+padding）
crop_region = get_mask_bounds(mask)
# Base64编码发送到IOPaint API
payload = {"image": b64_image, "mask": b64_mask}
response = requests.post(api_url, json=payload)
# 高斯模糊边缘融合
blended = Image.composite(repaired, original, blur_mask)
```

#### 4. 历史记录系统

```python
history = [
    {"type": "textboxes", "data": {...}},
    {"type": "background", "data": {"old_bg_path": ..., "new_bg_path": ...}},
    {"type": "inpaint_stroke", "data": {"stroke": ..., "mask_state": ...}}
]
```

---

## 📁 项目结构

```
ai-ppt-restorer/
├── modern_ppt_editor_full_enhanced.py  # 主程序（5500+行）
├── requirements.txt                     # 依赖列表
├── README.md                            # 本文档
├── docs/                                # 详细文档
│   ├── 全局撤销重做使用说明.txt
│   ├── 迭代修复使用说明.txt
│   ├── 自定义涂抹功能使用说明.txt
│   ├── IOPaint_使用说明.md
│   ├── 快速启动指南.txt
│   ├── 功能更新说明.txt
│   ├── GPU加速使用说明.md
│   └── 框选多选功能说明.md
├── check_gpu.py                         # GPU环境检查脚本
├── check_python_version.py              # Python版本检查脚本
├── temp_backgrounds/                    # 生成的背景图（自动创建）
└── ppt_editor_config.json              # 配置文件（自动生成）
```

---

## 🎯 使用场景

### 1. AI生成PPT后续编辑 ⭐
- Google Nano Banana Pro生成的PPT图片
- 需要修改文字内容但不想重新生成
- 保留精美设计的同时获得编辑能力

### 2. 素材去水印
- 去除图片中的水印
- 去除Logo和品牌标识
- 清理不必要的文字内容

### 3. PPT模板提取
- 从截图中提取可编辑的PPT
- 将图片格式的模板转换为PPTX
- 批量处理多页文档

### 4. 自媒体原创素材
- 去除版权文字
- 生成原创图片素材
- 批量处理减少侵权风险

---

## 📊 性能指标

| 指标 | 数值 |
|------|------|
| OCR识别速度 | 1-3秒/页（CPU），0.5-1秒/页（GPU） |
| 背景生成速度 | 5-30秒/页（取决于图片大小） |
| 支持图片尺寸 | 最大4000×4000像素 |
| 历史记录上限 | 50步 |
| 批量处理能力 | 无限制（逐页处理） |

---

## 🔧 常见问题

### Q1: IOPaint连接失败怎么办？

**A**: 检查以下几点：
1. IOPaint服务是否启动：`iopaint start --host 127.0.0.1 --port 8080`
2. 端口是否被占用：换一个端口试试
3. 防火墙是否拦截：临时关闭防火墙测试
4. API地址是否正确：在设置中点击"测试连接"

### Q2: OCR识别不准确？

**A**:
- 确保图片清晰，分辨率足够
- 文字颜色与背景对比度要高
- 可以手动进入涂抹模式补充遗漏区域
- 尝试开启GPU加速提高识别准确率

### Q3: 生成的背景有痕迹？

**A**:
- 扩大涂抹范围（稍微超出文字边缘5-10像素）
- 增加 `ldm_steps` 参数到40-50（提高修复质量）
- 使用迭代修复功能多次优化
- 确保IOPaint使用GPU模式

### Q4: 处理速度太慢？

**A**:
- 使用GPU版本的IOPaint（速度提升5-10倍）
- 缩小图片尺寸（如3000px→1500px）
- 使用框选工具代替笔刷（处理更快）
- 减少 `ldm_steps` 参数到20-25

### Q5: 切换页面后涂抹消失了？

**A**:
- 每页的涂抹蒙版是独立的
- 切换页面前请先生成背景
- 生成后涂抹会自动清空
- 可以使用全局撤销（Ctrl+Z）恢复

### Q6: GPU模式启动报错？

**A**:
- 检查是否安装了 `paddlepaddle-gpu`
- Python版本是否为3.8-3.12
- 检查CUDA和cuDNN版本是否兼容
- 运行 `python check_gpu.py` 诊断问题

### Q7: 能否撤销背景生成操作？

**A**:
- 可以！按 `Ctrl+Z` 撤销背景生成
- 可以撤销到原图状态
- 支持重做（`Ctrl+Y`）
- 最多支持50步历史记录

### Q8: 中文路径支持吗？

**A**:
- 完全支持中文路径
- 无需担心路径编码问题

---

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

### 开发环境搭建

```bash
# 克隆项目
git clone https://github.com/Tansuo2021/OCRPDF-TO-PPT.git
cd OCRPDF-TO-PPT

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装开发依赖
pip install -r requirements.txt
```

### 提交规范

- 功能开发：`feat: 添加XXX功能`
- Bug修复：`fix: 修复XXX问题`
- 文档更新：`docs: 更新XXX文档`
- 性能优化：`perf: 优化XXX性能`

---

## 📄 开源协议

本项目采用 [MIT License](LICENSE) 开源协议。

你可以自由地：
- ✅ 商业使用
- ✅ 修改源代码
- ✅ 分发和再授权
- ✅ 私人使用

但需要：
- 📋 保留版权声明
- 📋 保留许可证声明

---

## 🙏 致谢

感谢以下开源项目：

- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) - 强大的OCR识别引擎
- [IOPaint](https://github.com/Sanster/IOPaint) - 优秀的AI图像修复工具
- [python-pptx](https://github.com/scanny/python-pptx) - Python PPT生成库
- [PyMuPDF](https://github.com/pymupdf/PyMuPDF) - PDF处理工具

---

## 📮 联系方式

- Issue: [GitHub Issues](https://github.com/Tansuo2021/OCRPDF-TO-PPT/issues)
- Email: your.email@example.com

---

## 🌟 Star History

如果这个项目对你有帮助，请给一个⭐️Star支持一下！

[![Star History Chart](https://api.star-history.com/svg?repos=Tansuo2021/OCRPDF-TO-PPT&type=Date)](https://star-history.com/#Tansuo2021/OCRPDF-TO-PPT&Date)

---

<div align="center">

**Made with ❤️ by Tansuo**

[⬆ 回到顶部](#ai-ppt-restorer--ai生成ppt图片还原工具)

</div>

---

<a name="english"></a>

# English Documentation

## 📖 Introduction

AI tools like Google Nano Banana Pro generate visually stunning PPT images, but they output **non-editable image formats**. When you need to modify text, you have to regenerate everything - which is extremely inconvenient.

**AI PPT Restorer** solves this problem by using **OCR + AI Inpainting** technology to convert AI-generated PPT images into fully editable PowerPoint documents.

## 🚀 Key Features

- ✅ **OCR Text Recognition** - Auto-detect text position and content with PaddleOCR
- ✅ **AI Background Generation** - Intelligently remove text and generate clean backgrounds via IOPaint
- ✅ **Editable Text Layers** - Restore as fully editable text boxes
- ✅ **Custom Inpainting Tools** - Brush & rectangle selection for manual marking
- ✅ **Iterative Repair** - Edit on existing backgrounds for progressive refinement
- ✅ **Global Undo/Redo** - 50-step history with Ctrl+Z/Y support
- ✅ **Multi-format Support** - Import PNG/JPG/PDF, Export PPTX/PDF
- ✅ **GPU Acceleration** - 5-10x faster with NVIDIA GPU

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/Tansuo2021/OCRPDF-TO-PPT.git
cd OCRPDF-TO-PPT

# Install dependencies
pip install -r requirements.txt

# Install and start IOPaint
pip install iopaint
iopaint start --host 127.0.0.1 --port 8080

# Run application
python modern_ppt_editor_full_enhanced.py
```

## 📚 Quick Start

### Method 1: OCR Auto-Detection

1. Import image/PDF
2. Click "Detect - Current Page"
3. Click "Generate Background - Current Page"
4. Wait 5-30 seconds
5. Switch to "PPT Preview" mode
6. Edit text boxes and export

### Method 2: Manual Inpainting

1. Import image
2. Click "Enter Inpaint Mode"
3. Use brush/rectangle tools to mark regions
4. Click "Generate Background"
5. Edit and export

### Method 3: Combined Approach (Recommended) ⭐

1. Start with OCR auto-detect (covers most text quickly)
2. Enter inpaint mode to supplement missed areas
3. Generate background (combines OCR + manual marks)
4. Use iterative repair if needed
5. Edit and export

## 🛠️ Tech Stack

- **PaddleOCR** - Text detection and recognition
- **IOPaint** - AI-powered image inpainting
- **PIL/Pillow** - Image processing
- **python-pptx** - PowerPoint generation
- **PyMuPDF** - PDF processing
- **Tkinter** - GUI framework
- **NumPy** - Numerical operations
- **OpenCV** - Computer vision

## 🎯 Use Cases

### 1. AI-Generated PPT Editing ⭐
- Convert Google Nano Banana Pro output to editable format
- Modify text without regenerating entire design
- Preserve beautiful AI-generated aesthetics while gaining edit control

### 2. Watermark Removal
- Remove watermarks from images
- Clean up logos and branding
- Eliminate unwanted text overlays

### 3. PPT Template Extraction
- Extract editable PPT from screenshots
- Convert image templates to PPTX format
- Batch process multi-page documents

### 4. Content Creation
- Generate original materials for social media
- Remove copyright text from references
- Reduce infringement risk through batch processing

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| OCR Speed | 1-3s/page (CPU), 0.5-1s/page (GPU) |
| Background Generation | 5-30s/page (depends on image size) |
| Max Image Size | 4000×4000 pixels |
| History Limit | 50 steps |
| Batch Processing | Unlimited (sequential) |

## 🔧 Troubleshooting

### Q: IOPaint connection failed?
**A:** Check if service is running, port is available, firewall settings, and API URL is correct

### Q: OCR accuracy issues?
**A:** Ensure clear images, high contrast, use manual inpaint mode to supplement, or enable GPU acceleration

### Q: Background artifacts?
**A:** Expand mark region by 5-10px, increase `ldm_steps` to 40-50, use iterative repair

### Q: Slow processing?
**A:** Use GPU mode, reduce image size, use rectangle tool instead of brush, decrease `ldm_steps`

### Q: Can I undo background generation?
**A:** Yes! Press `Ctrl+Z` to undo, supports up to 50 steps, can undo to original image

## 📄 License

MIT License - Free for commercial and personal use

## 🙏 Acknowledgments

Thanks to: **PaddleOCR**, **IOPaint**, **python-pptx**, **PyMuPDF**

---

<div align="center">

**If this project helps you, please give it a ⭐️ Star!**

</div>
