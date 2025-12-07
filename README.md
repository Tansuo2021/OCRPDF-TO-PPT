# PPT编辑器专业版

一个基于OCR技术的智能PPT编辑工具，可以将图片/PDF快速转换为可编辑的PowerPoint文档。

## ✨ 主要功能

- 📄 **多格式导入**：支持图片（JPG/PNG）、PDF批量导入
- 🔍 **智能OCR识别**：基于PaddleOCR自动检测和识别文字区域
- ⚡ **GPU加速**：支持NVIDIA GPU加速，识别速度提升5-10倍
- ✏️ **文本编辑**：可视化编辑文本内容、字体、样式、对齐方式
- 📐 **对齐工具**：多框对齐、等间距分布、尺寸统一等批量操作
- 🎯 **多选功能**：支持框选、Ctrl+点击、全选等多种选择方式
- 💾 **项目保存**：保存/加载项目，支持自动保存
- 📤 **多格式导出**：导出为PPT、PDF、图片序列

## 🚀 快速开始

### 环境要求

- Python 3.8 - 3.12（推荐3.11）
- Windows/Linux/macOS

### 安装依赖

```bash
# 基础版本（CPU）
pip install -r requirements.txt

# GPU加速版本（需要NVIDIA显卡）
pip uninstall paddlepaddle
pip install paddlepaddle-gpu
```

### 运行程序

```bash
python modern_ppt_editor_full_enhanced.py
```

## 📦 主要依赖

- **PaddleOCR**：文字识别引擎
- **python-pptx**：PPT生成
- **Pillow**：图像处理
- **PyMuPDF**：PDF导入支持
- **tkinter**：图形界面

## 🎯 使用流程

1. **导入**：点击"导入图片"或"导入PDF"加载文件
2. **检测**：点击"检测"自动识别文字区域
3. **识别**：点击"识别"提取文字内容
4. **编辑**：调整文本框位置、大小、字体样式
5. **导出**：点击"生成PPT"导出最终文档

## 🔧 GPU加速设置

如果你有NVIDIA显卡，可以开启GPU加速：

1. 安装GPU版本PaddlePaddle（见上方安装说明）
2. 打开程序，点击右上角 ⚙ 设置
3. 选择"GPU - 速度快，需要NVIDIA显卡"
4. 点击"保存并加载OCR"

详细说明见 [GPU加速使用说明.md](GPU加速使用说明.md)

## 📚 功能文档

- [GPU加速使用说明](GPU加速使用说明.md)
- [框选多选功能说明](框选多选功能说明.md)

## 🛠️ 开发工具

项目包含两个诊断脚本：

```bash
# 检查Python版本兼容性
python check_python_version.py

# 检查GPU环境
python check_gpu.py
```

## ⚙️ 配置文件

程序配置保存在 `ppt_editor_config.json`：

```json
{
  "model_dir": "模型路径",
  "ocr_device": "cpu",  // 或 "gpu"
  "autosave_enabled": true,
  "autosave_interval": 300
}
```

## 📸 功能特色

### 批量操作
- 多页同时检测/识别
- 批量字号调整
- 批量样式应用

### 对齐工具
- 左/右/上/下对齐
- 水平/垂直居中
- 等间距分布
- 尺寸统一
- 对齐到画布中心

### 编辑功能
- 画框模式/选择模式切换
- 拖拽调整位置和大小
- 双击快速编辑
- 撤销/重做（50步）
- 复制/粘贴
- 方向键微调位置

## 🐛 常见问题

**Q: GPU模式启动报错？**
A: 检查是否安装了 `paddlepaddle-gpu`，Python版本是否为3.8-3.12

**Q: 识别不准确？**
A: 尝试调整图片分辨率，确保文字清晰；或手动调整文本框范围

**Q: 中文路径支持？**
A: 完全支持中文路径，无需担心

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## ⭐ 支持项目

如果这个项目对你有帮助，欢迎给个Star！
