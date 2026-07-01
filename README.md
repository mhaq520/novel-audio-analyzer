# 有声小说批量分析系统

基于本地 GPU（RTX 4060 8GB）的有声小说批量分析系统，实现从音频文件到结构化剧情摘要与行为标签的自动化流水线，并提供 Web 图形界面操作。

## 功能特性

- **批量音频处理** — 支持 MP3 / WAV / M4A / FLAC / AAC / OGG 格式，单文件支持 ≥1 小时
- **本地 ASR 转写** — 基于 faster-whisper + CTranslate2，利用 GPU 加速，支持中文语音识别
- **LLM 智能分析** — 调用 Qwen2.5-7B-Instruct（4-bit 量化）自动生成剧情摘要与行为关键词
- **转写缓存** — 以文件名+修改时间为 Key 缓存 ASR 结果，相同文件重复处理时跳过识别(未完成)
- **结果输出** — 每条音频的分析结果保存为独立 `.txt` 文件，统一存放于 `output/` 目录
- **Web GUI** — Flask 提供简洁界面，支持拖拽上传、进度展示、日志输出、结果下载

## 系统架构

```
audio files → [ASR (faster-whisper)] → transcript cache → [LLM (Qwen2.5-7B)] → .txt output
                    │                                                    │
               cache/ (json)                                       output/ (分析.txt)
```

## 环境要求

| 组件 | 要求 |
|------|------|
| 操作系统 | Windows 11 |
| GPU | NVIDIA RTX 4060 8GB（推荐） |
| 显存 | ≥ 6GB 可用 |
| Python | ≥ 3.11 |
| 包管理 | uv（推荐） |

## 快速开始

### 1. 安装依赖

```bash
# 创建虚拟环境并安装依赖
uv sync
```

> 项目已配置清华 PyPI 镜像与 PyTorch CUDA 12.1 索引。

### 2. 下载模型

**ASR 模型（faster-whisper large-v3，CTranslate2 格式）**

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
uv run python -c "from faster_whisper.utils import download_model; download_model('large-v3', output_dir='./models/whisper-ct2')"
```

**LLM 模型（DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored）**

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
huggingface-cli download --resume-download aifeifei798/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored --local-dir ./models/DarkIdol-Llama-3.1-8B-Instruct-1.2-Uncensored
```

### 3. 启动系统

```bash
uv run python app_flask.py
```

浏览器访问 `http://127.0.0.1:5000`。

## 使用说明

1. **上传文件** — 点击上传区域或拖拽音频文件到界面
2. **开始处理** — 点击「开始处理」，系统自动执行 ASR 转写 → LLM 分析
3. **查看结果** — 处理完成后可在页面预览和下载分析结果
4. **打开输出目录** — 点击「打开输出文件夹」查看所有生成文件

### 分析输出示例

```
【剧情摘要】
（模型生成的剧情主线与关键情节摘要，200 字以内）

【行为关键词】
逃亡, 对峙, 密谋, 追踪, 营救, 背叛, 潜伏, 反击
```

## 项目结构

```
novel-analyzer/
├── app_flask.py          # Flask 入口，路由与后台任务管理
├── pyproject.toml        # 项目配置与依赖声明
├── core/
│   ├── asr_engine.py      # ASR 引擎（faster-whisper 封装）
│   ├── cache_manager.py   # 转写缓存管理（MD5 Key + JSON 存储）
│   └── llm_engine.py      # LLM 引擎（Qwen2.5-7B 量化推理）
├── templates/
│   └── index.html         # Web 界面
├── models/                # 模型文件目录（已 gitignore）
├── cache/                 # ASR 转写缓存（已 gitignore）
├── uploads/               # 上传音频暂存（已 gitignore）
└── output/                # 分析结果输出（已 gitignore）
```

## 注意事项

- 模型文件体积较大（~15GB），请确保磁盘有足够空间
- ASR 和 LLM 分别占用显存，处理完毕后会自动释放
- 首次处理需加载模型，耗时较长；后续处理使用缓存可跳过 ASR 阶段
- 如需切换 LLM 提示词，修改 `core/llm_engine.py` 中的 `prompt` 模板

## 技术栈

- **ASR**: faster-whisper, CTranslate2
- **LLM**: Qwen2.5-7B-Instruct, transformers + bitsandbytes 4-bit
- **框架**: Flask
- **GPU**: CUDA 12.1, PyTorch
