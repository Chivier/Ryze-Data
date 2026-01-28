# 故障排查指南

## 目录

- [常见问题](#常见问题)
- [错误信息解析](#错误信息解析)
- [OCR 处理问题](#ocr-处理问题)
- [爬虫问题](#爬虫问题)
- [API 负载均衡问题](#api-负载均衡问题)
- [性能问题](#性能问题)
- [环境问题](#环境问题)
- [调试技巧](#调试技巧)
- [日志分析](#日志分析)

## 常见问题

### 1. 程序无法启动

#### 症状
```bash
ModuleNotFoundError: No module named 'xxx'
```

#### 解决方案
```bash
# 安装缺失的依赖
pip install -r requirements.txt

# 或单独安装
pip install xxx

# 验证安装
python -c "import xxx; print(xxx.__version__)"
```

#### 症状
```bash
ImportError: cannot import name 'ConfigManager' from 'src.config_manager'
```

#### 解决方案
```bash
# 确保在项目根目录运行
cd /path/to/Ryze-Data

# 设置 PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# 使用模块方式运行
python -m src.cli.main --help
```

### 2. 配置文件找不到

#### 症状
```
Warning: Config file config.json not found. Using default values.
```

#### 解决方案
```bash
# 复制配置模板
cp config.example.json config.json

# 指定配置文件路径
python -m src.cli.main --config ./config.json ocr

# 验证配置
python -m src.cli.main config-show
```

### 3. 权限错误

#### 症状
```
PermissionError: [Errno 13] Permission denied: '/data/logs/pipeline.log'
```

#### 解决方案
```bash
# 创建必要的目录
mkdir -p ./data ./logs

# 修改权限
chmod -R 755 ./data ./logs

# 或更改所有者
sudo chown -R $USER:$USER ./data ./logs
```

## 错误信息解析

### OCR 处理错误

#### 错误: PDF 文件损坏
```
Error: PDF file is corrupted or cannot be read
```

**原因**: PDF 文件下载不完整或格式不支持

**解决方案**:
```bash
# 验证 PDF 文件
file /data/pdfs/paper.pdf

# 使用 qpdf 修复
qpdf --check /data/pdfs/paper.pdf

# 重新获取 PDF
```

#### 错误: OCR 超时
```
TimeoutError: OCR processing exceeded 300 seconds
```

**原因**: 文件过大或系统资源不足

**解决方案**:
```bash
# 增加超时时间
export RYZE_OCR_TIMEOUT=600

# 减少批处理大小
export RYZE_BATCH_SIZE=1

# 禁用 GPU（如果 GPU 内存不足）
export RYZE_GPU_ENABLED=false
```

## OCR 处理问题

### OCR 输出质量差

#### 症状
- 文本识别不准确
- 图片无法正确提取
- 表格格式丢失

#### 解决方案

1. **检查 PDF 质量**
```bash
# 检查 PDF 是否为扫描件
pdfinfo paper.pdf
```

2. **调整 OCR 参数**
```python
# 在配置文件中调整
{
    "ocr": {
        "confidence_threshold": 0.9,
        "preserve_layout": true
    }
}
```

3. **使用 GPU 加速**
```bash
export RYZE_GPU_ENABLED=true
export CUDA_VISIBLE_DEVICES=0
```

### 分块 OCR 处理失败

#### 症状
```
Error: Chunked OCR failed for batch 3
```

#### 解决方案
```bash
# 检查失败的文件
cat /data/ocr_results/failed_files.log

# 重新处理失败的文件
python -m src.chunked_ocr --retry-failed

# 减少并行度
python -m src.chunked_ocr --workers 2
```

## 爬虫问题

### 连接被拒绝

#### 症状
```
ConnectionError: HTTPSConnectionPool(host='nature.com', port=443): Max retries exceeded
```

**原因**: 网络问题或被目标网站限制

**解决方案**:
```bash
# 使用代理
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# 增加重试次数和延迟
# 在配置文件中设置
{
    "processing": {
        "max_retries": 5,
        "retry_delay_seconds": 10
    }
}
```

### 爬取数据不完整

#### 症状
- CSV 文件缺少字段
- 部分文章未被爬取

#### 解决方案
```bash
# 检查爬取状态
python -m src.cli.main inspect stage scraping --detailed

# 重新爬取
python -m src.cli.main scrape --force
```

## API 负载均衡问题

### API 密钥无效

#### 症状
```
openai.AuthenticationError: Invalid API key
```

**解决方案**:
```bash
# 检查 API 密钥
echo $OPENAI_API_KEY

# 重新设置
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# 验证密钥
python -c "import openai; client = openai.OpenAI(); print(client.models.list())"
```

### 速率限制

#### 症状
```
RateLimitError: You exceeded your current quota
```

**解决方案**:
```bash
# 减少并发数
export RYZE_NUM_WORKERS=1

# 增加请求间隔
export RYZE_API_DELAY=2
```

### 负载均衡器请求失败

#### 症状
```
All API keys exhausted after retries
```

**解决方案**:
```python
# 检查统计信息
from src.api_key_balancer import OpenAIAPIBalancer

balancer = OpenAIAPIBalancer(api_keys)
stats = balancer.get_statistics()
print(stats)

# 查看哪些 key 失败
for worker in stats['workers']:
    print(f"Worker {worker['thread_id']}: "
          f"processed={worker['processed']}, "
          f"failed={worker['failed']}")
```

## 性能问题

### 处理速度慢

#### 诊断
```bash
# 查看系统资源
htop
nvidia-smi  # GPU 使用情况

# 检查 I/O 性能
iostat -x 1
```

#### 优化方案

1. **增加并行度**
```bash
export RYZE_NUM_WORKERS=8
export RYZE_BATCH_SIZE=20
```

2. **启用 GPU 加速**
```bash
export RYZE_GPU_ENABLED=true
export RYZE_GPU_MEMORY_LIMIT=0.8
```

3. **使用 SSD 存储**
```bash
# 将数据目录移到 SSD
export RYZE_DATA_ROOT=/ssd/ryze_data
```

### 内存不足

#### 症状
```
MemoryError: Unable to allocate array
```

#### 解决方案

1. **减少批处理大小**
```bash
export RYZE_BATCH_SIZE=1
```

2. **清理缓存**
```python
import gc
gc.collect()

# 清理 CUDA 缓存
import torch
torch.cuda.empty_cache()
```

## 环境问题

### Python 版本不兼容

```bash
# 检查 Python 版本
python --version  # 需要 3.8+

# 使用 pyenv 管理版本
pyenv install 3.9.15
pyenv local 3.9.15

# 或使用 conda
conda create -n ryze python=3.9
conda activate ryze
```

### 依赖冲突

```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate

# 重新安装依赖
pip install --upgrade pip
pip install -r requirements.txt

# 检查依赖冲突
pip check
```

### GPU 驱动问题

```bash
# 检查 CUDA 版本
nvidia-smi
nvcc --version

# 安装对应的 PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 验证 GPU 可用
python -c "import torch; print(torch.cuda.is_available())"
```

## 调试技巧

### 启用详细日志

```bash
# 设置日志级别
export RYZE_LOG_LEVEL=DEBUG

# 或在代码中
import logging
logging.basicConfig(level=logging.DEBUG)
```

### 使用断点调试

```python
# 使用 pdb
import pdb
pdb.set_trace()

# 使用 ipdb（更友好）
import ipdb
ipdb.set_trace()

# 条件断点
if error_condition:
    breakpoint()  # Python 3.7+
```

### 性能分析

```python
# 使用 cProfile
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()
# 你的代码
profiler.disable()

stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

## 日志分析

### 查找错误

```bash
# 查找所有错误
grep ERROR logs/pipeline.log

# 查找特定论文的错误
grep "nature04244.*ERROR" logs/pipeline.log

# 统计错误类型
grep ERROR logs/pipeline.log | cut -d'-' -f4 | sort | uniq -c
```

### 监控实时日志

```bash
# 实时查看日志
tail -f logs/pipeline.log

# 只看错误
tail -f logs/pipeline.log | grep --line-buffered ERROR
```

## 获取帮助

### 查看帮助信息

```bash
# 主命令帮助
python -m src.cli.main --help

# 子命令帮助
python -m src.cli.main inspect --help
python -m src.cli.main ocr --help
```

### 提交问题

在提交问题前，请收集以下信息：

1. **系统信息**
```bash
python --version
pip list | grep -E "openai|marker|pytest"
uname -a
```

2. **错误日志**
```bash
tail -n 100 logs/pipeline.log > error_log.txt
```

3. **配置信息**（去除敏感信息）
```bash
python -m src.cli.main config-show | sed 's/sk-.*/[REDACTED]/' > config_info.txt
```

4. **最小复现示例**
```python
# 能复现问题的最简代码
from src.config_manager import config
config.load()
# 触发错误的代码
```
