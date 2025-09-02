# 故障排查指南

## 目录

- [常见问题](#常见问题)
- [错误信息解析](#错误信息解析)
- [性能问题](#性能问题)
- [数据问题](#数据问题)
- [API 相关问题](#api-相关问题)
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
python -m src.cli.main --config ./config.json pipeline

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
# 重新下载 PDF
python -m src.cli.main download --force

# 验证 PDF 文件
file /data/pdfs/nature04244.pdf

# 使用 qpdf 修复
qpdf --check /data/pdfs/nature04244.pdf
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

### 爬虫错误

#### 错误: 连接被拒绝
```
ConnectionError: HTTPSConnectionPool(host='nature.com', port=443): Max retries exceeded
```

**原因**: 网络问题或被目标网站限制

**解决方案**:
```python
# 增加重试次数和延迟
config.processing.max_retries = 5
config.processing.retry_delay_seconds = 10

# 使用代理
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# 降低请求频率
time.sleep(5)  # 在请求之间添加延迟
```

### API 错误

#### 错误: API 密钥无效
```
openai.AuthenticationError: Invalid API key
```

**原因**: API 密钥错误或过期

**解决方案**:
```bash
# 检查 API 密钥
echo $OPENAI_API_KEY

# 重新设置
export OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxx

# 验证密钥
python -c "import openai; client = openai.OpenAI(); print(client.models.list())"
```

#### 错误: 速率限制
```
RateLimitError: You exceeded your current quota
```

**原因**: API 调用超过限制

**解决方案**:
```bash
# 减少并发数
export RYZE_NUM_WORKERS=1

# 增加请求间隔
export RYZE_API_DELAY=2

# 使用更小的批次
export RYZE_BATCH_SIZE=5
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

4. **优化内存使用**
```python
# 增加 Python 内存限制
import resource
resource.setrlimit(resource.RLIMIT_AS, (16 * 1024 * 1024 * 1024, -1))
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

3. **使用分块处理**
```python
# 处理大文件时分块
chunk_size = 1000
for i in range(0, len(data), chunk_size):
    chunk = data[i:i+chunk_size]
    process(chunk)
```

## 数据问题

### 数据不完整

#### 检查数据完整性
```bash
# 检查各阶段数据
python -m src.cli.main inspect all

# 查看特定阶段
python -m src.cli.main inspect stage ocr --detailed

# 验证单个文件
python -m src.cli.main inspect file /data/ocr_results/nature04244/nature04244.md
```

#### 修复缺失数据
```bash
# 重新运行特定阶段
python -m src.cli.main pipeline --stages ocr extract --force

# 处理特定论文
export RYZE_PAPER_ID=nature04244
python -m src.cli.main generate-qa
```

### 数据格式错误

#### 症状
```
json.JSONDecodeError: Expecting value: line 1 column 1
```

#### 诊断和修复
```python
# 验证 JSON 格式
import json

def validate_json(file_path):
    try:
        with open(file_path, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print(f"Invalid JSON: {e}")
        return False

# 修复 JSONL 文件
def fix_jsonl(input_file, output_file):
    valid_lines = []
    with open(input_file, 'r') as f:
        for i, line in enumerate(f):
            try:
                json.loads(line.strip())
                valid_lines.append(line)
            except:
                print(f"Skip invalid line {i+1}")
    
    with open(output_file, 'w') as f:
        for line in valid_lines:
            f.write(line)
```

## API 相关问题

### OpenAI API 问题

#### 使用自定义端点
```bash
# Azure OpenAI
export OPENAI_BASE_URL=https://your-resource.openai.azure.com/
export OPENAI_API_VERSION=2023-05-15

# 本地 LLM
export OPENAI_BASE_URL=http://localhost:11434/v1
export RYZE_LLM_MODEL=llama2:70b
```

#### API 响应错误处理
```python
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(min=1, max=10)
)
def call_openai_api(prompt):
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except openai.RateLimitError:
        time.sleep(60)  # 等待 1 分钟
        raise
    except openai.APIError as e:
        logger.error(f"API error: {e}")
        raise
```

### 网络问题

#### 使用代理
```bash
# HTTP 代理
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890

# SOCKS5 代理
export ALL_PROXY=socks5://127.0.0.1:1080

# 不使用代理的地址
export NO_PROXY=localhost,127.0.0.1
```

#### DNS 问题
```bash
# 更换 DNS
echo "nameserver 8.8.8.8" | sudo tee /etc/resolv.conf

# 清除 DNS 缓存
sudo systemd-resolve --flush-caches
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
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate  # Windows

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

### 内存分析

```python
# 使用 memory_profiler
from memory_profiler import profile

@profile
def memory_intensive_function():
    # 你的代码
    pass

# 运行: python -m memory_profiler your_script.py
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

### 分析处理时间

```bash
# 提取处理时间
grep "Processing completed" logs/pipeline.log | \
  awk '{print $NF}' | \
  awk '{sum+=$1; count++} END {print "Average:", sum/count}'
```

### 监控实时日志

```bash
# 实时查看日志
tail -f logs/pipeline.log

# 只看错误
tail -f logs/pipeline.log | grep --line-buffered ERROR

# 彩色输出
tail -f logs/pipeline.log | ccze -A
```

## 恢复和回滚

### 从检查点恢复

```bash
# 保存流水线状态
python -c "
from src.pipeline_manager import PipelineManager
from src.config_manager import config
config.load()
pipeline = PipelineManager(config)
pipeline.save_state('pipeline_checkpoint.json')
"

# 从检查点恢复
python -m src.cli.main pipeline --resume-from pipeline_checkpoint.json
```

### 数据备份

```bash
# 备份重要数据
tar -czf backup_$(date +%Y%m%d).tar.gz \
  data/sft_data \
  data/vlm_sft_data \
  logs/

# 恢复数据
tar -xzf backup_20240115.tar.gz
```

### 清理和重置

```bash
# 清理临时文件
find ./temp -type f -mtime +7 -delete

# 清理失败的处理
rm -rf data/ocr_results/*/failed_*

# 重置特定阶段
rm -rf data/sft_data/*
python -m src.cli.main generate-qa --force
```

## 获取帮助

### 查看帮助信息

```bash
# 主命令帮助
python -m src.cli.main --help

# 子命令帮助
python -m src.cli.main pipeline --help
python -m src.cli.main inspect --help
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