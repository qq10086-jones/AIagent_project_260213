FROM python:3.11-slim

# 1. 安装系统级依赖：FFmpeg 和基础库
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. 拷贝依赖定义并安装
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# 3. 拷贝所有代码
COPY . .

# 启动
CMD ["python","worker.py"]
