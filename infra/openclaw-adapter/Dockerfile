FROM python:3.11-slim

# 1. 仅使用 apt-get 安装必要的网络请求工具 (curl)
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# 2. 终极杀招：直接从 Docker 官方纯净 CLI 镜像中复制编译好的 docker 二进制程序
COPY --from=docker:cli /usr/local/bin/docker /usr/local/bin/docker

RUN pip install urllib3 boto3

WORKDIR /app
COPY server.py /app/server.py

EXPOSE 18081
CMD ["python", "/app/server.py"]