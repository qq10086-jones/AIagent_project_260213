@echo off
SETLOCAL EnableDelayedExpansion

echo ==========================================
echo   🤖 OpenClaw Nexus 一键启动装置 (Win32)
echo ==========================================
echo.

:: 1. 检查 Docker 运行状态
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Docker Desktop 未启动，请先运行 Docker Desktop！
    pause
    exit /b 1
)
echo [OK] Docker 已就绪。

:: 2. 检查 Ollama (宿主机运行模式)
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARN] 宿主机 Ollama 未响应，尝试拉起...
    start ollama serve
    echo 等待 Ollama 初始化...
    timeout /t 5 >nul
) else (
    echo [OK] Ollama 服务运行中。
)

:: 3. 启动项目容器
echo [INFO] 正在拉起 OpenClaw Nexus 核心链条...
cd /d "%~dp0\infra"
docker-compose up -d

if %errorlevel% neq 0 (
    echo [ERROR] 容器拉起失败，请检查 docker-compose.yml 配置！
    pause
    exit /b 1
)

echo.
echo [SUCCESS] 全链条已进入后台运行！
echo.
echo ==========================================
echo   🔗 服务访问入口:
echo   - Discord Bot: 直接在 Discord 输入指令 (如: $9432.T)
echo   - 控制面板 (Streamlit): http://localhost:8501
echo   - 文件存储 (MinIO): http://localhost:9001 (nexus / nexuspassword)
echo   - 数据库 (Postgres): localhost:5432
echo ==========================================
echo.
echo 按任意键退出此窗口 (服务将继续在后台运行)...
pause >nul
