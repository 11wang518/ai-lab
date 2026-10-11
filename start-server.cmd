@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo  AI Lab 本地大模型 + YOLO 工作台
echo  模型后端：Ollama / qwen2.5:1.5b
echo  启动后浏览器访问：http://127.0.0.1:8000/
echo  关闭本窗口即停止服务
echo ============================================
echo.
echo 正在启动，请稍候...
echo.

python run_server.py --provider ollama --model qwen2.5:1.5b --port 8000

echo.
echo --------------------------------------------
echo 服务已停止。
echo 如果上面有红色报错，通常是这两种情况：
echo   1. 没装 Python 或没加入 PATH
echo   2. 8000 端口被占用（脚本会自动往后找可用端口，看上面的启动日志）
echo --------------------------------------------
pause
