@echo off
cd /d "%~dp0"

echo ============================================
echo  AI Lab - local LLM + YOLO workbench
echo  LLM backend    : Ollama / qwen2.5:1.5b
echo  Open in browser: http://127.0.0.1:8000/
echo  Close this window to stop the server
echo ============================================
echo.
echo Starting, please wait...
echo.

rem Prefer our own fine-tuned weights; fall back to the official pretrained ones.
set WEIGHTS=runs\train\coco8-ft\weights\best.pt
if not exist "%WEIGHTS%" set WEIGHTS=yolo11n.pt
if not exist "%WEIGHTS%" set WEIGHTS=

if "%WEIGHTS%"=="" goto plain

echo Detection weights: %WEIGHTS%
python run_server.py --provider ollama --model qwen2.5:1.5b --port 8000 --vision-backend ultralytics --vision-weights "%WEIGHTS%" --vision-conf 0.45
goto done

:plain
python run_server.py --provider ollama --model qwen2.5:1.5b --port 8000

:done
echo.
echo --------------------------------------------
echo Server stopped.
echo If there are errors above, check:
echo   1. Python is installed and added to PATH
echo   2. Port 8000 is free (the script picks a free port automatically)
echo --------------------------------------------
pause
