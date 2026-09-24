@echo off
REM RAG 评测快速启动脚本 (Windows)

echo ================================
echo RAG 评测系统 - 快速启动
echo ================================
echo.

REM 1. 健康检查
echo 1. 运行健康检查...
python -m app.evaluation.health_check

if %errorlevel% neq 0 (
    echo.
    echo ❌ 健康检查失败，请先修复问题
    exit /b 1
)

echo.

REM 2. 选择评测模式
echo 请选择评测模式：
echo   1) 快速评测（仅 baseline，约 3 分钟）
echo   2) 完整评测（7 个配置对比，约 10 分钟）
echo   3) 自定义评测
echo.
set /p choice="请输入选项 (1-3): "

if "%choice%"=="1" (
    echo.
    echo 🚀 开始快速评测...
    python -m app.evaluation.cli quick
) else if "%choice%"=="2" (
    echo.
    echo 🚀 开始完整评测...
    python -m app.evaluation.cli full
) else if "%choice%"=="3" (
    set /p config_path="请输入配置文件路径: "
    echo.
    echo 🚀 开始自定义评测...
    python -m app.evaluation.cli custom --config "%config_path%"
) else (
    echo 无效选项
    exit /b 1
)

echo.
echo ================================
echo ✅ 评测完成！
echo ================================
pause
