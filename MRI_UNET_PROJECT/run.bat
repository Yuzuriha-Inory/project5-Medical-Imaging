@echo off
chcp 65001 >nul
setlocal

echo ====================================================
echo MRI U-Net 项目一键运行脚本
echo ====================================================

cd /d "%~dp0"

set PYTHON_EXE=D:\med_env\python.exe

echo 当前目录：%cd%
echo 使用解释器：%PYTHON_EXE%
echo.

if not exist "%PYTHON_EXE%" (
    echo [错误] 未找到 Python 解释器：
    echo %PYTHON_EXE%
    echo 请检查 PyCharm 中项目解释器的真实路径。
    pause
    exit /b 1
)

if not exist main.py (
    echo [错误] 未找到 main.py。
    echo 请确认 run.bat 放在项目根目录 D:\MRI_UNET_PROJECT 下。
    pause
    exit /b 1
)

if not exist config.py (
    echo [错误] 未找到 config.py。
    echo 请确认项目文件完整。
    pause
    exit /b 1
)

if not exist models (
    echo [错误] 未找到 models 文件夹。
    echo 请确认项目文件完整。
    pause
    exit /b 1
)

if not exist utils (
    echo [错误] 未找到 utils 文件夹。
    echo 请确认项目文件完整。
    pause
    exit /b 1
)

echo [信息] 检查核心依赖...
"%PYTHON_EXE%" -c "import torch, torchvision, numpy, pandas, matplotlib, sklearn, PIL, tqdm, skimage, scipy, SimpleITK; print('环境检查通过'); print('torch=', torch.__version__); print('torchvision=', torchvision.__version__)"

if errorlevel 1 (
    echo.
    echo [错误] 环境检查失败。
    echo 请检查 med_env 环境中的依赖是否安装完整。
    pause
    exit /b 1
)

echo.
echo ====================================================
echo 开始运行 main.py
echo 训练过程将实时显示在当前窗口中
echo ====================================================
echo.

"%PYTHON_EXE%" -u main.py

if errorlevel 1 (
    echo.
    echo [错误] main.py 运行失败。
    echo 请根据上方报错信息检查代码、数据路径或环境依赖。
    pause
    exit /b 1
)

echo.
echo ====================================================
echo 项目运行完成！
echo 输出结果请查看 config.py 中设置的 OUTPUT_DIR。
echo ====================================================

pause