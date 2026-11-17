@echo off
title Phishing Email Detection System - Streamlit Launcher
echo ===================================================================
echo       SHIELD: Phishing Email Detection System Web App
echo ===================================================================
echo.

if exist "D:\Programs\Python314\python.exe" (
    echo [INFO] Found Python at D:\Programs\Python314\python.exe
    "D:\Programs\Python314\python.exe" -m streamlit run app.py
    goto END
)

where python >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Found Python in system PATH
    python -m streamlit run app.py
    goto END
)

where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [INFO] Found Python Launcher (py)
    py -m streamlit run app.py
    goto END
)

echo [ERROR] Python not found on this system.
echo Please install Python 3.10+ or ensure Python is added to your PATH.
echo.

:END
pause
