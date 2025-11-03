@echo off
REM UDP CSV Data Relay System - Windows Launcher Script
REM This script launches the receiver, relay, and sender together on Windows

setlocal enabledelayedexpansion

REM Check if Python 3 is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python 3 is not installed or not in PATH
    exit /b 1
)

REM Get the current directory
set SCRIPT_DIR=%~dp0

REM Check if required Python files exist
for %%F in (receiver.py relay.py sender.py) do (
    if not exist "!SCRIPT_DIR!%%F" (
        echo Error: %%F not found in !SCRIPT_DIR!
        exit /b 1
    )
)

echo.
echo ========================================
echo UDP CSV Data Relay System
echo ========================================
echo.

REM Start the receiver in a new window
echo [1/3] Starting Receiver...
start "UDP Receiver" cmd /k "cd /d !SCRIPT_DIR! && python receiver.py"
timeout /t 2 /nobreak

REM Start the relay in a new window
echo [2/3] Starting Relay...
start "UDP Relay" cmd /k "cd /d !SCRIPT_DIR! && python relay.py"
timeout /t 2 /nobreak

REM Start the sender in a new window
echo [3/3] Starting Sender...
start "UDP Sender" cmd /k "cd /d !SCRIPT_DIR! && python sender.py"

echo.
echo ========================================
echo All components are running!
echo ========================================
echo.
echo Check the three new windows for:
echo   - UDP Receiver (listening on port 5001)
echo   - UDP Relay (forwarding from 5000 to 5001)
echo   - UDP Sender (reading CSV and sending)
echo.
echo To stop all processes, close each window or use stop_all.bat
echo.
