@echo off
rem Run the real-audio injection/recovery calibration in the tested environment.
cd /d "%~dp0.."
set "RECOVERY_PYTHON=%USERPROFILE%\miniconda3\envs\mamir\python.exe"
if not exist "%RECOVERY_PYTHON%" set "RECOVERY_PYTHON=python"
"%RECOVERY_PYTHON%" scripts\detector_recovery.py %*
if errorlevel 1 pause
