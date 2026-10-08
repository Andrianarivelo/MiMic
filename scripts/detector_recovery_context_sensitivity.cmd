@echo off
rem With no arguments, prepare only. Pass --stage detect or --stage combine explicitly.
cd /d "%~dp0.."
set "CONTEXT_RECOVERY_PYTHON=%USERPROFILE%\miniconda3\envs\mamir\python.exe"
if not exist "%CONTEXT_RECOVERY_PYTHON%" set "CONTEXT_RECOVERY_PYTHON=python"
"%CONTEXT_RECOVERY_PYTHON%" scripts\detector_recovery_context_sensitivity.py %*
if errorlevel 1 pause
