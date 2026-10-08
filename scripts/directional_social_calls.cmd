@echo off
rem Paired A-to-B / B-to-A vocal contexts.
cd /d "%~dp0.."
python scripts\directional_social_calls.py %*
if errorlevel 1 pause
