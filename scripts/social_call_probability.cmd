@echo off
rem Reproduce conditional vocalization probabilities and all scientific figures.
cd /d "%~dp0.."
python scripts\social_call_probability.py %*
if errorlevel 1 pause
