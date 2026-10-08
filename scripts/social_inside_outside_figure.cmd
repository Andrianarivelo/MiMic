@echo off
rem Reproduce WT/Het inside/outside social behavior boxstrips.
cd /d "%~dp0.."
python scripts\social_inside_outside_figure.py %*
if errorlevel 1 pause
