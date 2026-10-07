@echo off
rem Reproduce the supplied legacy image in its separate styling output folder.
cd /d "%~dp0.."
python scripts\restyle_lgdel_usv_figure.py %*
if errorlevel 1 pause
