@echo off
rem Run the complete reproducible social behavior / vocalization analysis.
cd /d "%~dp0.."
python scripts\social_vocal_correlations.py %*
if errorlevel 1 pause
