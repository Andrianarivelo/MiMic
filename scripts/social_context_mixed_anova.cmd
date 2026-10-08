@echo off
rem Four-condition raw-scale mixed ANOVA figure with Holm comparisons.
cd /d "%~dp0.."
python scripts\social_context_mixed_anova.py %*
if errorlevel 1 pause
