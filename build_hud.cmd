@echo off
setlocal
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0build_hud.ps1" %*
exit /b %ERRORLEVEL%
