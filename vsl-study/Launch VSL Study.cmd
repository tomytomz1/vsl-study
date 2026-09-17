@echo off
setlocal
cd /d "%~dp0"
set "PYW=%~dp0.venv\Scripts\pythonw.exe"
set "PY=%~dp0.venv\Scripts\python.exe"

if exist "%PYW%" (
    start "" "%PYW%" -m vsl_study app
    exit /b 0
)
if exist "%PY%" (
    "%PY%" -m vsl_study app
    exit /b %errorlevel%
)

echo VSL Study is not installed in this folder.
echo Missing: .venv\Scripts\python.exe
echo Follow the setup steps in README.md first.
pause
exit /b 1
