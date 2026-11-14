@echo off
setlocal DisableDelayedExpansion
pushd "%~dp0" || exit /b 1
if not exist ".venv\Scripts\python.exe" (
    echo Project Python environment is missing. See README.md.
    popd
    exit /b 1
)
set "PYTHONUTF8=1"
set "PYTHONNOUSERSITE=1"
".venv\Scripts\python.exe" app.py %*
set "APP_EXIT=%ERRORLEVEL%"
popd
exit /b %APP_EXIT%
