@echo off
REM ---------------------------------------------------------------
REM ControlPlane.ai - one-shot setup + run for Windows (cmd.exe)
REM Usage:  run.bat            (uses .venv)
REM         set VENV=.respo && run.bat
REM ---------------------------------------------------------------
setlocal

if "%VENV%"=="" set VENV=.venv

REM Prefer the py launcher, fall back to python on PATH
set PY=python
where py >nul 2>nul
if not errorlevel 1 set PY=py -3

if not exist "%VENV%\Scripts\python.exe" (
  echo [1/5] Creating virtual environment %VENV% ...
  %PY% -m venv "%VENV%"
  if errorlevel 1 goto fail
) else (
  echo [1/5] Reusing existing virtual environment %VENV%
)

set PYBIN=%VENV%\Scripts\python.exe

echo [2/5] Installing dependencies ...
"%PYBIN%" -m pip install --upgrade pip -q
if errorlevel 1 goto fail
"%PYBIN%" -m pip install -r requirements.txt
if errorlevel 1 goto fail

if not exist ".env" (
  echo [3/5] Creating .env from .env.example
  copy /y .env.example .env >nul
) else (
  echo [3/5] .env already present, leaving it alone
)

echo [4/5] Running tests ...
"%PYBIN%" -m pytest -q
if errorlevel 1 echo WARNING: some tests failed - continuing anyway.

echo [4b/5] Seeding router from offline preference data ...
"%PYBIN%" -m scripts.seed
if errorlevel 1 goto fail

echo.
echo [5/5] Starting API on http://127.0.0.1:8000   (Ctrl+C to stop)
echo       Interactive docs: http://127.0.0.1:8000/docs
echo.
"%PYBIN%" -m uvicorn controlplane.app:app --reload --port 8000
goto end

:fail
echo.
echo FAILED - see the error above.
exit /b 1

:end
endlocal
