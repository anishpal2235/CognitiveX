# ---------------------------------------------------------------
# ControlPlane.ai - one-shot setup + run for Windows PowerShell
# Usage:  .\run.ps1
#         $env:VENV=".respo"; .\run.ps1
#
# If PowerShell refuses to run this file, unblock it for this
# session only (does not change machine policy):
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
# ---------------------------------------------------------------
$ErrorActionPreference = "Stop"

$Venv = if ($env:VENV) { $env:VENV } else { ".venv" }

# Prefer the py launcher, fall back to python on PATH
$Py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } else { "python" }
$PyArgs = if ($Py -eq "py") { @("-3") } else { @() }

$PyBin = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $PyBin)) {
    Write-Host "[1/5] Creating virtual environment $Venv ..." -ForegroundColor Cyan
    & $Py @PyArgs -m venv $Venv
} else {
    Write-Host "[1/5] Reusing existing virtual environment $Venv" -ForegroundColor Cyan
}

Write-Host "[2/5] Installing dependencies ..." -ForegroundColor Cyan
& $PyBin -m pip install --upgrade pip -q
& $PyBin -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
    Write-Host "[3/5] Creating .env from .env.example" -ForegroundColor Cyan
    Copy-Item ".env.example" ".env"
} else {
    Write-Host "[3/5] .env already present, leaving it alone" -ForegroundColor Cyan
}

Write-Host "[4/5] Running tests ..." -ForegroundColor Cyan
& $PyBin -m pytest -q
if ($LASTEXITCODE -ne 0) {
    Write-Host "WARNING: some tests failed - continuing anyway." -ForegroundColor Yellow
}

Write-Host "[4b/5] Seeding router from offline preference data ..." -ForegroundColor Cyan
& $PyBin -m scripts.seed

Write-Host ""
Write-Host "[5/5] Starting API on http://127.0.0.1:8000   (Ctrl+C to stop)" -ForegroundColor Green
Write-Host "      Interactive docs: http://127.0.0.1:8000/docs" -ForegroundColor Green
Write-Host ""
& $PyBin -m uvicorn controlplane.app:app --reload --port 8000
