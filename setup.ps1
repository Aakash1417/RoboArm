# One-time setup: creates .venv with Python 3.11, installs dependencies, downloads the hand model,
# and runs the headless self-test.
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    Write-Host "Creating virtual environment (.venv) with Python 3.11..."
    py -3.11 -m venv .venv
}

.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt

if (-not (Test-Path .\models\hand_landmarker.task)) {
    Write-Host "Downloading MediaPipe hand landmarker model..."
    New-Item -ItemType Directory -Force models | Out-Null
    Invoke-WebRequest -Uri "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task" -OutFile models\hand_landmarker.task
}

.\.venv\Scripts\python.exe selftest.py
Write-Host ""
Write-Host "Setup done. Start the app with:  .\.venv\Scripts\python.exe main.py"
