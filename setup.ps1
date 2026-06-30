# Environment setup (Windows / PowerShell): venv + PyTorch + deps + setup check.
#   ./setup.ps1
# If scripts are blocked, run once:
#   Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (-not (Test-Path ".venv")) { python -m venv .venv }
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip

# PyTorch from the CUDA 12.8 index (Blackwell / RTX 50-series support).
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

python check_setup.py

Write-Host "`nReactivate later with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
