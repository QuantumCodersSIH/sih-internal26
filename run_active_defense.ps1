$ErrorActionPreference = "Stop"

# Run from the SignalScope project root.
New-Item -ItemType Directory -Force -Path "src\defense" | Out-Null
New-Item -ItemType Directory -Force -Path "scripts" | Out-Null
New-Item -ItemType Directory -Force -Path "results\active_defense" | Out-Null

Copy-Item "active_defense.py" "src\defense\active_defense.py" -Force

Write-Host "Checking syntax..." -ForegroundColor Yellow
python -m py_compile src\defense\active_defense.py

Write-Host ""
Write-Host "Running active-defence analysis on 100 REAL + 100 FAKE images..." -ForegroundColor Cyan

python src\defense\active_defense.py `
  --checkpoint "models\checkpoints\best_model.pt" `
  --real-dir "TEMP_TEST\REAL" `
  --fake-dir "TEMP_TEST\FAKE" `
  --limit 100 `
  --output-dir "results\active_defense"

Write-Host ""
Write-Host "Completed." -ForegroundColor Green
Write-Host "Open: results\active_defense\active_defense_report.md"
