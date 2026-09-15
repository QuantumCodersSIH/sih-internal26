$ErrorActionPreference = 'Stop'

Write-Host 'SignalScope minimal UI/robustness presentation fix' -ForegroundColor Cyan
Write-Host 'Changes: fixes broken text encoding and makes Active Defence a clearly global 200-image diagnostic.'
Write-Host 'No model weights, threshold, temperature, or prediction code are changed.'

if (-not (Test-Path .\app.py)) {
    throw 'app.py was not found. Run this script from the SignalScope project root.'
}

$backup = '.\app.py.before_ui_fix.bak'
Copy-Item .\app.py $backup -Force
Copy-Item .\signalscope_ui_fix_pkg\app.py .\app.py -Force

python -m py_compile .\app.py

Write-Host ''
Write-Host "Updated app.py. Backup: $backup" -ForegroundColor Green
Write-Host 'Run: python -m streamlit run app.py' -ForegroundColor Yellow
