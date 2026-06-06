$ErrorActionPreference = "Stop"

$root = "E:\thesis"
$python = "E:\thesis\.conda311\python.exe"

Write-Host "Running local passport transfer pipeline..."
& $python "$root\backend\scripts\run_passport_transfer_experiment.py" --prepare-data @args
