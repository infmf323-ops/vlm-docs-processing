$ErrorActionPreference = "Stop"

Set-Location "E:\thesis\backend"
$env:HF_HOME = "E:\thesis\.hf-cache"
$env:HUGGINGFACE_HUB_CACHE = "E:\thesis\.hf-cache\hub"
$env:HF_HUB_CACHE = "E:\thesis\.hf-cache\hub"
$env:TRANSFORMERS_CACHE = "E:\thesis\.hf-cache\transformers"
& "E:\thesis\.conda311\python.exe" worker.py
