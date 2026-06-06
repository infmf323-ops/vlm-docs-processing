$env:IMAGE_WIDTH = "640"
$env:IMAGE_HEIGHT = "480"
$env:MAX_LENGTH = "256"
$env:GRADIENT_ACCUMULATION_STEPS = "2"
$env:NUM_WORKERS = "2"
$env:USE_GRADIENT_CHECKPOINTING = "false"

& C:\Users\wasd\miniconda3\Scripts\conda.exe run -p E:\thesis\.conda311 python E:\thesis\train_local.py
