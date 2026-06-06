$ErrorActionPreference = "Stop"

$env:HF_HOME = "E:\thesis\.hf-cache"
$env:HUGGINGFACE_HUB_CACHE = "E:\thesis\.hf-cache\hub"
$env:HF_HUB_CACHE = "E:\thesis\.hf-cache\hub"
$env:TRANSFORMERS_CACHE = "E:\thesis\.hf-cache\transformers"

$env:NUM_EPOCHS = "3"
$env:BATCH_SIZE = "1"
$env:GRADIENT_ACCUMULATION_STEPS = "1"
$env:LEARNING_RATE = "2e-4"
$env:WEIGHT_DECAY = "0.01"
$env:WARMUP_RATIO = "0.1"
$env:MAX_IMAGE_SIDE = "768"
$env:MAX_LENGTH = "2048"
$env:OUTPUT_DIR = "E:\thesis\outputs\paddleocr_vl_multidoc_lora_pilot"

& "E:\thesis\.conda311\python.exe" "E:\thesis\backend\scripts\train_paddleocr_vl_lora.py"
