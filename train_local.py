import copy
import json
import os
import random
import time
from pathlib import Path

import PIL
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as ArrowDataset
from peft import LoraConfig, TaskType, get_peft_model
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import DonutProcessor, VisionEncoderDecoderModel, get_scheduler

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# CONFIG
DATASET_ID = os.getenv("DATASET_ID", "katanaml-org/invoices-donut-data-v1")
LOCAL_DATASET_CACHE_DIR = os.getenv(
    "LOCAL_DATASET_CACHE_DIR",
    "C:/Users/wasd/.cache/huggingface/datasets/katanaml-org___invoices-donut-data-v1/default/0.0.0/d2cde298e79c94fb05bc320999deb4b7889b0464",
)
MODEL_NAME = os.getenv("MODEL_NAME", "Bennet1996/donut-small")
TRAINING_MODE = os.getenv("TRAINING_MODE", "full").lower()
TASK_TOKEN = "<s_invoice>"
GT_FIELD = "ground_truth"

INCLUDE_ITEMS = os.getenv("INCLUDE_ITEMS", "false").lower() == "true"

MAX_TRAIN = int(os.getenv("MAX_TRAIN", "0"))
MAX_VAL = int(os.getenv("MAX_VAL", "0"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "192"))
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "480"))
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1"))
NUM_EPOCHS = int(os.getenv("NUM_EPOCHS", "5"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "2e-5"))
WEIGHT_DECAY = float(os.getenv("WEIGHT_DECAY", "0.01"))
GRADIENT_ACCUMULATION_STEPS = int(os.getenv("GRADIENT_ACCUMULATION_STEPS", "2"))
WARMUP_RATIO = float(os.getenv("WARMUP_RATIO", "0.1"))
MAX_GRAD_NORM = float(os.getenv("MAX_GRAD_NORM", "1.0"))
NUM_WORKERS = int(os.getenv("NUM_WORKERS", "2"))
SEED = int(os.getenv("SEED", "42"))
FREEZE_ENCODER = os.getenv("FREEZE_ENCODER", "false").lower() == "true"
EARLY_STOPPING_PATIENCE = int(os.getenv("EARLY_STOPPING_PATIENCE", "2"))
DORA_RANK = int(os.getenv("DORA_RANK", "8"))
DORA_ALPHA = int(os.getenv("DORA_ALPHA", "16"))
DORA_DROPOUT = float(os.getenv("DORA_DROPOUT", "0.05"))
DORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
]

MODEL_SLUG = MODEL_NAME.replace("/", "_")
RUN_SUFFIX = "dora" if TRAINING_MODE == "dora" else "ft"
OUTPUT_DIR = Path(
    os.getenv("OUTPUT_DIR", str(Path.cwd() / "outputs" / f"{MODEL_SLUG}_{RUN_SUFFIX}_best"))
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


seed_everything(SEED)

print("Pillow:", PIL.__version__)
print("Torch :", torch.__version__)
print("CUDA  :", torch.cuda.is_available())
print("Model :", MODEL_NAME)
print("Training mode:", TRAINING_MODE)
print("Freeze encoder:", FREEZE_ENCODER)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
use_amp = torch.cuda.is_available()
print("Device:", device)
if torch.cuda.is_available():
    print("GPU   :", torch.cuda.get_device_name(0))
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


def simplify_gt_parse(gt_parse, include_items=False):
    out = {}
    if "header" in gt_parse and isinstance(gt_parse["header"], dict):
        out["header"] = gt_parse["header"]
    else:
        header_keys = [
            "invoice_no",
            "invoice_date",
            "seller",
            "client",
            "seller_tax_id",
            "client_tax_id",
            "iban",
        ]
        header = {k: gt_parse[k] for k in header_keys if k in gt_parse}
        if header:
            out["header"] = header

    if include_items:
        if "items" in gt_parse and isinstance(gt_parse["items"], list):
            out["items"] = gt_parse["items"]
        else:
            item_keys = [
                "item_desc",
                "item_qty",
                "item_net_price",
                "item_net_worth",
                "item_vat",
                "item_gross_worth",
            ]
            item = {k: gt_parse[k] for k in item_keys if k in gt_parse}
            if item:
                out["items"] = [item]

    if "summary" in gt_parse and isinstance(gt_parse["summary"], dict):
        out["summary"] = gt_parse["summary"]
    else:
        summary_keys = ["total_net_worth", "total_vat", "total_gross_worth"]
        summary = {k: gt_parse[k] for k in summary_keys if k in gt_parse}
        if summary:
            out["summary"] = summary

    return out


def dict_to_donut_tags(obj):
    if isinstance(obj, dict):
        text = ""
        for k, v in obj.items():
            if k is None:
                continue
            k = str(k).strip()
            if not k:
                continue
            text += f"<s_{k}>" + dict_to_donut_tags(v) + f"</s_{k}>"
        return text
    if isinstance(obj, list):
        text = ""
        for item in obj:
            text += "<s_item>" + dict_to_donut_tags(item) + "</s_item>"
        return text
    if obj is None:
        return ""
    return str(obj)


def collect_special_tokens_from_obj(obj, tokens=None):
    if tokens is None:
        tokens = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k is None:
                continue
            k = str(k).strip()
            if not k:
                continue
            tokens.add(f"<s_{k}>")
            tokens.add(f"</s_{k}>")
            collect_special_tokens_from_obj(v, tokens)
    elif isinstance(obj, list):
        tokens.add("<s_item>")
        tokens.add("</s_item>")
        for item in obj:
            collect_special_tokens_from_obj(item, tokens)
    return tokens


def serialize_ground_truth(gt_raw):
    if gt_raw is None:
        raise ValueError("ground truth is None")
    if isinstance(gt_raw, str):
        gt_obj = json.loads(gt_raw)
    elif isinstance(gt_raw, dict):
        gt_obj = gt_raw
    else:
        raise TypeError(f"Unexpected ground_truth type: {type(gt_raw)}")

    if "gt_parse" not in gt_obj:
        raise KeyError(f"ground_truth has no 'gt_parse'. Keys: {list(gt_obj.keys())}")

    gt_parse = simplify_gt_parse(gt_obj["gt_parse"], include_items=INCLUDE_ITEMS)
    tagged = dict_to_donut_tags(gt_parse)
    if not tagged.strip():
        raise ValueError("ground truth is empty after serialization")
    return tagged, gt_parse


def prepare_item(item, task_token=TASK_TOKEN):
    image = item["image"]
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(image)}")
    tagged_text, gt_parse = serialize_ground_truth(item[GT_FIELD])
    return {
        "image": image.convert("RGB"),
        "target_text": task_token + tagged_text,
        "gt_parse": gt_parse,
    }


def build_prepared_dataset(split):
    prepared = []
    skipped = 0
    for idx in range(len(split)):
        try:
            prepared.append(prepare_item(split[idx]))
        except Exception as exc:
            skipped += 1
            print(f"Skipped #{idx}: {exc}")
    print(f"Prepared: {len(prepared)} | Skipped: {skipped}")
    return prepared


class InvoiceDonutDataset(Dataset):
    def __init__(self, data, processor, max_length=256, image_size=(1280, 960)):
        self.items = []
        self.processor = processor

        print("Pre-encoding dataset for faster training...")
        for item in tqdm(data, desc="Encoding samples"):
            image = item["image"].resize(image_size)
            target_text = item["target_text"]

            pixel_values = self.processor(
                images=image,
                return_tensors="pt",
                legacy=False,
            ).pixel_values.squeeze(0)

            labels = self.processor.tokenizer(
                target_text,
                add_special_tokens=False,
                max_length=max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            ).input_ids.squeeze(0)

            labels[labels == self.processor.tokenizer.pad_token_id] = -100

            self.items.append(
                {
                    "pixel_values": pixel_values,
                    "labels": labels,
                    "target_text": target_text,
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


@torch.no_grad()
def evaluate_loss(model, dataloader, device, use_amp):
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Validation", leave=False):
        pixel_values = batch["pixel_values"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss
        total_loss += loss.item()
    return total_loss / max(len(dataloader), 1)


def main():
    local_cache_dir = Path(LOCAL_DATASET_CACHE_DIR)
    if local_cache_dir.exists():
        hf_data = {
            "train": ArrowDataset.from_file(
                str(local_cache_dir / "invoices-donut-data-v1-train.arrow")
            ),
            "validation": ArrowDataset.from_file(
                str(local_cache_dir / "invoices-donut-data-v1-validation.arrow")
            ),
            "test": ArrowDataset.from_file(str(local_cache_dir / "invoices-donut-data-v1-test.arrow")),
        }
        print("Loaded dataset from local arrow cache:", local_cache_dir)
    else:
        hf_data = load_dataset(DATASET_ID)
    train_split = hf_data["train"]
    val_split = hf_data["validation"]

    print("train:", len(train_split))
    print("validation:", len(val_split))

    train_limit = len(train_split) if MAX_TRAIN <= 0 else min(MAX_TRAIN, len(train_split))
    val_limit = len(val_split) if MAX_VAL <= 0 else min(MAX_VAL, len(val_split))

    train_split_small = train_split.select(range(train_limit))
    val_split_small = val_split.select(range(val_limit))

    print("train used:", len(train_split_small))
    print("val used:", len(val_split_small))

    train_data = build_prepared_dataset(train_split_small)
    val_data = build_prepared_dataset(val_split_small)

    all_special_tokens = {TASK_TOKEN}
    for row in train_data + val_data:
        all_special_tokens.update(collect_special_tokens_from_obj(row["gt_parse"]))
    all_special_tokens = sorted(all_special_tokens)
    special_token_ids = []

    processor = DonutProcessor.from_pretrained(MODEL_NAME, use_fast=False)
    processor.tokenizer.pad_token = "<pad>"
    num_added = processor.tokenizer.add_special_tokens(
        {"additional_special_tokens": all_special_tokens}
    )
    for token in all_special_tokens:
        token_id = processor.tokenizer.convert_tokens_to_ids(token)
        if token_id != processor.tokenizer.unk_token_id:
            special_token_ids.append(token_id)

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    model.decoder.resize_token_embeddings(len(processor.tokenizer))

    decoder_start_token_id = processor.tokenizer.convert_tokens_to_ids(TASK_TOKEN)
    if decoder_start_token_id == processor.tokenizer.unk_token_id:
        raise ValueError("TASK_TOKEN became unk_token")

    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.eos_token_id
    model.config.decoder_start_token_id = decoder_start_token_id
    model.config.use_cache = False

    if TRAINING_MODE == "dora":
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=DORA_RANK,
            lora_alpha=DORA_ALPHA,
            lora_dropout=DORA_DROPOUT,
            target_modules=DORA_TARGET_MODULES,
            trainable_token_indices=special_token_ids,
            ensure_weight_tying=True,
            use_dora=True,
            bias="none",
        )
        model.decoder = get_peft_model(model.decoder, peft_config)
        model.decoder.print_trainable_parameters()

    model = model.to(device)
    if FREEZE_ENCODER and TRAINING_MODE != "dora":
        for param in model.encoder.parameters():
            param.requires_grad = False
    if os.getenv("USE_GRADIENT_CHECKPOINTING", "false").lower() == "true":
        model.gradient_checkpointing_enable()

    print(f"Added special tokens: {num_added}")
    print("tokenizer size:", len(processor.tokenizer))

    train_dataset = InvoiceDonutDataset(
        train_data, processor, max_length=MAX_LENGTH, image_size=IMAGE_SIZE
    )
    val_dataset = InvoiceDonutDataset(
        val_data, processor, max_length=MAX_LENGTH, image_size=IMAGE_SIZE
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable_params, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = {"train_loss": [], "val_loss": []}
    num_update_steps_per_epoch = max(
        1, (len(train_loader) + GRADIENT_ACCUMULATION_STEPS - 1) // GRADIENT_ACCUMULATION_STEPS
    )
    total_training_steps = NUM_EPOCHS * num_update_steps_per_epoch
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    best_val_loss = float("inf")
    best_state_dict = None
    best_epoch = -1
    epochs_without_improvement = 0

    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_train_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        for step, batch in enumerate(progress):
            batch_start = time.perf_counter()
            pixel_values = batch["pixel_values"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            forward_start = time.perf_counter()
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(pixel_values=pixel_values, labels=labels)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            forward_time = time.perf_counter() - forward_start

            backward_start = time.perf_counter()
            scaler.scale(loss).backward()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            backward_time = time.perf_counter() - backward_start

            optimizer_time = 0.0
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer_start = time.perf_counter()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
                prev_scale = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                if not use_amp or scaler.get_scale() >= prev_scale:
                    lr_scheduler.step()
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                optimizer_time = time.perf_counter() - optimizer_start

            real_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
            total_train_loss += real_loss
            progress.set_postfix(loss=real_loss, lr=optimizer.param_groups[0]["lr"])

            if step < 3:
                batch_time = time.perf_counter() - batch_start
                gpu_mem_gb = 0.0
                if torch.cuda.is_available():
                    gpu_mem_gb = torch.cuda.memory_allocated() / (1024 ** 3)
                print(
                    f"Step {step + 1}: total={batch_time:.2f}s "
                    f"forward={forward_time:.2f}s backward={backward_time:.2f}s "
                    f"optimizer={optimizer_time:.2f}s gpu_mem={gpu_mem_gb:.2f}GB"
                )

        if len(train_loader) % GRADIENT_ACCUMULATION_STEPS != 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, MAX_GRAD_NORM)
            prev_scale = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if not use_amp or scaler.get_scale() >= prev_scale:
                lr_scheduler.step()

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        avg_val_loss = evaluate_loss(model, val_loader, device, use_amp)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"\nEpoch {epoch + 1}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Val loss:   {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            print("Saved best checkpoint")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

        if EARLY_STOPPING_PATIENCE > 0 and epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    assert best_state_dict is not None, "best_state_dict was not saved"

    model.load_state_dict(best_state_dict)
    processor.save_pretrained(OUTPUT_DIR)
    if TRAINING_MODE == "dora":
        model.decoder.save_pretrained(OUTPUT_DIR)
        with open(OUTPUT_DIR / "base_model_name.txt", "w", encoding="utf-8") as fp:
            fp.write(MODEL_NAME)
    else:
        model.save_pretrained(OUTPUT_DIR)

    plt.figure(figsize=(8, 4))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_curve.png", dpi=160)

    training_summary = {
        "model_name": MODEL_NAME,
        "training_mode": TRAINING_MODE,
        "train_samples": len(train_data),
        "val_samples": len(val_data),
        "num_epochs_requested": NUM_EPOCHS,
        "num_epochs_completed": len(history["train_loss"]),
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "config": {
            "max_length": MAX_LENGTH,
            "image_size": list(IMAGE_SIZE),
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "max_grad_norm": MAX_GRAD_NORM,
            "freeze_encoder": FREEZE_ENCODER,
            "dora_rank": DORA_RANK,
            "dora_alpha": DORA_ALPHA,
            "dora_dropout": DORA_DROPOUT,
        },
        "history": history,
    }
    with open(OUTPUT_DIR / "training_summary.json", "w", encoding="utf-8") as fp:
        json.dump(training_summary, fp, ensure_ascii=False, indent=2)

    print("Best val loss:", best_val_loss)
    print("Best epoch:", best_epoch)
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
