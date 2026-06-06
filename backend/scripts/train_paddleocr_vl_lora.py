from __future__ import annotations

import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from PIL import Image
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


DATA_DIR = Path(os.getenv("MULTIDOC_DATA_DIR", "E:/thesis/data/multidoc"))
TRAIN_FILE = Path(os.getenv("MULTIDOC_TRAIN_FILE", str(DATA_DIR / "pilot_train.jsonl")))
VAL_FILE = Path(os.getenv("MULTIDOC_VAL_FILE", str(DATA_DIR / "pilot_val.jsonl")))
MODEL_NAME = os.getenv("PADDLEOCR_VL_MODEL_NAME", "PaddlePaddle/PaddleOCR-VL")
HF_CACHE = Path(os.getenv("HF_HOME", "E:/thesis/.hf-cache"))
OUTPUT_DIR = Path(
    os.getenv("OUTPUT_DIR", "E:/thesis/outputs/paddleocr_vl_multidoc_lora")
)
BASE_ADAPTER_DIR = os.getenv("BASE_ADAPTER_DIR", "").strip()

MAX_LENGTH = env_int("MAX_LENGTH", 2048)
MAX_NEW_TOKENS = env_int("MAX_NEW_TOKENS", 256)
BATCH_SIZE = env_int("BATCH_SIZE", 1)
NUM_EPOCHS = env_int("NUM_EPOCHS", 3)
LEARNING_RATE = env_float("LEARNING_RATE", 2e-4)
WEIGHT_DECAY = env_float("WEIGHT_DECAY", 0.01)
WARMUP_RATIO = env_float("WARMUP_RATIO", 0.1)
MAX_GRAD_NORM = env_float("MAX_GRAD_NORM", 1.0)
GRADIENT_ACCUMULATION_STEPS = env_int("GRADIENT_ACCUMULATION_STEPS", 2)
LORA_RANK = env_int("LORA_RANK", 8)
LORA_ALPHA = env_int("LORA_ALPHA", 16)
LORA_DROPOUT = env_float("LORA_DROPOUT", 0.05)
SEED = env_int("SEED", 42)
MAX_TRAIN_ROWS = env_int("MAX_TRAIN_ROWS", 0)
MAX_VAL_ROWS = env_int("MAX_VAL_ROWS", 0)
DRY_RUN = os.getenv("DRY_RUN", "0") == "1"
MAX_IMAGE_SIDE = env_int("MAX_IMAGE_SIDE", 768)
TARGET_FORMAT = os.getenv("TARGET_FORMAT", "json")


def configure_env() -> None:
    HF_CACHE.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")


def save_training_summary(*, output_dir: Path, best_val_loss: float, history: dict[str, list[float]]) -> None:
    summary = {
        "model_name": MODEL_NAME,
        "train_file": str(TRAIN_FILE),
        "val_file": str(VAL_FILE),
        "num_epochs": NUM_EPOCHS,
        "best_val_loss": best_val_loss,
        "config": {
            "max_length": MAX_LENGTH,
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_image_side": MAX_IMAGE_SIDE,
            "batch_size": BATCH_SIZE,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
            "lora_rank": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "target_format": TARGET_FORMAT,
            "base_adapter_dir": BASE_ADAPTER_DIR or None,
        },
        "history": history,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl(path: Path) -> list[dict]:
    def resolve_image_path(raw_path: str) -> str:
        candidate = Path(raw_path)
        if candidate.exists():
            return str(candidate)

        normalized = raw_path.replace("\\", "/")
        prefixes = ["E:/thesis/", "/kaggle/working/thesis_bundle/"]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                relative = normalized[len(prefix) :].lstrip("/")
                remapped = PROJECT_ROOT / Path(relative)
                if remapped.exists():
                    return str(remapped)

        marker = "data/multidoc/"
        if marker in normalized:
            relative = normalized.split(marker, 1)[1]
            remapped = PROJECT_ROOT / "data" / "multidoc" / Path(relative)
            if remapped.exists():
                return str(remapped)
        return raw_path

    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            if row.get("bootstrap_template") is True:
                continue
            if "image_path" in row:
                row["image_path"] = resolve_image_path(str(row["image_path"]))
            rows.append(row)
    return rows


def build_target(document_type: str, fields: dict) -> str:
    if TARGET_FORMAT == "passport_mrz" and document_type == "passport":
        value = fields.get("mrz")
        return "null" if value in (None, "") else str(value)
    if TARGET_FORMAT == "passport_flat" and document_type == "passport":
        ordered_keys = [
            "document_number",
            "surname",
            "given_names",
            "nationality",
            "date_of_birth",
            "sex",
            "place_of_birth",
            "date_of_issue",
            "date_of_expiry",
            "issuing_authority",
            "mrz",
        ]
        lines = []
        for key in ordered_keys:
            value = fields.get(key)
            if value is None:
                rendered = "null"
            else:
                rendered = str(value)
            lines.append(f"{key}: {rendered}")
        return "\n".join(lines)
    payload = {
        "document_type": document_type,
        "fields": fields,
    }
    return json.dumps(payload, ensure_ascii=False)


def build_instruction(document_type: str) -> str:
    if TARGET_FORMAT == "passport_mrz" and document_type == "passport":
        return (
            "Extract only the passport MRZ exactly as plain text. "
            "Return only the MRZ lines, preserving line breaks. "
            "Do not add labels, JSON, or explanation. "
            "If MRZ is missing, write null."
        )
    if TARGET_FORMAT == "passport_flat" and document_type == "passport":
        return (
            "Extract the passport fields exactly in plain text, one field per line. "
            "Use this exact format and exact field names only:\n"
            "document_number: ...\n"
            "surname: ...\n"
            "given_names: ...\n"
            "nationality: ...\n"
            "date_of_birth: ...\n"
            "sex: ...\n"
            "place_of_birth: ...\n"
            "date_of_issue: ...\n"
            "date_of_expiry: ...\n"
            "issuing_authority: ...\n"
            "mrz: ...\n"
            "If a value is missing, write null. "
            "Do not return JSON. Do not add explanation."
        )
    return "Extract the document into a JSON object with keys `document_type` and `fields`."


def prepare_image_for_training(image: Image.Image) -> Image.Image:
    prepared = image.copy()
    prepared.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), Image.Resampling.LANCZOS)
    return prepared


@dataclass
class EncodedSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    mm_token_type_ids: Optional[torch.Tensor]
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    labels: torch.Tensor
    target_text: str


class MultiDocDataset(Dataset):
    def __init__(self, rows: list[dict], processor, max_length: int) -> None:
        self.items: list[EncodedSample] = []
        self.processor = processor

        for row in tqdm(rows, desc="Encoding dataset"):
            image_path = Path(row["image_path"])
            image = prepare_image_for_training(Image.open(image_path).convert("RGB"))
            target_text = build_target(row["document_type"], row["fields"])

            user_message = {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {
                        "type": "text",
                        "text": (
                            build_instruction(row["document_type"])
                        ),
                    },
                ],
            }
            prompt_only = processor.apply_chat_template(
                [user_message],
                tokenize=False,
                add_generation_prompt=True,
            )
            full_prompt = processor.apply_chat_template(
                [
                    user_message,
                    {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "text",
                                "text": target_text,
                            }
                        ],
                    },
                ],
                tokenize=False,
                add_generation_prompt=False,
            )

            encoded_prompt = processor(
                text=prompt_only,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            encoded_inputs = processor(
                text=full_prompt,
                images=image,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
            )
            prompt_token_count = encoded_prompt["input_ids"].shape[1]
            full_token_count = encoded_inputs["input_ids"].shape[1]
            if prompt_token_count >= max_length:
                raise ValueError(
                    f"Prompt length {prompt_token_count} reaches MAX_LENGTH={max_length}. "
                    "Increase MAX_LENGTH for PaddleOCR-VL training."
                )
            encoded_labels = encoded_inputs["input_ids"].squeeze(0).clone()
            encoded_labels[:prompt_token_count] = -100
            encoded_labels[encoded_labels == processor.tokenizer.pad_token_id] = -100

            self.items.append(
                EncodedSample(
                    input_ids=encoded_inputs["input_ids"].squeeze(0),
                    attention_mask=encoded_inputs["attention_mask"].squeeze(0),
                    mm_token_type_ids=(
                        encoded_inputs["mm_token_type_ids"].squeeze(0)
                        if "mm_token_type_ids" in encoded_inputs
                        else None
                    ),
                    pixel_values=encoded_inputs["pixel_values"].squeeze(0),
                    image_grid_thw=encoded_inputs["image_grid_thw"].squeeze(0),
                    labels=encoded_labels,
                    target_text=target_text,
                )
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        item = self.items[index]
        return {
            "input_ids": item.input_ids,
            "attention_mask": item.attention_mask,
            "mm_token_type_ids": item.mm_token_type_ids,
            "pixel_values": item.pixel_values,
            "image_grid_thw": item.image_grid_thw,
            "labels": item.labels,
            "target_text": item.target_text,
        }


def collate_fn(batch: list[dict]) -> dict:
    collated = {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        # PaddleOCR-VL keeps visual tokens as a flat patch sequence tensor.
        # We concatenate patch sequences and keep per-sample layout in image_grid_thw.
        "pixel_values": torch.cat([item["pixel_values"] for item in batch], dim=0),
        "image_grid_thw": torch.stack([item["image_grid_thw"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
        "target_text": [item["target_text"] for item in batch],
    }
    if all(item["mm_token_type_ids"] is not None for item in batch):
        collated["mm_token_type_ids"] = torch.stack([item["mm_token_type_ids"] for item in batch])
    return collated


def build_model_inputs(batch: dict, device: torch.device) -> dict:
    model_inputs = {
        "input_ids": batch["input_ids"].to(device, non_blocking=True),
        "attention_mask": batch["attention_mask"].to(device, non_blocking=True),
        "pixel_values": batch["pixel_values"].to(device, non_blocking=True),
        "image_grid_thw": batch["image_grid_thw"].to(device, non_blocking=True),
        "labels": batch["labels"].to(device, non_blocking=True),
    }
    mm_token_type_ids = batch.get("mm_token_type_ids")
    if mm_token_type_ids is not None:
        model_inputs["mm_token_type_ids"] = mm_token_type_ids.to(device, non_blocking=True)
    return model_inputs


@torch.no_grad()
def evaluate_loss(model, dataloader, device, use_amp) -> float:
    model.eval()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Validation", leave=False):
        model_inputs = build_model_inputs(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(**model_inputs)
        total_loss += outputs.loss.item()
    return total_loss / max(len(dataloader), 1)


def _apply_transformers_patches() -> None:
    """Apply 3 patches required for PaddleOCR-VL + transformers 5.x compatibility."""
    # Patch 1: ROPE_INIT_FUNCTIONS["default"] is missing in transformers 5.x
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        if "default" not in ROPE_INIT_FUNCTIONS:
            import torch as _torch

            def _rope_default_init(config, device=None, seq_len=None, **kwargs):
                base = getattr(config, "rope_theta", 10000.0)
                dim = getattr(config, "head_dim", None) or (
                    config.hidden_size // config.num_attention_heads
                )
                inv_freq = 1.0 / (
                    base ** (_torch.arange(0, dim, 2, dtype=_torch.float32) / dim)
                )
                if device is not None:
                    inv_freq = inv_freq.to(device)
                return inv_freq, None

            ROPE_INIT_FUNCTIONS["default"] = _rope_default_init
            print("[patch] ROPE_INIT_FUNCTIONS['default'] injected")
    except Exception as _e:
        print(f"[patch] ROPE patch skipped: {_e}")

    # Patch 2: create_causal_mask signature changed in transformers 5.x
    try:
        import transformers.modeling_attn_mask_utils as _mamu

        if hasattr(_mamu, "create_causal_mask"):
            import inspect as _inspect
            _sig = _inspect.signature(_mamu.create_causal_mask)
            if "past_key_values_length" not in _sig.parameters:
                _orig_ccm = _mamu.create_causal_mask

                def _patched_ccm(*args, past_key_values_length=0, **kwargs):
                    return _orig_ccm(*args, **kwargs)

                _mamu.create_causal_mask = _patched_ccm

                import transformers.models.qwen2_vl.modeling_qwen2_vl as _qwen2
                if hasattr(_qwen2, "create_causal_mask"):
                    _qwen2.create_causal_mask = _patched_ccm

                print("[patch] create_causal_mask patched")
    except Exception as _e:
        print(f"[patch] causal_mask patch skipped: {_e}")

    # Patch 3: _init_weights calls compute_default_rope_parameters which doesn't exist
    try:
        import transformers.modeling_utils as _tmu
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS as _ROPE

        def _rope_default_init_fn(config, device=None, **kw):
            import torch as _torch
            base = getattr(config, "rope_theta", 10000.0)
            dim = getattr(config, "head_dim", None) or (
                config.hidden_size // config.num_attention_heads
            )
            inv_freq = 1.0 / (
                base ** (_torch.arange(0, dim, 2, dtype=_torch.float32) / dim)
            )
            if device is not None:
                inv_freq = inv_freq.to(device)
            return inv_freq, None

        _orig_init_weights = _tmu.PreTrainedModel._init_weights

        def _patched_init_weights(self, module):
            if (
                hasattr(module, "rope_type")
                and module.rope_type == "default"
                and not hasattr(module, "compute_default_rope_parameters")
            ):
                module.compute_default_rope_parameters = lambda config, device=None, **kw: _rope_default_init_fn(config, device=device)
            return _orig_init_weights(self, module)

        _tmu.PreTrainedModel._init_weights = _patched_init_weights
        print("[patch] _init_weights patched (compute_default_rope_parameters)")
    except Exception as _e:
        print(f"[patch] _init_weights patch skipped: {_e}")


def main() -> None:
    configure_env()
    _apply_transformers_patches()
    seed_everything(SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = torch.cuda.is_available()
    print("Device:", device)
    print("Model :", MODEL_NAME)
    print("Train :", TRAIN_FILE)
    print("Val   :", VAL_FILE)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=str(HF_CACHE))
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(HF_CACHE),
        trust_remote_code=True,
        attn_implementation="eager",
    )
    if BASE_ADAPTER_DIR:
        print("Resume adapter:", BASE_ADAPTER_DIR)
        model = PeftModel.from_pretrained(
            model,
            BASE_ADAPTER_DIR,
            is_trainable=True,
            local_files_only=True,
        )
    else:
        lora_config = LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    model = model.to(device)
    model.print_trainable_parameters()

    train_rows = load_jsonl(TRAIN_FILE)
    val_rows = load_jsonl(VAL_FILE)
    if MAX_TRAIN_ROWS > 0:
        train_rows = train_rows[:MAX_TRAIN_ROWS]
    if MAX_VAL_ROWS > 0:
        val_rows = val_rows[:MAX_VAL_ROWS]
    if not train_rows:
        raise SystemExit(f"Training set is empty: {TRAIN_FILE}")
    if not val_rows:
        print(f"Warning: validation set is empty -> {VAL_FILE}")
    train_dataset = MultiDocDataset(train_rows, processor, MAX_LENGTH)
    val_dataset = MultiDocDataset(val_rows, processor, MAX_LENGTH)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    if DRY_RUN:
        print("Dry run mode: validating one forward pass without optimizer steps.")
        batch = next(iter(train_loader))
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                print(f"Dry run batch {key}: {tuple(value.shape)}")
        model_inputs = build_model_inputs(batch, device)
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(**model_inputs)
        print(f"Dry run loss: {outputs.loss.item():.4f}")
        print("Dry run completed successfully.")
        return

    optimizer = AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    num_update_steps_per_epoch = max(
        1,
        (len(train_loader) + GRADIENT_ACCUMULATION_STEPS - 1)
        // GRADIENT_ACCUMULATION_STEPS,
    )
    total_training_steps = NUM_EPOCHS * num_update_steps_per_epoch
    warmup_steps = int(total_training_steps * WARMUP_RATIO)
    scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_training_steps,
    )

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state_dict = None
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(NUM_EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        total_train_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        for step, batch in enumerate(progress):
            batch_start = time.perf_counter()
            model_inputs = build_model_inputs(batch, device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(**model_inputs)
                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

            real_loss = loss.item() * GRADIENT_ACCUMULATION_STEPS
            total_train_loss += real_loss
            progress.set_postfix(loss=real_loss, lr=optimizer.param_groups[0]["lr"])

            if step < 2:
                print(
                    f"Step {step + 1}: total={time.perf_counter() - batch_start:.2f}s "
                    f"gpu={'on' if torch.cuda.is_available() else 'off'}"
                )

        avg_train_loss = total_train_loss / max(len(train_loader), 1)
        avg_val_loss = evaluate_loss(model, val_loader, device, use_amp)
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(f"Epoch {epoch + 1}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Val loss:   {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = {
                key: value.detach().cpu().clone() for key, value in model.state_dict().items()
            }
            model.save_pretrained(OUTPUT_DIR)
            processor.save_pretrained(OUTPUT_DIR)
            save_training_summary(
                output_dir=OUTPUT_DIR,
                best_val_loss=best_val_loss,
                history=history,
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    processor.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)
    save_training_summary(
        output_dir=OUTPUT_DIR,
        best_val_loss=best_val_loss,
        history=history,
    )
    print("Saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
