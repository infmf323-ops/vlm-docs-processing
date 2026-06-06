import json
import os
from pathlib import Path
import re
from difflib import SequenceMatcher

import torch
from datasets import load_dataset
from datasets.arrow_dataset import Dataset as ArrowDataset
from peft import PeftModel
from PIL import Image
from tqdm.auto import tqdm
from transformers import DonutProcessor, VisionEncoderDecoderModel


DATASET_ID = os.getenv("DATASET_ID", "katanaml-org/invoices-donut-data-v1")
LOCAL_DATASET_CACHE_DIR = os.getenv(
    "LOCAL_DATASET_CACHE_DIR",
    "C:/Users/wasd/.cache/huggingface/datasets/katanaml-org___invoices-donut-data-v1/default/0.0.0/d2cde298e79c94fb05bc320999deb4b7889b0464",
)
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "Bennet1996/donut-small")
FULL_MODEL_PATH = os.getenv(
    "FULL_MODEL_PATH", str(Path("E:/thesis/outputs/Bennet1996_donut-small_ft_best"))
)
DORA_MODEL_PATH = os.getenv(
    "DORA_MODEL_PATH", str(Path("E:/thesis/outputs/Bennet1996_donut-small_dora_best"))
)
TASK_TOKEN = "<s_invoice>"
GT_FIELD = "ground_truth"
INCLUDE_ITEMS = os.getenv("INCLUDE_ITEMS", "false").lower() == "true"
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "480"))
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
OUTPUT_PATH = Path(
    os.getenv("OUTPUT_PATH", str(Path("E:/thesis/validation_model_comparison.json")))
)

FIELDS = [
    "invoice_no",
    "invoice_date",
    "seller",
    "client",
    "seller_tax_id",
    "client_tax_id",
    "iban",
    "total_net_worth",
    "total_vat",
    "total_gross_worth",
]


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

    if "summary" in gt_parse and isinstance(gt_parse["summary"], dict):
        out["summary"] = gt_parse["summary"]
    else:
        summary_keys = ["total_net_worth", "total_vat", "total_gross_worth"]
        summary = {k: gt_parse[k] for k in summary_keys if k in gt_parse}
        if summary:
            out["summary"] = summary

    return out


def serialize_ground_truth(gt_raw):
    if isinstance(gt_raw, str):
        gt_obj = json.loads(gt_raw)
    elif isinstance(gt_raw, dict):
        gt_obj = gt_raw
    else:
        raise TypeError(f"Unexpected ground_truth type: {type(gt_raw)}")

    gt_parse = simplify_gt_parse(gt_obj["gt_parse"], include_items=INCLUDE_ITEMS)
    if not gt_parse:
        raise ValueError("ground truth is empty after serialization")
    return gt_parse


def prepare_item(item):
    image = item["image"]
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(image)}")
    gt_parse = serialize_ground_truth(item[GT_FIELD])
    return {
        "image": image.convert("RGB"),
        "gt_parse": gt_parse,
    }


def clean_prediction_text(text):
    for token in ["<pad>", "</s>", "<s>", "<unk>"]:
        text = text.replace(token, "")
    return text.strip()


def extract_field(text, field):
    open_tag = f"<s_{field}>"
    close_tag = f"</s_{field}>"
    start = text.find(open_tag)
    if start == -1:
        return None
    start += len(open_tag)
    end = text.find(close_tag, start)
    if end == -1:
        return None
    return text[start:end].strip()


def format_space_date(value):
    if value is None:
        return None
    parts = re.findall(r"\d{2,4}", value)
    if len(parts) < 3:
        return None
    if len(parts[2]) == 2:
        parts[2] = f"20{parts[2]}"
    return f"{parts[0]}/{parts[1]}/{parts[2]}"


def format_german_amount(value):
    if value is None:
        return None
    match = re.search(r"(\d{1,3}(?:\.\d{3})*(?:,\d{2})|\d+(?:,\d{2}))", str(value))
    if not match:
        return None
    return f"$ {match.group(1)}"


def infer_schema_from_text(text):
    cleaned = clean_prediction_text(text)
    pred = {field: extract_field(cleaned, field) for field in FIELDS}
    if any(value is not None for value in pred.values()):
        return pred

    all_amounts = re.findall(r"\d{1,3}(?:\.\d{3})*(?:,\d{2})|\d+(?:,\d{2})", cleaned)
    fallback_tax_id = re.findall(r"\b\d{3}-\d{2}-\d{4}\b", cleaned)
    fallback_iban = re.search(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b", cleaned)
    fallback_invoice_no = re.search(r"\b\d{6,10}\b", cleaned)
    name = extract_field(cleaned, "Name")
    street = extract_field(cleaned, "Straße")
    city = extract_field(cleaned, "Stadt")
    birth = extract_field(cleaned, "Geburtsdatum")
    german_net = extract_field(cleaned, "Gesetzl. Netto")
    german_gross = extract_field(cleaned, "Gesamtbrutto")

    return {
        "invoice_no": fallback_invoice_no.group(0) if fallback_invoice_no else None,
        "invoice_date": format_space_date(birth),
        "seller": " ".join(part for part in [name, street, city] if part) or None,
        "client": None,
        "seller_tax_id": fallback_tax_id[0] if len(fallback_tax_id) > 0 else None,
        "client_tax_id": fallback_tax_id[1] if len(fallback_tax_id) > 1 else None,
        "iban": fallback_iban.group(0) if fallback_iban else None,
        "total_net_worth": format_german_amount(german_net or (all_amounts[0] if all_amounts else None)),
        "total_vat": None,
        "total_gross_worth": format_german_amount(
            german_gross or (all_amounts[1] if len(all_amounts) > 1 else None)
        ),
    }


def flatten_gt(gt_parse):
    flat = {}
    header = gt_parse.get("header", {})
    summary = gt_parse.get("summary", {})
    for key in [
        "invoice_no",
        "invoice_date",
        "seller",
        "client",
        "seller_tax_id",
        "client_tax_id",
        "iban",
    ]:
        flat[key] = header.get(key)
    for key in ["total_net_worth", "total_vat", "total_gross_worth"]:
        flat[key] = summary.get(key)
    return flat


def normalize_text(value):
    if value is None:
        return None
    return " ".join(str(value).strip().lower().split())


def normalize_amount(value):
    if value is None:
        return None
    text = normalize_text(value)
    text = text.replace("$", "").replace(" ", "")
    return text


def parse_amount(value):
    if value is None:
        return None
    text = normalize_amount(value)
    if text is None:
        return None
    text = text.replace(".", "").replace(",", ".")
    try:
        return float(text)
    except ValueError:
        return None


def digit_similarity(pred, gt):
    pred_digits = "".join(ch for ch in str(pred) if ch.isdigit())
    gt_digits = "".join(ch for ch in str(gt) if ch.isdigit())
    if not pred_digits or not gt_digits:
        return 0.0
    return SequenceMatcher(None, pred_digits, gt_digits).ratio()


def token_similarity(pred, gt):
    pred_tokens = set(normalize_text(pred).split()) if pred is not None else set()
    gt_tokens = set(normalize_text(gt).split()) if gt is not None else set()
    if not pred_tokens or not gt_tokens:
        return 0.0
    inter = len(pred_tokens & gt_tokens)
    return (2 * inter) / (len(pred_tokens) + len(gt_tokens))


def date_similarity(pred, gt):
    pred_parts = re.findall(r"\d{1,4}", str(pred))
    gt_parts = re.findall(r"\d{1,4}", str(gt))
    if not pred_parts or not gt_parts:
        return 0.0
    if len(pred_parts) >= 3 and len(gt_parts) >= 3:
        matches = sum(1 for p, g in zip(pred_parts[:3], gt_parts[:3]) if p == g)
        return matches / 3
    return SequenceMatcher(None, "/".join(pred_parts), "/".join(gt_parts)).ratio()


def amount_similarity(pred, gt):
    pred_value = parse_amount(pred)
    gt_value = parse_amount(gt)
    if pred_value is None or gt_value is None:
        return 0.0
    if gt_value == 0:
        return 1.0 if pred_value == 0 else 0.0
    rel_error = abs(pred_value - gt_value) / abs(gt_value)
    return max(0.0, 1.0 - rel_error)


def soft_field_score(field, pred, gt):
    if gt is None:
        return None
    if pred is None:
        return 0.0
    if field in {"invoice_no", "seller_tax_id", "client_tax_id", "iban"}:
        return digit_similarity(pred, gt)
    if field == "invoice_date":
        return date_similarity(pred, gt)
    if field in {"total_net_worth", "total_vat", "total_gross_worth"}:
        return amount_similarity(pred, gt)
    return token_similarity(pred, gt)


def field_matches(field, pred, gt):
    if gt is None:
        return None
    if field in {"total_net_worth", "total_vat", "total_gross_worth"}:
        return normalize_amount(pred) == normalize_amount(gt)
    return normalize_text(pred) == normalize_text(gt)


def predict(model, processor, image, device):
    pixel_values = processor(
        images=image.resize(IMAGE_SIZE),
        return_tensors="pt",
        legacy=False,
    ).pixel_values.to(device)

    decoder_input_ids = torch.tensor(
        [[processor.tokenizer.convert_tokens_to_ids(TASK_TOKEN)]], device=device
    )

    outputs = model.generate(
        pixel_values,
        decoder_input_ids=decoder_input_ids,
        max_length=MAX_LENGTH,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
    )
    return processor.batch_decode(outputs, skip_special_tokens=False)[0]


def load_processor_and_model(model_name_or_path, device):
    model_path = Path(model_name_or_path)
    processor = DonutProcessor.from_pretrained(model_name_or_path, use_fast=False)

    if model_path.exists() and (model_path / "adapter_config.json").exists():
        base_model_name = (model_path / "base_model_name.txt").read_text(encoding="utf-8").strip()
        model = VisionEncoderDecoderModel.from_pretrained(base_model_name)
        model.decoder.resize_token_embeddings(len(processor.tokenizer))
        model.decoder = PeftModel.from_pretrained(model.decoder, model_name_or_path)
        model.decoder = model.decoder.merge_and_unload()
    else:
        model = VisionEncoderDecoderModel.from_pretrained(model_name_or_path)

    return processor, model.to(device).eval()


def evaluate_model(model_name_or_path, prepared_items, device):
    processor, model = load_processor_and_model(model_name_or_path, device)

    metrics = {field: {"correct": 0, "total": 0} for field in FIELDS}
    soft_metrics = {field: {"score_sum": 0.0, "total": 0} for field in FIELDS}
    document_exact = 0
    document_soft_sum = 0.0
    examples = []

    for idx, item in enumerate(tqdm(prepared_items, desc=f"Evaluating {model_name_or_path}")):
        pred_text = clean_prediction_text(predict(model, processor, item["image"], device))
        gt_flat = flatten_gt(item["gt_parse"])
        pred_flat = infer_schema_from_text(pred_text)

        all_match = True
        doc_scores = []
        for field in FIELDS:
            match = field_matches(field, pred_flat[field], gt_flat[field])
            if match is None:
                continue
            metrics[field]["total"] += 1
            if match:
                metrics[field]["correct"] += 1
            else:
                all_match = False

            soft_score = soft_field_score(field, pred_flat[field], gt_flat[field])
            if soft_score is not None:
                soft_metrics[field]["score_sum"] += soft_score
                soft_metrics[field]["total"] += 1
                doc_scores.append(soft_score)

        if all_match:
            document_exact += 1
        if doc_scores:
            document_soft_sum += sum(doc_scores) / len(doc_scores)

        if len(examples) < 3:
            examples.append(
                {
                    "index": idx,
                    "ground_truth": gt_flat,
                    "prediction": pred_flat,
                    "soft_scores": {
                        field: soft_field_score(field, pred_flat[field], gt_flat[field])
                        for field in FIELDS
                    },
                }
            )

    field_accuracy = {}
    field_similarity = {}
    for field, stat in metrics.items():
        total = stat["total"]
        field_accuracy[field] = 0.0 if total == 0 else stat["correct"] / total
    for field, stat in soft_metrics.items():
        total = stat["total"]
        field_similarity[field] = 0.0 if total == 0 else stat["score_sum"] / total

    return {
        "model": str(model_name_or_path),
        "num_samples": len(prepared_items),
        "document_exact_match": document_exact / max(len(prepared_items), 1),
        "document_soft_score": document_soft_sum / max(len(prepared_items), 1),
        "field_accuracy": field_accuracy,
        "field_similarity": field_similarity,
        "examples": examples,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    local_cache_dir = Path(LOCAL_DATASET_CACHE_DIR)
    if local_cache_dir.exists():
        validation = ArrowDataset.from_file(
            str(local_cache_dir / "invoices-donut-data-v1-validation.arrow")
        )
        print(f"Loaded validation split from local arrow cache: {local_cache_dir}")
    else:
        validation = load_dataset(DATASET_ID)["validation"]

    prepared = []
    skipped = 0
    for item in validation:
        try:
            prepared.append(prepare_item(item))
        except Exception:
            skipped += 1

    base_result = evaluate_model(BASE_MODEL_NAME, prepared, device)
    full_result = evaluate_model(FULL_MODEL_PATH, prepared, device)
    dora_result = evaluate_model(DORA_MODEL_PATH, prepared, device)

    result = {
        "dataset": DATASET_ID,
        "num_valid_samples": len(prepared),
        "num_skipped_samples": skipped,
        "base": base_result,
        "full": full_result,
        "dora": dora_result,
    }
    OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved validation comparison to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
