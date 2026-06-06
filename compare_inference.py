import json
import os
import random
import re
from pathlib import Path

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
FINETUNED_MODEL_PATH = os.getenv(
    "FINETUNED_MODEL_PATH", str(Path("E:/thesis/outputs/Bennet1996_donut-small_ft_best"))
)
TASK_TOKEN = "<s_invoice>"
GT_FIELD = "ground_truth"
INCLUDE_ITEMS = os.getenv("INCLUDE_ITEMS", "false").lower() == "true"
IMAGE_WIDTH = int(os.getenv("IMAGE_WIDTH", "640"))
IMAGE_HEIGHT = int(os.getenv("IMAGE_HEIGHT", "480"))
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "256"))
NUM_SAMPLES = int(os.getenv("NUM_SAMPLES", "5"))
SEED = int(os.getenv("SEED", "42"))
OUTPUT_PATH = Path(os.getenv("OUTPUT_PATH", str(Path("E:/thesis/inference_comparison.json"))))


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


def serialize_ground_truth(gt_raw):
    if isinstance(gt_raw, str):
        gt_obj = json.loads(gt_raw)
    elif isinstance(gt_raw, dict):
        gt_obj = gt_raw
    else:
        raise TypeError(f"Unexpected ground_truth type: {type(gt_raw)}")

    gt_parse = simplify_gt_parse(gt_obj["gt_parse"], include_items=INCLUDE_ITEMS)
    tagged = dict_to_donut_tags(gt_parse)
    if not tagged.strip():
        raise ValueError("ground truth is empty after serialization")
    return tagged, gt_parse


def prepare_item(item):
    image = item["image"]
    if not isinstance(image, Image.Image):
        raise TypeError(f"Expected PIL.Image, got {type(image)}")
    tagged_text, gt_parse = serialize_ground_truth(item[GT_FIELD])
    return {
        "image": image.convert("RGB"),
        "target_text": TASK_TOKEN + tagged_text,
        "gt_parse": gt_parse,
    }


def clean_prediction_text(text):
    for token in ["<pad>", "</s>", "<s>", "<unk>"]:
        text = text.replace(token, "")
    return text.strip()


def normalize_for_match(text):
    return " ".join(clean_prediction_text(text).split()).strip().lower()


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


def extract_german_tag(text, tag):
    return extract_field(text, tag)


def normalize_amount(value):
    if value is None:
        return None
    text = str(value).strip()
    text = re.sub(r"\s+", " ", text)
    return text


def format_german_amount(value):
    if value is None:
        return None
    value = normalize_amount(value)
    match = re.search(r"(\d{1,3}(?:\.\d{3})*(?:,\d{2})|\d+(?:,\d{2}))", value)
    if not match:
        return None
    return f"$ {match.group(1)}"


def format_space_date(value):
    if value is None:
        return None
    parts = re.findall(r"\d{2,4}", value)
    if len(parts) < 3:
        return None
    if len(parts[2]) == 2:
        parts[2] = f"20{parts[2]}"
    return f"{parts[0]}/{parts[1]}/{parts[2]}"


def infer_schema_from_text(text):
    cleaned = clean_prediction_text(text)
    pred = {
        "invoice_no": extract_field(cleaned, "invoice_no"),
        "invoice_date": extract_field(cleaned, "invoice_date"),
        "seller": extract_field(cleaned, "seller"),
        "client": extract_field(cleaned, "client"),
        "seller_tax_id": extract_field(cleaned, "seller_tax_id"),
        "client_tax_id": extract_field(cleaned, "client_tax_id"),
        "iban": extract_field(cleaned, "iban"),
        "total_net_worth": extract_field(cleaned, "total_net_worth"),
        "total_vat": extract_field(cleaned, "total_vat"),
        "total_gross_worth": extract_field(cleaned, "total_gross_worth"),
    }
    if any(value is not None for value in pred.values()):
        return pred

    all_amounts = re.findall(r"\d{1,3}(?:\.\d{3})*(?:,\d{2})|\d+(?:,\d{2})", cleaned)
    german_name = extract_german_tag(cleaned, "Name")
    german_street = extract_german_tag(cleaned, "Straße")
    german_city = extract_german_tag(cleaned, "Stadt")
    german_birth = extract_german_tag(cleaned, "Geburtsdatum")
    german_net = extract_german_tag(cleaned, "Gesetzl. Netto")
    german_gross = extract_german_tag(cleaned, "Gesamtbrutto")
    fallback_iban = re.search(r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b", cleaned)
    fallback_tax_id = re.findall(r"\b\d{3}-\d{2}-\d{4}\b", cleaned)
    fallback_invoice_no = re.search(r"\b\d{6,10}\b", cleaned)

    return {
        "invoice_no": fallback_invoice_no.group(0) if fallback_invoice_no else None,
        "invoice_date": format_space_date(german_birth),
        "seller": " ".join(part for part in [german_name, german_street, german_city] if part) or None,
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


def schema_to_donut_tags(pred):
    header_fields = [
        "invoice_no",
        "invoice_date",
        "seller",
        "client",
        "seller_tax_id",
        "client_tax_id",
        "iban",
    ]
    summary_fields = ["total_net_worth", "total_vat", "total_gross_worth"]

    header = "".join(
        f"<s_{field}>{pred[field]}</s_{field}>" for field in header_fields if pred.get(field) is not None
    )
    summary = "".join(
        f"<s_{field}>{pred[field]}</s_{field}>" for field in summary_fields if pred.get(field) is not None
    )
    return f"{TASK_TOKEN}<s_header>{header}</s_header><s_summary>{summary}</s_summary>"


def load_valid_samples():
    local_cache_dir = Path(LOCAL_DATASET_CACHE_DIR)
    if local_cache_dir.exists():
        dataset = ArrowDataset.from_file(
            str(local_cache_dir / "invoices-donut-data-v1-validation.arrow")
        )
        print(f"Loaded validation split from local arrow cache: {local_cache_dir}")
    else:
        dataset = load_dataset(DATASET_ID)["validation"]
    prepared = []
    for idx in range(len(dataset)):
        try:
            item = prepare_item(dataset[idx])
            item["index"] = idx
            prepared.append(item)
        except Exception:
            continue
    return prepared


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
        early_stopping=True,
        pad_token_id=processor.tokenizer.pad_token_id,
        eos_token_id=processor.tokenizer.eos_token_id,
        use_cache=True,
        num_beams=1,
        bad_words_ids=[[processor.tokenizer.unk_token_id]],
        return_dict_in_generate=True,
    )
    text = processor.batch_decode(outputs.sequences, skip_special_tokens=False)[0]
    return text


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


def main():
    random.seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samples = load_valid_samples()
    chosen = random.sample(samples, min(NUM_SAMPLES, len(samples)))

    base_processor, base_model = load_processor_and_model(BASE_MODEL_NAME, device)
    finetuned_processor, finetuned_model = load_processor_and_model(FINETUNED_MODEL_PATH, device)

    results = {
        "base_model": BASE_MODEL_NAME,
        "finetuned_model": FINETUNED_MODEL_PATH,
        "num_samples": len(chosen),
        "samples": [],
    }

    for sample in tqdm(chosen, desc="Comparing inference"):
        gt_text = sample["target_text"]
        base_text = predict(base_model, base_processor, sample["image"], device)
        finetuned_text = predict(finetuned_model, finetuned_processor, sample["image"], device)
        base_schema = infer_schema_from_text(base_text)
        finetuned_schema = infer_schema_from_text(finetuned_text)
        base_formatted = schema_to_donut_tags(base_schema)
        finetuned_formatted = schema_to_donut_tags(finetuned_schema)

        results["samples"].append(
            {
                "dataset_index": sample["index"],
                "ground_truth": gt_text,
                "base_raw_prediction": base_text,
                "base_formatted_prediction": base_formatted,
                "base_prediction_schema": base_schema,
                "finetuned_raw_prediction": finetuned_text,
                "finetuned_formatted_prediction": finetuned_formatted,
                "finetuned_prediction_schema": finetuned_schema,
                "base_exact_match": normalize_for_match(base_formatted) == normalize_for_match(gt_text),
                "finetuned_exact_match": normalize_for_match(finetuned_formatted)
                == normalize_for_match(gt_text),
            }
        )

    base_matches = sum(1 for x in results["samples"] if x["base_exact_match"])
    finetuned_matches = sum(1 for x in results["samples"] if x["finetuned_exact_match"])
    results["summary"] = {
        "base_exact_matches": base_matches,
        "finetuned_exact_matches": finetuned_matches,
    }

    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved comparison to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
