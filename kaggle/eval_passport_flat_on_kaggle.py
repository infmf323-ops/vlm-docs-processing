from __future__ import annotations

import json
import os
import re
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "PaddlePaddle/PaddleOCR-VL"
HF_CACHE = ROOT / ".hf-cache"
DATASET_PATH = Path(
    os.getenv(
        "PASSPORT_EVAL_DATASET",
        str(ROOT / "data" / "multidoc" / "passport_eval_diverse_v1.jsonl"),
    )
)
OUTPUT_PATH = Path(os.getenv("PASSPORT_EVAL_OUTPUT", str(ROOT / "passport_curriculum_flat_eval_kaggle.json")))
ADAPTER_DIR = Path(
    os.getenv(
        "PASSPORT_ADAPTER_DIR",
        str(ROOT / "outputs" / "paddleocr_vl_passport_curriculum_flat_kaggle"),
    )
)
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "768"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "192"))
MAX_GENERATION_TIME = float(os.getenv("MAX_GENERATION_TIME", "300"))
PROJECT_ROOT = ROOT
COUNTRY_MAP = {
    # Full 3-letter ICAO codes → expected nationality string (matching ground truth)
    "RUS": "RUSSIAN FEDERATION",
    "USA": "UNITED STATES OF AMERICA",
    "JPN": "JAPAN",
    "FRA": "FRANCE",
    "DEU": "GERMANY",
    "TUR": "TURKEY",
    "GRC": "GREECE",
    "IND": "INDIA",
    "CHN": "CHINA",
    "GBR": "UNITED KINGDOM",
    "CAN": "CANADA",
    "AUS": "AUSTRALIA",
    "NLD": "NETHERLANDS",
    "BEL": "BELGIUM",
    "ESP": "SPAIN",
    "ITA": "ITALY",
    "PRT": "PORTUGAL",
    "CHE": "SWITZERLAND",
    "AUT": "AUSTRIA",
    "SWE": "SWEDEN",
    "NOR": "NORWAY",
    "FIN": "FINLAND",
    "DNK": "DENMARK",
    "POL": "POLAND",
    "CZE": "CZECH REPUBLIC",
    "HUN": "HUNGARY",
    "ARE": "UNITED ARAB EMIRATES",
    "BRA": "BRAZIL",
    "ARG": "ARGENTINA",
    "MEX": "MEXICO",
    "KOR": "KOREA",
    "SGP": "SINGAPORE",
    "MYS": "MALAYSIA",
    "THA": "THAILAND",
    "IDN": "INDONESIA",
    "PAK": "PAKISTAN",
    "BGD": "BANGLADESH",
    "NPL": "NEPAL",
    "ZAF": "SOUTH AFRICA",
    "NGA": "NIGERIA",
    "EGY": "EGYPT",
    # Single-letter codes: Germany uses "D" in MRZ (D<< stripped to D)
    "D": "GERMANY",
}
ALLOWED_FIELDS = {
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
}


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
    return raw_path


def prepare_image(image: Image.Image) -> Image.Image:
    prepared = image.convert("RGB")
    width, height = prepared.size
    longest = max(width, height)
    if longest <= MAX_IMAGE_SIDE:
        return prepared
    scale = MAX_IMAGE_SIDE / float(longest)
    return prepared.resize(
        (max(1, int(width * scale)), max(1, int(height * scale))),
        Image.Resampling.LANCZOS,
    )


def load_rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        row["image_path"] = resolve_image_path(str(row["image_path"]))
        rows.append(row)
    return rows


def flatten_scalars(value, prefix=""):
    out = {}
    if value is None:
        return out
    if isinstance(value, dict):
        for key, nested in value.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            out.update(flatten_scalars(nested, new_prefix))
        return out
    if isinstance(value, list):
        if value and all(not isinstance(item, (dict, list)) for item in value):
            out[prefix] = ",".join(str(item) for item in value)
        else:
            for index, nested in enumerate(value):
                out.update(flatten_scalars(nested, f"{prefix}[{index}]"))
        return out
    out[prefix] = str(value)
    return out


def compute_field_accuracy(expected, predicted):
    if not expected:
        return 0.0
    correct = 0
    for key, value in expected.items():
        if predicted.get(key) == value:
            correct += 1
    return correct / len(expected)


def compute_precision_recall_f1(expected, predicted):
    expected_items = set(expected.items())
    predicted_items = set(predicted.items())
    tp = len(expected_items & predicted_items)
    fp = len(predicted_items - expected_items)
    fn = len(expected_items - predicted_items)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return precision, recall, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def normalize_generic_text(value: str) -> str:
    cleaned = value.upper().strip()
    cleaned = cleaned.replace("’", "'")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def normalize_document_number(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize_generic_text(value))


def normalize_date_value(value: str) -> str:
    cleaned = normalize_generic_text(value)
    cleaned = cleaned.replace("/", ".").replace("-", ".")
    month_map = {
        "JAN": "01",
        "FEB": "02",
        "MAR": "03",
        "APR": "04",
        "MAY": "05",
        "JUN": "06",
        "JUL": "07",
        "AUG": "08",
        "SEP": "09",
        "OCT": "10",
        "NOV": "11",
        "DEC": "12",
    }
    month_match = re.match(r"^(\d{1,2}) ([A-Z]{3}) (\d{4})$", cleaned)
    if month_match:
        dd, mon, yyyy = month_match.groups()
        return f"{int(dd):02d}.{month_map[mon]}.{yyyy}"

    numeric_match = re.match(r"^(\d{1,2})\.(\d{1,2})\.(\d{4})$", cleaned)
    if numeric_match:
        a, b, yyyy = numeric_match.groups()
        return f"{int(a):02d}.{int(b):02d}.{yyyy}"

    return cleaned


def normalize_field_value(key: str, value: str) -> str:
    if key == "document_number":
        return normalize_document_number(value)
    if key in {"date_of_birth", "date_of_issue", "date_of_expiry"}:
        return normalize_date_value(value)
    if key == "mrz":
        return re.sub(r"\s+", "", normalize_generic_text(value))
    if key == "sex":
        return normalize_generic_text(value)[:1]
    if key in {"surname", "given_names", "nationality", "place_of_birth", "issuing_authority"}:
        normalized = normalize_generic_text(value)
        normalized = normalized.replace(",", " ")
        normalized = re.sub(r"[.]+", ".", normalized)
        normalized = re.sub(r"\s+", " ", normalized)
        return normalized.strip()
    return normalize_generic_text(value)


def normalize_flat_fields(fields: dict[str, str]) -> dict[str, str]:
    return {
        key: normalize_field_value(key, str(value))
        for key, value in fields.items()
        if value is not None
    }


def compute_field_match_breakdown(expected: dict[str, str], predicted: dict[str, str]) -> dict[str, dict[str, str | bool | None]]:
    breakdown: dict[str, dict[str, str | bool | None]] = {}
    for key, expected_value in expected.items():
        predicted_value = predicted.get(key)
        breakdown[key] = {
            "expected": expected_value,
            "predicted": predicted_value,
            "match": predicted_value == expected_value,
        }
    return breakdown


def build_prompt() -> str:
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


def clean_generated_text(text: str) -> str:
    cleaned = text.replace("班级：___", "")
    cleaned = cleaned.replace("班级", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def normalize_name_token(text: str) -> str:
    cleaned = text.replace("<", " ").replace(">", " ")
    cleaned = re.sub(r"[^A-Za-zА-Яа-яЁё' -]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.upper()


def format_mrz_date(compact: str) -> str | None:
    if len(compact) != 6 or not compact.isdigit():
        return None
    yy, mm, dd = compact[:2], compact[2:4], compact[4:6]
    year = int(yy)
    full_year = 1900 + year if year >= 50 else 2000 + year
    return f"{dd}.{mm}.{full_year}"


def should_recover_names_from_first_line(
    country: str | None,
    surname_token: str,
    given_tokens: list[str],
    payload: str,
) -> bool:
    if country:
        return True

    if not surname_token or not given_tokens:
        return False

    # Without a country code, be conservative and only trust short Arabic-style
    # family names such as `P<FARSI<<AHMAD<AL...`. This blocks hallucinated
    # synthetic identities like `P<MANSOURI<<LAYLA<<AL...` on unrelated samples.
    has_second_line_signal = bool(re.search(r"\d{6}[MF<]\d{6}", payload))
    if has_second_line_signal:
        return True

    last_given = given_tokens[-1] if given_tokens else ""
    if last_given not in {"AL", "EL"}:
        return False

    return len(surname_token) <= 6


def parse_mrz_like(text: str) -> dict:
    raw = clean_generated_text(text)
    upper_raw = raw.upper()

    # Locate the MRZ first-line start.  Standard passports use "P<NNN..." but
    # German passports use "PPD<<..." (P=passport, P=personal, D=Germany code).
    # Try each marker in preference order; skip if not found.
    payload: str | None = None
    for marker, skip in [("P<", 2), ("PPD<<", 5), ("PP<", 3)]:
        idx = upper_raw.find(marker)
        if idx != -1:
            payload = upper_raw[idx + skip :].replace("><", "<<")
            payload = re.sub(r"[^A-Z0-9<]", "", payload)
            break

    if not payload:
        return {}

    sections = [section for section in payload.split("<<") if section]
    if not sections:
        return {}

    first = sections[0]
    # Extract 3-char country code if recognisable.
    country_candidate = first[:3] if len(first) >= 3 else ""
    country = country_candidate if country_candidate in COUNTRY_MAP else None
    surname_token = first[3:] if country else first

    # Given names live only in sections[1] (between first "<<" and next "<<").
    # Iterating over sections[1:] would pull in second-line data: doc numbers,
    # nationality codes, date fragments — which contaminates given_names badly.
    given_tokens: list[str] = []
    if len(sections) > 1:
        for token in sections[1].split("<"):
            token = token.strip()
            if token:
                given_tokens.append(token)

    # Drop short alnum garbage tokens like `Z4` that often appear after a
    # mostly-correct MRZ fragment and otherwise block Arabic-style recovery.
    given_tokens = [
        token
        for token in given_tokens
        if not any(ch.isdigit() for ch in token) or len(token) > 2
    ]

    base_surname_token = surname_token
    if given_tokens and surname_token:
        arabic_prefix = given_tokens[-1]
        if arabic_prefix in {"AL", "EL"} and not surname_token.startswith(f"{arabic_prefix} "):
            surname_token = f"{arabic_prefix} {surname_token}"

    result: dict[str, str] = {}
    if should_recover_names_from_first_line(country, base_surname_token, given_tokens, payload):
        surname = normalize_name_token(surname_token)
        if surname:
            result["surname"] = surname

        given_names = normalize_name_token(" ".join(given_tokens))
        if given_names:
            result["given_names"] = given_names

    if country:
        result["nationality"] = COUNTRY_MAP[country]

    # Parse the second MRZ line using strict ICAO 9303 field widths:
    #   doc_number(9) + check_digit(1) + nationality(3) + DOB(6) + check(1)
    #   + sex(1) + expiry(6) + [remaining]
    # Using {9} for doc_number (not {8,10}) prevents the check digit from being
    # absorbed into the document number for alphanumeric passport numbers.
    # Nationality uses [A-Z<]{3} to handle Germany's "D<<" ICAO code.
    second_line = re.search(
        r"([A-Z0-9<]{9})[0-9<]([A-Z<]{3})(\d{6})[0-9<]([MF<])(\d{6})",
        payload,
    )
    if second_line:
        result["mrz"] = raw

        doc_num = second_line.group(1).replace("<", "")
        if doc_num and any(ch.isdigit() for ch in doc_num):
            result["document_number"] = doc_num

        # Nationality: strip fill chars, then look up (handles "D<<" → "D" → "GERMANY")
        nat_raw = second_line.group(2)
        nat_key = nat_raw.replace("<", "")
        nat_value = COUNTRY_MAP.get(nat_key) or COUNTRY_MAP.get(nat_raw)
        if nat_value and "nationality" not in result:
            result["nationality"] = nat_value

        date_of_birth = format_mrz_date(second_line.group(3))
        if date_of_birth:
            result["date_of_birth"] = date_of_birth

        sex = second_line.group(4).replace("<", "")
        if sex in {"M", "F"}:
            result["sex"] = sex

        date_of_expiry = format_mrz_date(second_line.group(5))
        if date_of_expiry:
            result["date_of_expiry"] = date_of_expiry

    return result


def parse_non_mrz_fallback(text: str) -> dict:
    """Last-resort heuristic extraction when no MRZ-like content is found."""
    raw = clean_generated_text(text)
    result: dict[str, str] = {}
    upper = raw.upper()

    # Russian Cyrillic OCR artifacts that indicate a Russian passport
    if any(hint in upper for hint in ("POCC", "POCCH", "FCEDEPAIIA", "POCCИЙСКАЯ")):
        result["nationality"] = "RUSSIAN FEDERATION"

    # Try to recover document number from flat-format output lines like
    # "document_number: AB1234567" even when MRZ was not found.
    flat_match = re.search(r"DOCUMENT[_\s]NUMBER\s*:\s*([A-Z0-9]{6,12})", upper)
    if flat_match:
        result["document_number"] = flat_match.group(1)
    else:
        # Generic alphanumeric passport-number pattern as last resort
        passport_number = re.search(r"\b([A-Z]{1,2}\d{7,8})\b", upper)
        if passport_number:
            result["document_number"] = passport_number.group(1)

    # Try to recover sex from flat output
    if "document_number" in result or "nationality" in result:
        sex_match = re.search(r"\bSEX\s*:\s*([MF])\b", upper)
        if sex_match:
            result["sex"] = sex_match.group(1)

    return result


def parse_flat_passport_output(text: str) -> dict:
    fields: dict[str, str] = {}
    for line in text.splitlines():
        match = re.match(r"^\s*([a-z_]+)\s*:\s*(.*)\s*$", line.strip(), flags=re.I)
        if not match:
            continue
        key = match.group(1).lower()
        value = match.group(2).strip()
        if key not in ALLOWED_FIELDS:
            continue
        if value.lower() == "null" or value == "":
            continue
        fields[key] = value

    recovered = parse_mrz_like(text)
    if not recovered:
        recovered = parse_non_mrz_fallback(text)

    for key, value in recovered.items():
        if key not in fields:
            fields[key] = value
    return fields


def main() -> None:
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    processor = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=str(HF_CACHE), local_files_only=True)
    tokenizer = processor.tokenizer
    base_model = AutoModelForImageTextToText.from_pretrained(MODEL_NAME, cache_dir=str(HF_CACHE), local_files_only=True)
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR), local_files_only=True)
    model = model.to(device)
    model.eval()

    rows = load_rows(DATASET_PATH)
    prompt_text = build_prompt()

    samples = []
    sum_acc = sum_prec = sum_rec = sum_f1 = 0.0
    sum_norm_acc = sum_norm_prec = sum_norm_rec = sum_norm_f1 = 0.0

    for i, row in enumerate(rows, 1):
        print(f"[{i}/{len(rows)}] {row['id']}", flush=True)
        image = prepare_image(Image.open(row["image_path"]).convert("RGB"))
        prompt = processor.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt_text},
                    ],
                }
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        encoded = processor(text=prompt, images=image, return_tensors="pt")
        encoded = {k: v.to(device) for k, v in encoded.items()}

        generation_kwargs = {
            "max_new_tokens": MAX_NEW_TOKENS,
            "max_time": MAX_GENERATION_TIME,
            "do_sample": False,
            "repetition_penalty": 1.3,
            "no_repeat_ngram_size": 4,
        }
        if tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if tokenizer.unk_token_id is not None:
            generation_kwargs["bad_words_ids"] = [[tokenizer.unk_token_id]]

        with torch.no_grad():
            output_ids = model.generate(**encoded, **generation_kwargs)

        generated_text = processor.batch_decode(
            output_ids[:, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0].strip()

        predicted_fields = parse_flat_passport_output(generated_text)

        expected_flat = flatten_scalars(row["fields"])
        predicted_flat = flatten_scalars(predicted_fields)
        acc = compute_field_accuracy(expected_flat, predicted_flat)
        prec, rec, f1 = compute_precision_recall_f1(expected_flat, predicted_flat)
        normalized_expected_flat = normalize_flat_fields(expected_flat)
        normalized_predicted_flat = normalize_flat_fields(predicted_flat)
        norm_acc = compute_field_accuracy(normalized_expected_flat, normalized_predicted_flat)
        norm_prec, norm_rec, norm_f1 = compute_precision_recall_f1(normalized_expected_flat, normalized_predicted_flat)

        sum_acc += acc
        sum_prec += prec
        sum_rec += rec
        sum_f1 += f1
        sum_norm_acc += norm_acc
        sum_norm_prec += norm_prec
        sum_norm_rec += norm_rec
        sum_norm_f1 += norm_f1

        samples.append(
            {
                "id": row["id"],
                "expected_fields": row["fields"],
                "predicted_fields": predicted_fields,
                "generated_text": generated_text,
                "metrics": {
                    "field_accuracy": acc,
                    "precision": prec,
                    "recall": rec,
                    "f1_score": f1,
                },
                "normalized_metrics": {
                    "field_accuracy": norm_acc,
                    "precision": norm_prec,
                    "recall": norm_rec,
                    "f1_score": norm_f1,
                },
                "field_breakdown": compute_field_match_breakdown(expected_flat, predicted_flat),
                "normalized_field_breakdown": compute_field_match_breakdown(
                    normalized_expected_flat,
                    normalized_predicted_flat,
                ),
            }
        )

    summary = {
        "dataset_size": len(rows),
        "metrics": {
            "field_accuracy": sum_acc / len(rows),
            "precision": sum_prec / len(rows),
            "recall": sum_rec / len(rows),
            "f1_score": sum_f1 / len(rows),
        },
        "normalized_metrics": {
            "field_accuracy": sum_norm_acc / len(rows),
            "precision": sum_norm_prec / len(rows),
            "recall": sum_norm_rec / len(rows),
            "f1_score": sum_norm_f1 / len(rows),
        },
        "samples": samples,
    }
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", OUTPUT_PATH)
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print(json.dumps(summary["normalized_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
