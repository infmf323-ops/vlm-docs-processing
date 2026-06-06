from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path

import torch
from peft import PeftModel
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor


ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "PaddlePaddle/PaddleOCR-VL"
HF_CACHE = ROOT / ".hf-cache"
DATASET_PATH = Path(
    os.getenv(
        "PASSPORT_EVAL_DATASET",
        str(ROOT / "data" / "multidoc" / "passport_eval_diverse_v1.jsonl"),
    )
)
OUTPUT_PATH = Path(os.getenv("PASSPORT_EVAL_OUTPUT", str(ROOT / "passport_mrz_eval_kaggle.json")))
ADAPTER_DIR = Path(
    os.getenv(
        "PASSPORT_ADAPTER_DIR",
        str(ROOT / "outputs" / "paddleocr_vl_passport_russian_finetune_mrz_v3_kaggle"),
    )
)
MAX_IMAGE_SIDE = int(os.getenv("MAX_IMAGE_SIDE", "768"))
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "128"))
MAX_GENERATION_TIME = float(os.getenv("MAX_GENERATION_TIME", "300"))
PROJECT_ROOT = ROOT
MRZ_DERIVABLE_FIELDS = {
    "document_number",
    "surname",
    "given_names",
    "nationality",
    "date_of_birth",
    "sex",
    "date_of_expiry",
    # 'mrz' itself is also derivable — handled separately if present
}

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


def strip_accents(value: str) -> str:
    """Convert accented chars to their ASCII base (e.g. Ç→C, É→E, Ñ→N)."""
    return "".join(
        c for c in unicodedata.normalize("NFD", value)
        if unicodedata.category(c) != "Mn"
    )


def normalize_generic_text(value: str) -> str:
    cleaned = strip_accents(value).upper().strip()
    cleaned = cleaned.replace("’", "’")
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned


def normalize_document_number(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", normalize_generic_text(value))


def normalize_date_value(value: str) -> str:
    cleaned = normalize_generic_text(value)
    cleaned = cleaned.replace("/", ".").replace("-", ".")
    month_map = {
        "JAN": "01", "FEB": "02", "MAR": "03", "APR": "04",
        "MAY": "05", "JUN": "06", "JUL": "07", "AUG": "08",
        "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12",
    }
    month_match = re.match(r"^(\d{1,2}) ([A-Z]{3}) (\d{4})$", cleaned)
    if month_match:
        dd, mon, yyyy = month_match.groups()
        if mon in month_map:
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
        normalized = normalized.strip()
        # Expand 3-letter ISO country codes stored in ground truth (e.g. "ARE" → "UNITED ARAB EMIRATES")
        if key == "nationality" and len(normalized) == 3 and normalized in COUNTRY_MAP:
            normalized = COUNTRY_MAP[normalized]
        # Normalise native-language nationality names to English equivalents
        # (e.g. "FRANCAISE"/"FRANÇAISE" on French passport → "FRANCE" as in COUNTRY_MAP)
        _NAT_ALIASES: dict[str, str] = {
            "FRANCAISE": "FRANCE",
            "FRANCAIS": "FRANCE",
            "RUSSE": "RUSSIAN FEDERATION",
            "RUSSO": "RUSSIAN FEDERATION",
            "EMIRATS ARABES UNIS": "UNITED ARAB EMIRATES",
        }
        if key == "nationality" and normalized in _NAT_ALIASES:
            normalized = _NAT_ALIASES[normalized]
        return normalized
    return normalize_generic_text(value)


def normalize_flat_fields(fields: dict[str, str]) -> dict[str, str]:
    return {
        key: normalize_field_value(key, str(value))
        for key, value in fields.items()
        if value is not None
    }


def build_prompt() -> str:
    return (
        "Extract only the passport MRZ exactly as plain text. "
        "Return only the MRZ lines, preserving line breaks. "
        "Do not add labels, JSON, or explanation. "
        "If MRZ is missing, write null."
    )


def clean_generated_text(text: str) -> str:
    cleaned = text.replace("зЏ­зє§пјљ___", "")
    cleaned = cleaned.replace("зЏ­зє§", "")
    # BUG FIX: was `text.replace(...)` — must use `cleaned` to preserve prior subs
    cleaned = cleaned.replace("\\r", "\n").replace("\\n", "\n")
    cleaned = cleaned.replace("\r", "\n")
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    return cleaned.strip()


def normalize_name_token(text: str) -> str:
    cleaned = text.replace("<", " ").replace(">", " ")
    cleaned = re.sub(r"[^A-Z' -]+", " ", cleaned.upper())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned.upper()


def format_mrz_date(compact: str) -> str | None:
    if len(compact) != 6 or not compact.isdigit():
        return None
    yy, mm, dd = compact[:2], compact[2:4], compact[4:6]
    year = int(yy)
    full_year = 1900 + year if year >= 50 else 2000 + year
    return f"{dd}.{mm}.{full_year}"


def extract_mrz_text(text: str) -> str | None:
    cleaned = clean_generated_text(text)
    if not cleaned or cleaned.lower() == "null":
        return None

    lines = [line.strip().upper() for line in cleaned.splitlines() if line.strip()]
    mrz_lines = []
    for line in lines:
        compact = re.sub(r"[^A-Z0-9<]", "", line)
        if len(compact) >= 20:
            mrz_lines.append(compact)

    if len(mrz_lines) >= 2:
        return "\n".join(mrz_lines[:2])
    if len(mrz_lines) == 1:
        compact = mrz_lines[0]
        if compact.startswith("P<") and len(compact) > 44:
            return compact[:44] + "\n" + compact[44:]
        return compact

    compact = re.sub(r"[^A-Z0-9<]", "", cleaned.upper())
    if compact.startswith("P<") and len(compact) >= 44:
        if len(compact) > 44:
            return compact[:44] + "\n" + compact[44:]
        return compact
    return None


def parse_mrz_text(mrz_text: str | None) -> dict:
    if not mrz_text:
        return {}

    lines = [re.sub(r"[^A-Z0-9<]", "", line.upper()) for line in mrz_text.splitlines() if line.strip()]
    if not lines:
        return {}

    first_line = lines[0]
    second_line = lines[1] if len(lines) > 1 else ""
    result: dict[str, str] = {"mrz": "\n".join(lines)}

    # Strip first-line prefix to get to country code + name payload.
    # German passports: PPD<< (P=passport, P=personal, D=Germany, <<)
    # Standard passports: P<NNN...
    prefix_stripped = first_line
    if prefix_stripped.startswith("PPD<<"):
        prefix_stripped = prefix_stripped[5:]
    elif prefix_stripped.startswith("P<"):
        prefix_stripped = prefix_stripped[2:]
    elif prefix_stripped.startswith("PP<"):
        prefix_stripped = prefix_stripped[3:]
    elif prefix_stripped.startswith("P"):
        prefix_stripped = prefix_stripped[1:]
    # Recovery: model sometimes omits leading 'P', outputting '<CCCNAME<<GIVEN' instead of 'P<CCCNAME<<GIVEN'.
    # Detect by: starts with '<' (not '<<'), and next 3 chars are a known country code.
    if prefix_stripped.startswith("<") and not prefix_stripped.startswith("<<"):
        candidate = prefix_stripped[1:]
        if len(candidate) >= 3 and candidate[:3] in COUNTRY_MAP:
            prefix_stripped = candidate
    # Recovery: model outputs '<<CCCNAME<<GIVEN' (double-filler prefix, e.g. '<rho<RUS...' -> '<<RUS...')
    # Strip the leading '<<' and treat rest as country+name.
    if prefix_stripped.startswith("<<"):
        candidate = prefix_stripped[2:]
        if len(candidate) >= 3 and candidate[:3] in COUNTRY_MAP:
            prefix_stripped = candidate

    country = None
    name_payload = prefix_stripped
    if len(prefix_stripped) >= 3:
        possible_country = prefix_stripped[:3]
        if possible_country in COUNTRY_MAP:
            country = possible_country
            name_payload = prefix_stripped[3:]

    if "<<" in name_payload:
        surname_token, given_token = name_payload.split("<<", 1)
        given_names = normalize_name_token(given_token)
        surname = normalize_name_token(surname_token)
        given_parts = [part for part in given_names.split() if part]
        if country is None and given_parts and given_parts[-1] in {"AL", "EL"}:
            surname = f"{given_parts[-1]} {surname}".strip()
        if surname:
            result["surname"] = surname
        if given_names:
            result["given_names"] = given_names
        if country in COUNTRY_MAP:
            result["nationality"] = COUNTRY_MAP[country]

    # Parse second line with strict ICAO 9303 field widths:
    #   doc_number(9) + check_digit(1) + nationality(3) + DOB(6) + check(1) + sex(1) + expiry(6)
    # Using {9} prevents absorbing the check digit into the document number.
    # Nationality uses [A-Z<]{3} to handle Germany's "D<<" ICAO code.
    search_line = second_line or first_line
    m2 = re.search(
        r"([A-Z0-9<]{9})[0-9<]([A-Z<]{3})(\d{6})[0-9<]([MF<])(\d{6})",
        search_line,
    )
    if m2:
        doc_num = m2.group(1).replace("<", "")
        if doc_num and any(ch.isdigit() for ch in doc_num):
            result["document_number"] = doc_num

        nat_raw = m2.group(2)
        nat_key = nat_raw.replace("<", "")
        nat_value = COUNTRY_MAP.get(nat_key) or COUNTRY_MAP.get(nat_raw)
        if nat_value and "nationality" not in result:
            result["nationality"] = nat_value

        date_of_birth = format_mrz_date(m2.group(3))
        if date_of_birth:
            result["date_of_birth"] = date_of_birth
        sex = m2.group(4).replace("<", "")
        if sex in {"M", "F"}:
            result["sex"] = sex
        date_of_expiry = format_mrz_date(m2.group(5))
        if date_of_expiry:
            result["date_of_expiry"] = date_of_expiry

    return result


def _apply_transformers_patches() -> None:
    import math
    import torch
    import transformers.modeling_utils as _tmu

    # ── Patch 1: inject ROPE_INIT_FUNCTIONS["default"] ────────────────────────
    # transformers 5.x removed the "default" key; remote code still looks it up.
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if "default" not in ROPE_INIT_FUNCTIONS:
            def _rope_default_init(config, device=None, **kw):
                base = float(getattr(config, "rope_theta", 10000.0))
                dim  = int(getattr(config, "head_dim",
                       getattr(config, "hidden_size", 896) //
                       getattr(config, "num_attention_heads", 8)))
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                return inv_freq, 1.0
            ROPE_INIT_FUNCTIONS["default"] = _rope_default_init
            print("[patch] ROPE_INIT_FUNCTIONS['default'] injected")
    except Exception as e:
        print(f"[patch] ROPE skip: {e}")

    # ── Patch 2: robust causal mask (transformers 5.x returns None via SDPA) ──
    try:
        import transformers.masking_utils as _mu

        def _robust_causal_mask(config, inputs_embeds=None, input_embeds=None,
                                past_key_values=None, output_attentions=False, **kw):
            emb = inputs_embeds if inputs_embeds is not None else input_embeds
            if emb is None:
                return None
            seq_len  = emb.shape[1]
            past_len = past_key_values.get_seq_length() if past_key_values is not None else 0
            total    = seq_len + past_len
            min_val  = torch.finfo(emb.dtype).min
            mask = torch.full((seq_len, total), fill_value=min_val,
                              device=emb.device, dtype=emb.dtype)
            mask = torch.triu(mask, diagonal=1 + past_len)
            return mask[None, None].expand(emb.shape[0], 1, -1, -1)

        _mu.create_causal_mask = _robust_causal_mask
        print("[patch] create_causal_mask replaced")
    except Exception as e:
        print(f"[patch] causal_mask skip: {e}")

    # ── Patch 3: _init_weights fix for RotaryEmbedding without method ─────────
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

        def _rope_default_init_fn(config, device=None, **kw):
            base = float(getattr(config, "rope_theta", 10000.0))
            dim  = int(getattr(config, "head_dim",
                   getattr(config, "hidden_size", 896) //
                   getattr(config, "num_attention_heads", 8)))
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
            return inv_freq, 1.0

        _orig_init_weights = _tmu.PreTrainedModel._init_weights

        def _patched_init_weights(self, module):
            if (hasattr(module, "rope_type") and
                    module.rope_type == "default" and
                    not hasattr(module, "compute_default_rope_parameters")):
                module.compute_default_rope_parameters = lambda config, device=None, **kw: _rope_default_init_fn(config)
            return _orig_init_weights(self, module)

        _tmu.PreTrainedModel._init_weights = _patched_init_weights
        print("[patch] _init_weights patched for RotaryEmbedding")
    except Exception as e:
        print(f"[patch] _init_weights skip: {e}")


def main() -> None:
    os.environ["HF_HOME"] = str(HF_CACHE)
    os.environ["HF_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(HF_CACHE / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(HF_CACHE / "transformers")

    # Must patch BEFORE from_pretrained so remote code sees patched symbols
    _apply_transformers_patches()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    processor = AutoProcessor.from_pretrained(
        MODEL_NAME,
        cache_dir=str(HF_CACHE),
        trust_remote_code=True,
    )
    tokenizer = processor.tokenizer
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        cache_dir=str(HF_CACHE),
        trust_remote_code=True,
        torch_dtype=torch.float16,
        attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR))
    model = model.to(device)
    model.eval()

    rows = load_rows(DATASET_PATH)
    prompt_text = build_prompt()

    samples = []
    sum_acc = sum_prec = sum_rec = sum_f1 = 0.0
    sum_norm_acc = sum_norm_prec = sum_norm_rec = sum_norm_f1 = 0.0
    sum_mrz_acc = sum_mrz_prec = sum_mrz_rec = sum_mrz_f1 = 0.0

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
            "repetition_penalty": 1.12,
            "no_repeat_ngram_size": 3,
        }
        if tokenizer.pad_token_id is not None:
            generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
        if tokenizer.eos_token_id is not None:
            generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
        if tokenizer.unk_token_id is not None:
            generation_kwargs["bad_words_ids"] = [[tokenizer.unk_token_id]]

        with torch.no_grad():
            output_ids = model.generate(
                **encoded,
                **generation_kwargs,
            )

        generated_text = processor.batch_decode(
            output_ids[:, encoded["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0].strip()

        extracted_mrz = extract_mrz_text(generated_text)
        predicted_fields = parse_mrz_text(extracted_mrz)

        expected_flat = flatten_scalars(row["fields"])
        predicted_flat = flatten_scalars(predicted_fields)
        acc = compute_field_accuracy(expected_flat, predicted_flat)
        prec, rec, f1 = compute_precision_recall_f1(expected_flat, predicted_flat)

        normalized_expected_flat = normalize_flat_fields(expected_flat)
        normalized_predicted_flat = normalize_flat_fields(predicted_flat)
        norm_acc = compute_field_accuracy(normalized_expected_flat, normalized_predicted_flat)
        norm_prec, norm_rec, norm_f1 = compute_precision_recall_f1(
            normalized_expected_flat, normalized_predicted_flat
        )

        # MRZ-derivable-only metrics (excludes place_of_birth, date_of_issue, issuing_authority, mrz raw text)
        mrz_expected = {k: v for k, v in normalized_expected_flat.items()
                        if k in MRZ_DERIVABLE_FIELDS}
        mrz_predicted = {k: v for k, v in normalized_predicted_flat.items()
                         if k in MRZ_DERIVABLE_FIELDS}
        mrz_acc = compute_field_accuracy(mrz_expected, mrz_predicted)
        mrz_prec, mrz_rec, mrz_f1 = compute_precision_recall_f1(mrz_expected, mrz_predicted)

        sum_acc += acc
        sum_prec += prec
        sum_rec += rec
        sum_f1 += f1
        sum_norm_acc += norm_acc
        sum_norm_prec += norm_prec
        sum_norm_rec += norm_rec
        sum_norm_f1 += norm_f1
        sum_mrz_acc += mrz_acc
        sum_mrz_prec += mrz_prec
        sum_mrz_rec += mrz_rec
        sum_mrz_f1 += mrz_f1

        samples.append(
            {
                "id": row["id"],
                "expected_fields": row["fields"],
                "predicted_fields": predicted_fields,
                "generated_text": generated_text,
                "extracted_mrz": extracted_mrz,
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
                "mrz_derivable_metrics": {
                    "field_accuracy": mrz_acc,
                    "precision": mrz_prec,
                    "recall": mrz_rec,
                    "f1_score": mrz_f1,
                },
            }
        )

    n = len(rows)
    summary = {
        "dataset_size": n,
        "metrics": {
            "field_accuracy": sum_acc / n,
            "precision": sum_prec / n,
            "recall": sum_rec / n,
            "f1_score": sum_f1 / n,
        },
        "normalized_metrics": {
            "field_accuracy": sum_norm_acc / n,
            "precision": sum_norm_prec / n,
            "recall": sum_norm_rec / n,
            "f1_score": sum_norm_f1 / n,
        },
        "mrz_derivable_metrics": {
            "field_accuracy": sum_mrz_acc / n,
            "precision": sum_mrz_prec / n,
            "recall": sum_mrz_rec / n,
            "f1_score": sum_mrz_f1 / n,
        },
        "samples": samples,
    }
    OUTPUT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("saved:", OUTPUT_PATH)
    print("=== All fields ===")
    print(json.dumps(summary["metrics"], ensure_ascii=False, indent=2))
    print("=== Normalized all fields ===")
    print(json.dumps(summary["normalized_metrics"], ensure_ascii=False, indent=2))
    print("=== MRZ-derivable fields only (surname/given/nationality/doc_num/dob/sex/expiry) ===")
    print(json.dumps(summary["mrz_derivable_metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
