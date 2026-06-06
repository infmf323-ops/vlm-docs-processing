from __future__ import annotations

import json
import re
import shutil
from pathlib import Path


ROOT = Path("E:/thesis")
DATA_DIR = ROOT / "data" / "multidoc"
IMAGES_DIR = DATA_DIR / "images"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
TEMPLATES_FILE = DATA_DIR / "templates.jsonl"
INFERENCE_COMPARISON = ROOT / "inference_comparison.json"

INVOICE_GT_TAGS = {
    "invoice_no": "invoice_no",
    "invoice_date": "invoice_date",
    "seller_name": "seller",
    "buyer_name": "client",
    "seller_tax_id": "seller_tax_id",
    "buyer_tax_id": "client_tax_id",
    "iban": "iban",
    "total_net": "total_net_worth",
    "total_tax": "total_vat",
    "total_gross": "total_gross_worth",
}


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def maybe_copy(source: Path, target_name: str) -> str | None:
    if not source.exists():
        return None
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    target_path = IMAGES_DIR / target_name
    if not target_path.exists():
        shutil.copy2(source, target_path)
    return str(target_path).replace("\\", "/")


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<s_{re.escape(tag)}>(.*?)</s_{re.escape(tag)}>", text, flags=re.DOTALL)
    if not match:
        return None
    value = re.sub(r"\s+", " ", match.group(1)).strip()
    return value or None


def load_verified_invoice_rows() -> list[dict]:
    if not INFERENCE_COMPARISON.exists():
        return []

    comparison = json.loads(INFERENCE_COMPARISON.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for sample in comparison.get("samples", []):
        dataset_index = sample.get("dataset_index")
        ground_truth = sample.get("ground_truth")
        if dataset_index is None or not ground_truth:
            continue

        image_source = ROOT / "report_assets" / f"val_{dataset_index}.png"
        image_target = maybe_copy(image_source, f"invoice_val_{dataset_index}.png")
        if not image_target:
            continue

        values = {
            key: extract_tag(ground_truth, tag_name) for key, tag_name in INVOICE_GT_TAGS.items()
        }
        rows.append(
            {
                "id": f"invoice_val_{dataset_index}",
                "image_path": image_target,
                "document_type": "invoice",
                "fields": {
                    "invoice_no": values["invoice_no"],
                    "invoice_date": values["invoice_date"],
                    "currency": None,
                    "seller": {
                        "name": values["seller_name"],
                        "address": None,
                        "tax_id": values["seller_tax_id"],
                        "iban": values["iban"],
                    },
                    "buyer": {
                        "name": values["buyer_name"],
                        "address": None,
                        "tax_id": values["buyer_tax_id"],
                    },
                    "total_net": values["total_net"],
                    "total_tax": values["total_tax"],
                    "total_gross": values["total_gross"],
                },
                "source": "inference_comparison_ground_truth",
            }
        )
    return rows


def main() -> None:
    train_rows: list[dict] = []
    val_rows: list[dict] = []
    template_rows: list[dict] = []

    passport_source = Path(r"C:\Users\wasd\Videos\Michelle_Obama's_U.S._passport_(2013-2018).png")
    passport_target = maybe_copy(passport_source, "passport_michelle_obama.png")
    if passport_target:
        train_rows.append(
            {
                "id": "passport_michelle_obama",
                "image_path": passport_target,
                "document_type": "passport",
                "fields": {
                    "document_number": "910239248",
                    "surname": "OBAMA",
                    "given_names": "MICHELLE",
                    "nationality": "UNITED STATES OF AMERICA",
                    "date_of_birth": "17 Jan 1964",
                    "sex": "F",
                    "place_of_birth": "ILLINOIS, U.S.A.",
                    "date_of_issue": "06 Dec 2013",
                    "date_of_expiry": "05 Dec 2018",
                    "issuing_authority": "United States Department of State",
                },
            }
        )

    verified_invoice_rows = load_verified_invoice_rows()
    if verified_invoice_rows:
        val_ids = {"invoice_val_1"}
        for row in verified_invoice_rows:
            if row["id"] in val_ids:
                val_rows.append(row)
            else:
                train_rows.append(row)

    train_sample = DATA_DIR / "train.sample.jsonl"
    val_sample = DATA_DIR / "val.sample.jsonl"
    if train_sample.exists():
        for line in train_sample.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["bootstrap_template"] = True
                template_rows.append(row)
    if val_sample.exists():
        for line in val_sample.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                row["bootstrap_template"] = True
                template_rows.append(row)

    write_jsonl(TRAIN_FILE, train_rows)
    write_jsonl(VAL_FILE, val_rows)
    write_jsonl(TEMPLATES_FILE, template_rows)

    print(TRAIN_FILE)
    print(VAL_FILE)
    print(TEMPLATES_FILE)
    print(IMAGES_DIR)


if __name__ == "__main__":
    main()
