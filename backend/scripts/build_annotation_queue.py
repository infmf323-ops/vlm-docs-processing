from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path("E:/thesis")
OUTPUT = ROOT / "data" / "multidoc" / "annotation_queue.jsonl"
INFERENCE_COMPARISON = ROOT / "inference_comparison.json"

INVOICE_TAGS = {
    "invoice_no": "invoice_no",
    "invoice_date": "invoice_date",
    "seller": "seller",
    "client": "client",
    "seller_tax_id": "seller_tax_id",
    "client_tax_id": "client_tax_id",
    "iban": "iban",
    "total_net": "total_net_worth",
    "total_tax": "total_vat",
    "total_gross": "total_gross_worth",
}


def extract_tag(text: str, tag: str) -> str | None:
    match = re.search(rf"<s_{re.escape(tag)}>(.*?)</s_{re.escape(tag)}>", text, flags=re.DOTALL)
    if not match:
        return None
    value = re.sub(r"\s+", " ", match.group(1)).strip()
    return value or None


def make_empty_invoice_fields() -> dict:
    return {
        "invoice_no": None,
        "invoice_date": None,
        "currency": None,
        "seller": {
            "name": None,
            "address": None,
            "tax_id": None,
            "iban": None,
        },
        "buyer": {
            "name": None,
            "address": None,
            "tax_id": None,
        },
        "total_net": None,
        "total_tax": None,
        "total_gross": None,
    }


def make_invoice_candidate(path: Path) -> dict:
    return {
        "id": path.stem,
        "image_path": str(path).replace("\\", "/"),
        "document_type": "invoice",
        "fields": make_empty_invoice_fields(),
        "status": "needs_annotation",
    }


def parse_invoice_ground_truth(text: str) -> dict:
    extracted = {
        target_key: extract_tag(text, source_tag) for target_key, source_tag in INVOICE_TAGS.items()
    }
    return {
        "invoice_no": extracted["invoice_no"],
        "invoice_date": extracted["invoice_date"],
        "currency": None,
        "seller": {
            "name": extracted["seller"],
            "address": None,
            "tax_id": extracted["seller_tax_id"],
            "iban": extracted["iban"],
        },
        "buyer": {
            "name": extracted["client"],
            "address": None,
            "tax_id": extracted["client_tax_id"],
        },
        "total_net": extracted["total_net"],
        "total_tax": extracted["total_tax"],
        "total_gross": extracted["total_gross"],
    }


def load_invoice_drafts() -> dict[str, dict]:
    if not INFERENCE_COMPARISON.exists():
        return {}

    comparison = json.loads(INFERENCE_COMPARISON.read_text(encoding="utf-8"))
    drafts: dict[str, dict] = {}

    for sample in comparison.get("samples", []):
        dataset_index = sample.get("dataset_index")
        ground_truth = sample.get("ground_truth")
        if dataset_index is None or not ground_truth:
            continue
        drafts[f"val_{dataset_index}"] = {
            "fields": parse_invoice_ground_truth(ground_truth),
            "source": "inference_comparison_ground_truth",
        }
    return drafts


def main() -> None:
    candidates: list[dict] = []
    invoice_drafts = load_invoice_drafts()

    for path in sorted((ROOT / "report_assets").glob("*.png")):
        candidate = make_invoice_candidate(path)
        draft = invoice_drafts.get(path.stem)
        if draft:
            candidate["fields"] = draft["fields"]
            candidate["status"] = "draft_from_previous_experiment"
            candidate["draft_source"] = draft["source"]
        candidates.append(candidate)

    passport_source = Path(r"C:\Users\wasd\Videos\Michelle_Obama's_U.S._passport_(2013-2018).png")
    if passport_source.exists():
        candidates.append(
            {
                "id": "passport_michelle_obama_review",
                "image_path": str(passport_source).replace("\\", "/"),
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
                "status": "verified_reference",
            }
        )

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", encoding="utf-8") as fp:
        for row in candidates:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(OUTPUT)
    print(f"records={len(candidates)}")
    print(f"invoice_drafts={sum(1 for row in candidates if row['document_type'] == 'invoice' and row['status'] != 'needs_annotation')}")


if __name__ == "__main__":
    main()
