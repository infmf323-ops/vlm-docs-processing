from __future__ import annotations

import json
from pathlib import Path

import requests


ROOT = Path("E:/thesis")
OUTPUT = ROOT / "data" / "multidoc" / "external" / "docxpand_release_manifest.json"
API_URL = "https://api.github.com/repos/QuickSign/docxpand/releases/tags/v1.0.0"


def main() -> None:
    response = requests.get(API_URL, timeout=30)
    response.raise_for_status()
    payload = response.json()

    assets = [
        {
            "name": asset.get("name"),
            "size": asset.get("size"),
            "download_url": asset.get("browser_download_url"),
        }
        for asset in payload.get("assets", [])
    ]

    manifest = {
        "dataset": "DocXPand-25k",
        "tag": payload.get("tag_name"),
        "published_at": payload.get("published_at"),
        "assets": assets,
        "notes": {
            "license": "CC-BY-NC-SA 4.0",
            "import_role": "primary_supervised_identity_source",
            "expected_usage": "download, extract, convert with convert_docxpand_to_multidoc.py",
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(OUTPUT)
    print(f"assets={len(assets)}")


if __name__ == "__main__":
    main()
