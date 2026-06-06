"""End-to-end smoke-тест запущенного API: регистрация → загрузка документа →
ожидание обработки → вывод результата и валидности MRZ.

Запуск (когда подняты API + worker):
    python scripts/smoke_test_api.py путь/к/паспорту.png
    python scripts/smoke_test_api.py путь/к/счёту.pdf --type invoice --engine donut

Зависимости: httpx (есть в requirements.txt). Это проверка «живого» сервиса,
а не юнит-тест — поэтому вынесена в scripts, а не в tests.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import httpx

ROOT = "http://localhost:8000"
BASE = ROOT + "/api/v1"
EMAIL, PASSWORD = "smoke@test.local", "smoke-pass-12345"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="путь к изображению/PDF документа")
    ap.add_argument("--type", default="passport", help="тип документа (passport/invoice/...)")
    ap.add_argument("--engine", default="paddleocr_vl", help="движок (paddleocr_vl/donut)")
    ap.add_argument("--timeout", type=int, default=300, help="ожидание обработки, сек")
    args = ap.parse_args()

    if not os.path.exists(args.image):
        print(f"Файл не найден: {args.image}")
        return 2

    with httpx.Client(timeout=60) as c:
        # 1) health
        try:
            h = c.get(BASE + "/system/health")
            print(f"[health] {h.status_code} {h.text}")
        except Exception as exc:
            print(f"API недоступен на {ROOT}: {exc}\nЗапущен ли uvicorn?")
            return 2

        # 2) регистрация (если уже есть — игнорируем) + логин
        c.post(BASE + "/auth/register",
               json={"email": EMAIL, "full_name": "Smoke", "password": PASSWORD})
        login = c.post(BASE + "/auth/login", json={"email": EMAIL, "password": PASSWORD})
        if login.status_code != 200:
            print(f"[login] не удалось: {login.status_code} {login.text}")
            return 2
        headers = {"Authorization": f"Bearer {login.json()['access_token']}"}
        print("[auth] OK")

        # 3) загрузка документа
        with open(args.image, "rb") as f:
            up = c.post(
                BASE + "/jobs", headers=headers,
                files={"file": (os.path.basename(args.image), f, "application/octet-stream")},
                data={"document_type": args.type, "extraction_engine": args.engine},
            )
        if up.status_code != 201:
            print(f"[upload] ошибка: {up.status_code} {up.text}")
            return 2
        job_id = up.json()["id"]
        print(f"[upload] задание #{job_id} создано")

        # 4) ожидание результата
        deadline = time.time() + args.timeout
        detail = {}
        while time.time() < deadline:
            detail = c.get(f"{BASE}/jobs/{job_id}", headers=headers).json()
            if detail["status"] in ("done", "failed"):
                break
            time.sleep(2)

        print(f"\n[status] {detail.get('status')}")
        if detail.get("status") == "failed":
            print("[error]", detail.get("error_message"))
            return 1

        raw = detail.get("raw_result") or {}
        norm = detail.get("normalized_result") or {}
        if raw.get("mode") == "passport_mrz":
            v = (norm.get("validation") or {}).get("is_valid")
            print(f"[MRZ] режим passport_mrz | контрольные цифры пройдены: {v} | "
                  f"исправлено ошибок: {raw.get('n_corrections', 0)}")
        else:
            print(f"[mode] {raw.get('mode')}")
        print("\n[normalized_result]")
        print(json.dumps(norm, ensure_ascii=False, indent=2)[:1200])
        print("\nГотово. Если статус done и MRZ валидна — сервис обрабатывает документы end-to-end.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
