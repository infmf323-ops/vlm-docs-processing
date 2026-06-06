"""Общие фикстуры для тестов API.

Тесты используют изолированную in-memory SQLite-базу и подменяют обращения к
хранилищу S3 и ML-движку, поэтому для их запуска не нужны ни PostgreSQL/MinIO,
ни тяжёлые библиотеки (torch/transformers). Если эти библиотеки не установлены,
они автоматически заменяются заглушками — это позволяет гонять API-тесты в
лёгком CI-окружении.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


# --- Заглушки тяжёлых/инфраструктурных зависимостей (только если их нет) ---
def _stub_module(name: str) -> None:
    try:
        __import__(name)
        return
    except Exception:
        pass
    module = types.ModuleType(name)

    def __getattr__(_attr: str):  # PEP 562: любой атрибут -> мок
        return MagicMock()

    module.__getattr__ = __getattr__  # type: ignore[attr-defined]
    sys.modules[name] = module


for _name in ("torch", "transformers", "peft", "boto3", "fitz"):
    _stub_module(_name)


from sqlalchemy import create_engine  # noqa: E402
from sqlalchemy.orm import sessionmaker  # noqa: E402
from sqlalchemy.pool import StaticPool  # noqa: E402

# Патчим engine до импорта приложения, чтобы create_all шёл в SQLite, а не в Postgres.
import app.db.session as db_session  # noqa: E402

_test_engine = create_engine(
    "sqlite://",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    future=True,
)
_TestingSessionLocal = sessionmaker(
    bind=_test_engine, autoflush=False, autocommit=False, future=True
)
db_session.engine = _test_engine
db_session.SessionLocal = _TestingSessionLocal

from app.db.base import Base  # noqa: E402
import app.models.user  # noqa: E402,F401  (регистрация моделей в метаданных)
import app.models.job  # noqa: E402,F401
from app.db.session import get_db  # noqa: E402
from app.main import app  # noqa: E402


def _override_get_db():
    db = _TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = _override_get_db


class FakeStorage:
    """In-memory подмена StorageService (без обращений к S3/MinIO)."""

    store: dict[str, bytes] = {}

    def __init__(self, *args, **kwargs) -> None:
        pass

    def ensure_bucket(self) -> None:
        pass

    def upload_bytes(self, key: str, data: bytes, content_type: str) -> str:
        FakeStorage.store[key] = data
        return key

    def upload_json(self, key: str, payload: dict) -> str:
        import json

        FakeStorage.store[key] = json.dumps(payload).encode("utf-8")
        return key

    def download_bytes(self, key: str) -> bytes:
        return FakeStorage.store[key]

    def delete_object(self, key: str) -> None:
        FakeStorage.store.pop(key, None)

    def make_download_url(self, key: str, expires: int = 3600) -> str:
        return f"https://storage.test/{key}"


class _FakeNormalized:
    """Минимальный объект normalized_result с model_dump()."""

    def __init__(self, payload: dict) -> None:
        self._payload = payload

    def model_dump(self, mode: str | None = None) -> dict:
        return self._payload


class _FakeOutput:
    def __init__(self, normalized: dict, raw: dict) -> None:
        self.normalized_result = _FakeNormalized(normalized)
        self.raw_result = raw


class FakeExtractionService:
    """Детерминированный движок: помечает паспорта как корректный MRZ."""

    def __init__(self, *args, **kwargs) -> None:
        pass

    def predict(self, *, image, source_filename, request):
        doc_type = getattr(request.document_type, "value", "invoice")
        if doc_type == "passport":
            normalized = {
                "document_type": "passport",
                "fields": {"surname": "ИВАНОВ", "given_names": "ИВАН"},
                "validation": {"is_valid": True, "issues": []},
            }
            raw = {"mode": "passport_mrz", "corrected": False, "n_corrections": 0}
        else:
            normalized = {
                "document_type": doc_type,
                "fields": {"invoice_no": "A-1"},
                "validation": {"is_valid": True, "issues": []},
            }
            raw = {"mode": "heuristic"}
        return _FakeOutput(normalized, raw)


@pytest.fixture(autouse=True)
def _patch_externals(monkeypatch):
    """Подменяем хранилище, чтение изображения и ML-движок во всех тестах."""
    import app.services.jobs as jobs_module
    import app.api.routes.jobs as jobs_routes

    FakeStorage.store = {}
    monkeypatch.setattr(jobs_module, "StorageService", FakeStorage)
    monkeypatch.setattr(jobs_routes, "StorageService", FakeStorage)
    monkeypatch.setattr(jobs_module, "ExtractionService", FakeExtractionService)
    monkeypatch.setattr(
        jobs_module,
        "load_first_page_image",
        lambda content, content_type: (object(), b"PNGPREVIEW"),
    )
    monkeypatch.setattr(
        jobs_module,
        "load_all_page_images",
        lambda content, content_type, max_pages=20: [object()],
    )
    yield


@pytest.fixture(autouse=True)
def _reset_db():
    """Чистая схема для каждого теста."""
    Base.metadata.drop_all(bind=_test_engine)
    Base.metadata.create_all(bind=_test_engine)
    yield


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def auth_client(client):
    """Клиент с зарегистрированным и авторизованным пользователем."""
    client.post(
        "/api/v1/auth/register",
        json={
            "email": "user@test.io",
            "full_name": "Test User",
            "password": "secret123",
        },
    )
    token = client.post(
        "/api/v1/auth/login",
        json={"email": "user@test.io", "password": "secret123"},
    ).json()["access_token"]
    client.headers.update({"Authorization": f"Bearer {token}"})
    return client
