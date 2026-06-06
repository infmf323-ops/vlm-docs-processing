"""Интеграционные тесты HTTP API (auth + jobs).

Запуск (в backend-окружении с установленными зависимостями):
    pytest tests/test_api.py -v
"""
from __future__ import annotations

PREFIX = "/api/v1"


# --------------------------- Аутентификация ---------------------------
def test_register_returns_user(client):
    resp = client.post(
        f"{PREFIX}/auth/register",
        json={"email": "a@test.io", "full_name": "Alice", "password": "pw123456"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["email"] == "a@test.io"
    assert body["full_name"] == "Alice"
    assert "password" not in body and "password_hash" not in body


def test_register_duplicate_rejected(client):
    payload = {"email": "dup@test.io", "full_name": "Dup", "password": "pw123456"}
    assert client.post(f"{PREFIX}/auth/register", json=payload).status_code == 200
    resp = client.post(f"{PREFIX}/auth/register", json=payload)
    assert resp.status_code == 400


def test_login_success_and_wrong_password(client):
    client.post(
        f"{PREFIX}/auth/register",
        json={"email": "b@test.io", "full_name": "Bob", "password": "pw123456"},
    )
    ok = client.post(
        f"{PREFIX}/auth/login", json={"email": "b@test.io", "password": "pw123456"}
    )
    assert ok.status_code == 200
    assert ok.json()["access_token"]

    bad = client.post(
        f"{PREFIX}/auth/login", json={"email": "b@test.io", "password": "WRONG"}
    )
    assert bad.status_code == 401


def test_jobs_require_authentication(client):
    assert client.get(f"{PREFIX}/jobs").status_code == 401


# ------------------------------- Jobs --------------------------------
def _upload(client, *, filename="doc.png", document_type="invoice", content=b"img"):
    return client.post(
        f"{PREFIX}/jobs",
        files={"file": (filename, content, "image/png")},
        data={"document_type": document_type, "extraction_engine": "donut"},
    )


def test_create_list_and_detail(auth_client):
    created = _upload(auth_client, filename="invoice.png")
    assert created.status_code == 201
    job_id = created.json()["id"]
    assert created.json()["status"] == "queued"

    listing = auth_client.get(f"{PREFIX}/jobs")
    assert listing.status_code == 200
    assert any(j["id"] == job_id for j in listing.json())

    detail = auth_client.get(f"{PREFIX}/jobs/{job_id}")
    assert detail.status_code == 200
    assert detail.json()["original_filename"] == "invoice.png"
    assert detail.json()["requested_document_type"] == "invoice"


def test_create_empty_file_rejected(auth_client):
    resp = _upload(auth_client, content=b"")
    assert resp.status_code == 400


def test_detail_not_found(auth_client):
    assert auth_client.get(f"{PREFIX}/jobs/999999").status_code == 404


def test_retry_resets_status(auth_client):
    job_id = _upload(auth_client).json()["id"]
    resp = auth_client.post(f"{PREFIX}/jobs/{job_id}/retry")
    assert resp.status_code == 200
    assert resp.json()["status"] == "queued"


def test_delete_job(auth_client):
    job_id = _upload(auth_client).json()["id"]
    assert auth_client.delete(f"{PREFIX}/jobs/{job_id}").status_code == 204
    assert auth_client.get(f"{PREFIX}/jobs/{job_id}").status_code == 404


def test_user_isolation(auth_client, client):
    """Пользователь не видит и не может удалить чужие задания."""
    job_id = _upload(auth_client).json()["id"]

    # второй пользователь
    client.post(
        f"{PREFIX}/auth/register",
        json={"email": "c@test.io", "full_name": "Carol", "password": "pw123456"},
    )
    token = client.post(
        f"{PREFIX}/auth/login", json={"email": "c@test.io", "password": "pw123456"}
    ).json()["access_token"]
    other = {"Authorization": f"Bearer {token}"}

    assert client.get(f"{PREFIX}/jobs/{job_id}", headers=other).status_code == 404
    assert client.delete(f"{PREFIX}/jobs/{job_id}", headers=other).status_code == 404


def test_list_filter_by_status(auth_client):
    _upload(auth_client)  # queued
    resp = auth_client.get(f"{PREFIX}/jobs", params={"status": "queued"})
    assert resp.status_code == 200
    assert all(j["status"] == "queued" for j in resp.json())
    assert auth_client.get(f"{PREFIX}/jobs", params={"status": "done"}).json() == []


def test_stats_and_processing(auth_client):
    # пустая статистика
    empty = auth_client.get(f"{PREFIX}/jobs/stats").json()
    assert empty["total"] == 0

    # создаём паспорт и прогоняем обработку напрямую через сервис
    job_id = _upload(auth_client, filename="passport.png", document_type="passport").json()[
        "id"
    ]

    from app.db.session import SessionLocal
    from app.services.jobs import JobService

    db = SessionLocal()
    try:
        processed = JobService(db).process_next_job()
        assert processed is not None
        assert processed.status.value == "done"
    finally:
        db.close()

    stats = auth_client.get(f"{PREFIX}/jobs/stats").json()
    assert stats["total"] == 1
    assert stats["by_status"].get("done") == 1
    assert stats["by_document_type"].get("passport") == 1
    assert stats["mrz_total"] == 1
    assert stats["mrz_valid"] == 1
    assert stats["mrz_valid_rate"] == 1.0
    assert stats["success_rate"] == 1.0

    # фильтр по типу документа после обработки
    filtered = auth_client.get(f"{PREFIX}/jobs", params={"document_type": "passport"})
    assert len(filtered.json()) == 1


def test_export_csv(auth_client):
    _upload(auth_client, filename="one.png")
    resp = auth_client.get(f"{PREFIX}/jobs/export.csv")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/csv")
    text = resp.text
    assert "id,original_filename,status" in text.splitlines()[0]
    assert "one.png" in text


# ------------------------- Надёжность / лимиты -------------------------
def test_upload_rejects_unsupported_type(auth_client):
    resp = auth_client.post(
        f"{PREFIX}/jobs",
        files={"file": ("note.txt", b"hello", "text/plain")},
        data={"document_type": "invoice", "extraction_engine": "donut"},
    )
    assert resp.status_code == 415


def test_upload_rejects_oversized(auth_client):
    from app.core.config import get_settings

    settings = get_settings()
    original = settings.max_upload_mb
    settings.max_upload_mb = 0  # любой непустой файл превысит лимит
    try:
        resp = _upload(auth_client, content=b"x" * 16)
        assert resp.status_code == 413
    finally:
        settings.max_upload_mb = original


def test_list_pagination(auth_client):
    for i in range(3):
        _upload(auth_client, filename=f"f{i}.png")
    page1 = auth_client.get(f"{PREFIX}/jobs", params={"limit": 2}).json()
    assert len(page1) == 2
    page2 = auth_client.get(f"{PREFIX}/jobs", params={"limit": 2, "offset": 2}).json()
    assert len(page2) == 1


def test_worker_marks_job_failed_on_engine_error(auth_client, monkeypatch):
    job_id = _upload(auth_client).json()["id"]

    import app.services.jobs as jobs_module

    class BoomEngine:
        def __init__(self, *args, **kwargs):
            pass

        def predict(self, **kwargs):
            raise RuntimeError("engine boom")

    monkeypatch.setattr(jobs_module, "ExtractionService", BoomEngine)

    from app.db.session import SessionLocal
    from app.services.jobs import JobService

    db = SessionLocal()
    try:
        processed = JobService(db).process_next_job()
        assert processed is not None
        assert processed.status.value == "failed"
        assert "boom" in (processed.error_message or "")
    finally:
        db.close()

    detail = auth_client.get(f"{PREFIX}/jobs/{job_id}").json()
    assert detail["status"] == "failed"


def test_multipage_processing(auth_client, monkeypatch):
    import app.services.jobs as jobs_module

    # Эмулируем двухстраничный документ.
    monkeypatch.setattr(
        jobs_module,
        "load_all_page_images",
        lambda content, content_type, max_pages=20: [object(), object()],
    )
    _upload(auth_client, filename="multi.pdf", document_type="invoice")

    from app.db.session import SessionLocal
    from app.services.jobs import JobService

    db = SessionLocal()
    try:
        processed = JobService(db).process_next_job()
        assert processed is not None
        assert processed.status.value == "done"
        assert processed.raw_result.get("mode") == "multipage"
        assert processed.raw_result.get("page_count") == 2
        assert len(processed.raw_result.get("pages", [])) == 2
    finally:
        db.close()


def test_document_type_filter_before_processing(auth_client):
    # Паспорт загружен, но ещё НЕ обработан — он всё равно должен попадать в фильтр
    # по типу и в статистику (тип берётся из запрошенного при загрузке).
    _upload(auth_client, filename="p.png", document_type="passport")
    filtered = auth_client.get(f"{PREFIX}/jobs", params={"document_type": "passport"})
    assert len(filtered.json()) == 1
    stats = auth_client.get(f"{PREFIX}/jobs/stats").json()
    assert stats["by_document_type"].get("passport") == 1


def test_retry_reenqueues_in_queue_mode(auth_client, monkeypatch):
    # В режиме очереди создание и повтор задания должны публиковать его в RQ.
    from app.core.config import get_settings
    import app.services.queue as queue_module

    calls: list[int] = []
    monkeypatch.setattr(queue_module, "enqueue_job", lambda jid: calls.append(jid))
    settings = get_settings()
    original = settings.use_task_queue
    settings.use_task_queue = True
    try:
        job_id = _upload(auth_client).json()["id"]  # create_job ставит в очередь
        assert calls == [job_id]
        auth_client.post(f"{PREFIX}/jobs/{job_id}/retry")  # retry публикует повторно
        assert calls == [job_id, job_id]
    finally:
        settings.use_task_queue = original


def test_detail_uses_requested_type_column_when_request_json_missing(auth_client):
    # Тип в деталях должен браться из колонки requested_document_type даже если
    # request.json в хранилище потерян/повреждён (раньше здесь был дефолт invoice).
    import app.services.jobs as jobs_module

    job_id = _upload(
        auth_client, filename="p.png", document_type="passport"
    ).json()["id"]

    store = jobs_module.StorageService.store  # in-memory FakeStorage
    for key in [k for k in list(store) if k.endswith("request.json")]:
        store.pop(key)

    detail = auth_client.get(f"{PREFIX}/jobs/{job_id}").json()
    assert detail["requested_document_type"] == "passport"
