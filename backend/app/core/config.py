from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(".env", ".env.local"),
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    app_name: str = "Invoice Extraction UI"
    api_v1_prefix: str = "/api/v1"
    secret_key: str = "change-me"
    access_token_expire_minutes: int = 60 * 24

    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "invoice_ui"
    postgres_user: str = "invoice_user"
    postgres_password: str = "invoice_password"

    s3_endpoint_url: str | None = None
    s3_region: str = "us-east-1"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"
    s3_bucket: str = "invoice-jobs"

    model_path: Path = Path("E:/thesis/outputs/Bennet1996_donut-small_ft_best")
    model_name: str = "Bennet1996/donut-small"
    paddleocr_vl_model_name: str = "PaddlePaddle/PaddleOCR-VL"
    paddleocr_vl_adapter_dir: Path | None = None
    # MRZ-путь для паспортов: дообученный LoRA-адаптер чтения машиночитаемой зоны
    passport_mrz_adapter_dir: Path | None = None
    passport_mrz_prompt: str = (
        "Read both lines of the passport machine readable zone (MRZ) exactly, "
        "line by line. Preserve all < characters."
    )
    passport_mrz_max_new_tokens: int = 100
    mrz_localize: bool = True
    mrz_postcorrect: bool = True
    qwen25_vl_model_name: str = "Qwen/Qwen2.5-VL-3B-Instruct"
    huggingface_home: Path = Path("E:/thesis/.hf-cache")
    max_length: int = 256
    image_width: int = 640
    image_height: int = 480
    device: str = "auto"
    process_one_job_at_a_time: bool = True

    frontend_origin: str = "http://localhost:5173"
    frontend_origin_alt: str = "http://127.0.0.1:5173"

    # Создавать таблицы при старте (для разработки). В продакшене с Alembic — false.
    auto_create_tables: bool = True

    # Очередь задач (RQ + Redis). False = последовательный воркер-поллер.
    use_task_queue: bool = False
    redis_url: str = "redis://localhost:6379/0"

    # Ограничения загрузки документов
    max_upload_mb: int = 20
    allowed_content_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/jpg",
        "image/webp",
        "image/tiff",
        "application/pdf",
    )

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def image_size(self) -> tuple[int, int]:
        return (self.image_width, self.image_height)

    @property
    def cors_origins(self) -> list[str]:
        origins = [self.frontend_origin, self.frontend_origin_alt]
        return list(dict.fromkeys(origin for origin in origins if origin))


@lru_cache
def get_settings() -> Settings:
    return Settings()
