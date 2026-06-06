from __future__ import annotations

import os
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock

import torch
from peft import PeftModel
from PIL import Image
from transformers import DonutProcessor, VisionEncoderDecoderModel

from app.core.config import get_settings
from app.schemas.documents import (
    DocumentType,
    InvoiceFields,
    MultiDocumentExtractionResult,
    PartyInfo,
    ProcessingEngine,
    ProcessingMeta,
    ValidationSummary,
)
from app.schemas.jobs import JobRequestOptions


FIELDS = [
    "invoice_no",
    "invoice_date",
    "seller",
    "client",
    "seller_tax_id",
    "client_tax_id",
    "iban",
    "total_net_worth",
    "total_vat",
    "total_gross_worth",
]
TASK_TOKEN = "<s_invoice>"


class DonutInvoiceExtractionEngine:
    _instance = None
    _lock = Lock()
    engine_name = ProcessingEngine.DONUT

    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.device = self._resolve_device(settings.device)
        self.processor, self.model = self._load_model()

    @classmethod
    def instance(cls) -> "DonutInvoiceExtractionEngine":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    def _resolve_device(self, configured: str) -> torch.device:
        if configured == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(configured)

    def _load_model(self):
        settings = self.settings
        model_name_or_path = str(settings.model_path)
        model_path = Path(model_name_or_path)
        cache_root = Path(settings.huggingface_home)
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_root)
        os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
        os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
        os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")
        processor = DonutProcessor.from_pretrained(
            model_name_or_path,
            use_fast=False,
            cache_dir=str(settings.huggingface_home),
        )

        if model_path.exists() and (model_path / "adapter_config.json").exists():
            base_model_name = (model_path / "base_model_name.txt").read_text(
                encoding="utf-8"
            ).strip()
            model = VisionEncoderDecoderModel.from_pretrained(
                base_model_name,
                cache_dir=str(settings.huggingface_home),
            )
            model.decoder.resize_token_embeddings(len(processor.tokenizer))
            model.decoder = PeftModel.from_pretrained(model.decoder, model_name_or_path)
            model.decoder = model.decoder.merge_and_unload()
        else:
            model = VisionEncoderDecoderModel.from_pretrained(
                model_name_or_path,
                cache_dir=str(settings.huggingface_home),
            )

        model = model.to(self.device).eval()
        return processor, model

    def predict(
        self,
        *,
        image: Image.Image,
        source_filename: str,
        request: JobRequestOptions,
    ) -> tuple[MultiDocumentExtractionResult, dict]:
        if request.document_type != DocumentType.INVOICE:
            raise ValueError("The current Donut engine supports only invoice extraction.")

        started_at = datetime.now(timezone.utc)
        started_ts = time.perf_counter()

        pixel_values = self.processor(
            images=image.resize(self.settings.image_size),
            return_tensors="pt",
            legacy=False,
        ).pixel_values.to(self.device)

        decoder_input_ids = torch.tensor(
            [[self.processor.tokenizer.convert_tokens_to_ids(TASK_TOKEN)]],
            device=self.device,
        )

        with torch.inference_mode():
            outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.settings.max_length,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
            )

        raw_text = self.processor.batch_decode(outputs, skip_special_tokens=False)[0]
        normalized = self._extract_schema(
            raw_text,
            source_filename=source_filename,
            started_at=started_at,
        )
        self._validate_prediction(normalized)

        raw_payload = {
            "generated_text": raw_text,
            "engine": self.engine_name.value,
            "requested_document_type": request.document_type.value,
        }
        normalized.raw_result = raw_payload
        normalized.processing_meta.inference_finished_at = datetime.now(timezone.utc)
        normalized.processing_meta.elapsed_ms = int(
            (time.perf_counter() - started_ts) * 1000
        )
        return normalized, raw_payload

    def _extract_field(self, text: str, field: str) -> str | None:
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

    def _extract_schema(
        self,
        text: str,
        *,
        source_filename: str,
        started_at: datetime,
    ) -> MultiDocumentExtractionResult:
        cleaned = text
        for token in ["<pad>", "<unk>", "<s>", "</s>"]:
            cleaned = cleaned.replace(token, "")

        schema = {field: self._extract_field(cleaned, field) for field in FIELDS}
        return MultiDocumentExtractionResult(
            document_type=DocumentType.INVOICE,
            source_filename=source_filename,
            fields=InvoiceFields(
                invoice_no=schema["invoice_no"],
                invoice_date=schema["invoice_date"],
                seller=PartyInfo(
                    name=schema["seller"],
                    tax_id=schema["seller_tax_id"],
                    iban=schema["iban"],
                ),
                buyer=PartyInfo(
                    name=schema["client"],
                    tax_id=schema["client_tax_id"],
                ),
                total_net=schema["total_net_worth"],
                total_tax=schema["total_vat"],
                total_gross=schema["total_gross_worth"],
            ),
            raw_text=re.sub(r"\s+", " ", cleaned).strip(),
            validation=ValidationSummary(is_valid=True, issues=[]),
            processing_meta=ProcessingMeta(
                engine=self.engine_name,
                model_name=self.settings.model_name,
                model_version=str(self.settings.model_path),
                inference_started_at=started_at,
                device=str(self.device),
                page_count=1,
            ),
        )

    def _validate_prediction(self, normalized: MultiDocumentExtractionResult) -> None:
        raw_cleaned_text = normalized.raw_text or ""
        fields = normalized.fields.model_dump()

        populated_fields = sum(
            1
            for value in self._flatten_values(fields)
            if value is not None and str(value).strip()
        )
        open_tags = re.findall(r"<s_[a-zA-Z0-9_.-]+>", raw_cleaned_text)
        tag_counts = Counter(open_tags)
        most_common_tag_count = tag_counts.most_common(1)[0][1] if tag_counts else 0

        if populated_fields == 0:
            raise ValueError(
                "Модель не смогла извлечь поля из документа. Похоже, результат генерации выродился."
            )

        if most_common_tag_count >= 20:
            raise ValueError(
                "Модель зациклилась при генерации ответа. Попробуйте другой документ или повторный запуск."
            )

    def _flatten_values(self, value):
        if isinstance(value, dict):
            for nested in value.values():
                yield from self._flatten_values(nested)
            return
        if isinstance(value, list):
            for nested in value:
                yield from self._flatten_values(nested)
            return
        yield value
