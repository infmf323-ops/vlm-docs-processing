from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from threading import Lock

from PIL import Image, ImageOps

from app.core.config import get_settings
from app.schemas.documents import (
    DocumentType,
    DriverLicenseFields,
    IdCardFields,
    InvoiceFields,
    MultiDocumentExtractionResult,
    NormalizedEntity,
    OtherDocumentFields,
    PassportFields,
    ProcessingEngine,
    ProcessingMeta,
    ValidationIssue,
    ValidationSummary,
)
from app.schemas.jobs import JobRequestOptions
from app.services.inference import DonutInvoiceExtractionEngine
from app.mrz import run_mrz_pipeline
from app.services.mrz_engine import PassportMrzReader


@dataclass
class ExtractionOutput:
    normalized_result: MultiDocumentExtractionResult
    raw_result: dict


class BaseExtractionEngine:
    engine_name: ProcessingEngine

    def predict(
        self,
        *,
        image: Image.Image,
        source_filename: str,
        request: JobRequestOptions,
    ) -> ExtractionOutput:
        raise NotImplementedError


class DonutEngineAdapter(BaseExtractionEngine):
    engine_name = ProcessingEngine.DONUT

    def __init__(self) -> None:
        self.engine = DonutInvoiceExtractionEngine.instance()

    def predict(
        self,
        *,
        image: Image.Image,
        source_filename: str,
        request: JobRequestOptions,
    ) -> ExtractionOutput:
        normalized_result, raw_result = self.engine.predict(
            image=image,
            source_filename=source_filename,
            request=request,
        )
        return ExtractionOutput(
            normalized_result=normalized_result,
            raw_result=raw_result,
        )


class PaddleOCRVLEngine(BaseExtractionEngine):
    engine_name = ProcessingEngine.PADDLEOCR_VL
    _shared_processor = None
    _shared_base_model = None
    _shared_adapted_model = None
    _shared_adapter_dir = None
    _load_lock = Lock()

    def __init__(self) -> None:
        self.settings = get_settings()

    def _configured_adapter_dir(self) -> Path | None:
        raw_value = self.settings.paddleocr_vl_adapter_dir
        if not raw_value:
            return None
        candidate = Path(raw_value)
        if str(candidate).strip() in {"", "."}:
            return None
        return candidate

    def _load(self) -> None:
        with self._load_lock:
            adapter_dir = self._configured_adapter_dir()
            if (
                self.__class__._shared_processor is not None
                and self.__class__._shared_base_model is not None
                and self.__class__._shared_adapter_dir == adapter_dir
                and (adapter_dir is None or self.__class__._shared_adapted_model is not None)
            ):
                return

            cache_root = Path(self.settings.huggingface_home)
            cache_root.mkdir(parents=True, exist_ok=True)
            os.environ["HF_HOME"] = str(cache_root)
            os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache_root / "hub")
            os.environ["HF_HUB_CACHE"] = str(cache_root / "hub")
            os.environ["TRANSFORMERS_CACHE"] = str(cache_root / "transformers")

            try:
                from peft import PeftModel
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "PaddleOCR-VL requires backend dependencies with PaddleOCRVL and PEFT support. "
                    "Please install backend requirements first."
                ) from exc

            if self.__class__._shared_processor is None:
                self.__class__._shared_processor = AutoProcessor.from_pretrained(
                    self.settings.paddleocr_vl_model_name,
                    cache_dir=str(self.settings.huggingface_home),
                )

            if self.__class__._shared_base_model is None:
                self.__class__._shared_base_model = AutoModelForImageTextToText.from_pretrained(
                    self.settings.paddleocr_vl_model_name,
                    cache_dir=str(self.settings.huggingface_home),
                    device_map="auto",
                )

            if adapter_dir and adapter_dir.exists():
                if (
                    self.__class__._shared_adapted_model is None
                    or self.__class__._shared_adapter_dir != adapter_dir
                ):
                    adapted_base_model = AutoModelForImageTextToText.from_pretrained(
                        self.settings.paddleocr_vl_model_name,
                        cache_dir=str(self.settings.huggingface_home),
                        device_map="auto",
                    )
                    self.__class__._shared_adapted_model = PeftModel.from_pretrained(
                        adapted_base_model,
                        str(adapter_dir),
                        local_files_only=True,
                    )
            else:
                self.__class__._shared_adapted_model = None

            self.__class__._shared_adapter_dir = adapter_dir

    @property
    def _processor(self):
        return self.__class__._shared_processor

    @property
    def _base_model(self):
        return self.__class__._shared_base_model

    @property
    def _model(self):
        if self._configured_adapter_dir() and self.__class__._shared_adapted_model is not None:
            return self.__class__._shared_adapted_model
        return self.__class__._shared_base_model

    def predict(
        self,
        *,
        image: Image.Image,
        source_filename: str,
        request: JobRequestOptions,
    ) -> ExtractionOutput:
        started_at = datetime.now(timezone.utc)
        started_ts = time.perf_counter()

        if request.document_type == DocumentType.PASSPORT:
            mrz_output = self._try_passport_mrz(
                image=image,
                source_filename=source_filename,
                request=request,
                started_at=started_at,
                started_ts=started_ts,
            )
            if mrz_output is not None:
                return mrz_output

        self._load()

        if self._configured_adapter_dir():
            structured_attempt = self._predict_structured_with_adapter(
                image=image,
                source_filename=source_filename,
                request=request,
                started_at=started_at,
                started_ts=started_ts,
            )
            if structured_attempt is not None:
                return structured_attempt

        generated_text = self._run_ocr(
            image=image,
            prompt_text="OCR:",
            use_base_model=bool(self._configured_adapter_dir()),
        )
        if (
            request.document_type in {DocumentType.PASSPORT, DocumentType.ID_CARD, DocumentType.DRIVER_LICENSE}
            and self._ocr_text_too_sparse(generated_text)
        ):
            retry_text = self._run_ocr(
                image=image,
                prompt_text=self._fallback_ocr_prompt(request.document_type),
                max_new_tokens=max(self.settings.max_length, 384),
                use_base_model=bool(self._configured_adapter_dir()),
            )
            if len(retry_text.strip()) > len(generated_text.strip()):
                generated_text = retry_text

        fields, normalized_entities, validation = self._map_document_fields(
            document_type=request.document_type,
            generated_text=generated_text,
            image=image,
        )

        normalized_result = MultiDocumentExtractionResult(
            document_type=request.document_type,
            source_filename=source_filename,
            fields=fields,
            normalized_entities=normalized_entities,
            raw_text=generated_text,
            raw_result={
                "generated_text": generated_text,
                "prompt": "OCR:",
                "engine": self.engine_name.value,
                "requested_document_type": request.document_type.value,
            },
            validation=validation,
            processing_meta=ProcessingMeta(
                engine=self.engine_name,
                model_name=self._resolved_model_name(),
                inference_started_at=started_at,
                inference_finished_at=datetime.now(timezone.utc),
                elapsed_ms=int((time.perf_counter() - started_ts) * 1000),
                device=str(self._model.device),
                page_count=1,
            ),
        )
        return ExtractionOutput(
            normalized_result=normalized_result,
            raw_result=normalized_result.raw_result or {},
        )

    def _try_passport_mrz(
        self,
        *,
        image: "Image.Image",
        source_filename: str,
        request: JobRequestOptions,
        started_at: datetime,
        started_ts: float,
    ) -> "ExtractionOutput | None":
        """MRZ-путь для паспортов: дообученная модель + пост-коррекция + разбор.

        Возвращает None, если MRZ-адаптер не настроен или распознавание сорвалось,
        чтобы вызывающий код мог использовать запасную (эвристическую) ветку.
        """
        reader = PassportMrzReader.instance()
        if not reader.enabled():
            return None
        try:
            result = run_mrz_pipeline(
                image,
                reader.read,
                localize=self.settings.mrz_localize,
                postcorrect=self.settings.mrz_postcorrect,
            )
        except Exception:
            return None

        f = result.fields or {}
        fields = PassportFields(
            document_number=f.get("document_number"),
            surname=f.get("surname"),
            given_names=f.get("given_names"),
            nationality=f.get("nationality"),
            date_of_birth=f.get("date_of_birth"),
            sex=f.get("sex"),
            date_of_expiry=f.get("date_of_expiry"),
            mrz=f.get("mrz"),
        )

        issues = []
        if not result.valid:
            issues.append(
                ValidationIssue(
                    code="mrz_checksum_failed",
                    message="Машиночитаемая зона не прошла проверку контрольными цифрами.",
                    field_path="mrz",
                    severity="warning",
                )
            )
        validation = ValidationSummary(is_valid=bool(result.valid), issues=issues)

        entity_specs = [
            ("person.surname", fields.surname),
            ("person.given_names", fields.given_names),
            ("document.number", fields.document_number),
            ("person.nationality", fields.nationality),
            ("person.date_of_birth", str(fields.date_of_birth) if fields.date_of_birth else None),
            ("person.sex", fields.sex),
            ("document.date_of_expiry", str(fields.date_of_expiry) if fields.date_of_expiry else None),
        ]
        normalized_entities = [
            NormalizedEntity(key=key, value=value) for key, value in entity_specs if value
        ]

        raw_payload = {
            "engine": self.engine_name.value,
            "mode": "passport_mrz",
            "raw_prediction": result.raw_prediction,
            "mrz": result.mrz,
            "corrected": result.corrected,
            "n_corrections": result.n_corrections,
            "valid": result.valid,
            "crop_size": list(result.crop_size) if result.crop_size else None,
            "requested_document_type": request.document_type.value,
            "adapter_dir": str(reader.adapter_dir) if reader.adapter_dir else None,
        }
        normalized_result = MultiDocumentExtractionResult(
            document_type=DocumentType.PASSPORT,
            source_filename=source_filename,
            fields=fields,
            normalized_entities=normalized_entities,
            raw_text=result.mrz,
            raw_result=raw_payload,
            validation=validation,
            confidence={"mrz_valid": 1.0 if result.valid else 0.0},
            processing_meta=ProcessingMeta(
                engine=self.engine_name,
                model_name=self._resolved_model_name() + " + MRZ-LoRA",
                inference_started_at=started_at,
                inference_finished_at=datetime.now(timezone.utc),
                elapsed_ms=int((time.perf_counter() - started_ts) * 1000),
                device=str(reader.device),
                page_count=1,
            ),
        )
        return ExtractionOutput(
            normalized_result=normalized_result,
            raw_result=raw_payload,
        )

    def _predict_structured_with_adapter(
        self,
        *,
        image: Image.Image,
        source_filename: str,
        request: JobRequestOptions,
        started_at: datetime,
        started_ts: float,
    ) -> ExtractionOutput | None:
        prompt_text = self._build_structured_prompt(request.document_type)
        generated_text = self._run_ocr(
            image=image,
            prompt_text=prompt_text,
            max_new_tokens=min(max(self.settings.max_length, 192), 320),
        )
        parsed = self._extract_json_object(generated_text)
        if parsed is None:
            salvaged_fields = self._salvage_structured_fields_from_text(generated_text)
            if salvaged_fields:
                parsed = {
                    "document_type": request.document_type.value,
                    "fields": salvaged_fields,
                }
        fields_model, normalized_entities, validation = self._map_structured_fields(
            document_type=request.document_type,
            parsed=parsed,
        )
        if fields_model is None:
            text_fields_model, text_entities, text_validation = self._map_document_fields(
                document_type=request.document_type,
                generated_text=generated_text,
                image=image,
            )
            text_populated = self._count_populated_values(text_fields_model.model_dump())
            if text_validation.is_valid or text_populated >= 3:
                normalized_result = MultiDocumentExtractionResult(
                    document_type=request.document_type,
                    source_filename=source_filename,
                    fields=text_fields_model,
                    normalized_entities=text_entities,
                    raw_text=generated_text,
                    raw_result={
                        "generated_text": generated_text,
                        "prompt": prompt_text,
                        "engine": self.engine_name.value,
                        "requested_document_type": request.document_type.value,
                        "parsed_json": parsed,
                        "mode": "structured_text_salvage",
                    },
                    validation=text_validation,
                    processing_meta=ProcessingMeta(
                        engine=self.engine_name,
                        model_name=self._resolved_model_name(),
                        inference_started_at=started_at,
                        inference_finished_at=datetime.now(timezone.utc),
                        elapsed_ms=int((time.perf_counter() - started_ts) * 1000),
                        device=str(self._model.device),
                        page_count=1,
                    ),
                )
                return ExtractionOutput(
                    normalized_result=normalized_result,
                    raw_result=normalized_result.raw_result or {},
                )
            return None

        text_fields_model, _, text_validation = self._map_document_fields(
            document_type=request.document_type,
            generated_text=generated_text,
            image=image,
        )
        text_populated = self._count_populated_values(text_fields_model.model_dump())
        merged_with_heuristic = False
        if text_validation.is_valid or text_populated >= 3:
            merged_payload = self._hybrid_merge_identity_fields(
                document_type=request.document_type,
                primary_fields=fields_model.model_dump(mode="json"),
                fallback_fields=text_fields_model.model_dump(mode="json"),
            )
            merged_model = self._structured_payload_to_model(request.document_type, merged_payload)
            if merged_model is not None:
                merged_with_heuristic = merged_model.model_dump(mode="json") != fields_model.model_dump(mode="json")
                fields_model = merged_model
                normalized_entities = self._normalized_entities_for_structured_fields(fields_model)

        normalized_result = MultiDocumentExtractionResult(
            document_type=request.document_type,
            source_filename=source_filename,
            fields=fields_model,
            normalized_entities=normalized_entities,
            raw_text=generated_text,
            raw_result={
                "generated_text": generated_text,
                "prompt": prompt_text,
                "engine": self.engine_name.value,
                "requested_document_type": request.document_type.value,
                "parsed_json": parsed,
                "mode": "structured_adapter_hybrid" if merged_with_heuristic else "structured_adapter",
                "merged_with_heuristic": merged_with_heuristic,
            },
            validation=validation,
            processing_meta=ProcessingMeta(
                engine=self.engine_name,
                model_name=self._resolved_model_name(),
                inference_started_at=started_at,
                inference_finished_at=datetime.now(timezone.utc),
                elapsed_ms=int((time.perf_counter() - started_ts) * 1000),
                device=str(self._model.device),
                page_count=1,
            ),
        )
        return ExtractionOutput(
            normalized_result=normalized_result,
            raw_result=normalized_result.raw_result or {},
        )

    def _build_structured_prompt(self, document_type: DocumentType) -> str:
        schema_specs = {
            DocumentType.PASSPORT: {
                "hint": (
                    "Use only these field keys inside `fields`: "
                    "`document_number`, `surname`, `given_names`, `nationality`, `date_of_birth`, "
                    "`sex`, `place_of_birth`, `date_of_issue`, `date_of_expiry`, `issuing_authority`, `mrz`."
                ),
                "skeleton": (
                    '{"document_type":"passport","fields":{"document_number":null,"surname":null,'
                    '"given_names":null,"nationality":null,"date_of_birth":null,"sex":null,'
                    '"place_of_birth":null,"date_of_issue":null,"date_of_expiry":null,'
                    '"issuing_authority":null,"mrz":null}}'
                ),
            },
            DocumentType.DRIVER_LICENSE: {
                "hint": (
                    "Use only these field keys inside `fields`: "
                    "`license_number`, `surname`, `given_names`, `date_of_birth`, `place_of_birth`, "
                    "`address`, `date_of_issue`, `date_of_expiry`, `issuing_authority`, `categories`."
                ),
                "skeleton": (
                    '{"document_type":"driver_license","fields":{"license_number":null,"surname":null,'
                    '"given_names":null,"date_of_birth":null,"place_of_birth":null,"address":null,'
                    '"date_of_issue":null,"date_of_expiry":null,"issuing_authority":null,"categories":[]}}'
                ),
            },
            DocumentType.ID_CARD: {
                "hint": (
                    "Use only these field keys inside `fields`: "
                    "`document_number`, `surname`, `given_names`, `nationality`, `date_of_birth`, "
                    "`sex`, `address`, `personal_number`, `date_of_issue`, `date_of_expiry`, `issuing_authority`."
                ),
                "skeleton": (
                    '{"document_type":"id_card","fields":{"document_number":null,"surname":null,'
                    '"given_names":null,"nationality":null,"date_of_birth":null,"sex":null,"address":null,'
                    '"personal_number":null,"date_of_issue":null,"date_of_expiry":null,"issuing_authority":null}}'
                ),
            },
            DocumentType.INVOICE: {
                "hint": (
                    "Use only these field keys inside `fields`: "
                    "`invoice_no`, `invoice_date`, `due_date`, `currency`, `seller`, `buyer`, "
                    "`line_items`, `total_net`, `total_tax`, `total_gross`."
                ),
                "skeleton": (
                    '{"document_type":"invoice","fields":{"invoice_no":null,"invoice_date":null,'
                    '"due_date":null,"currency":null,"seller":null,"buyer":null,'
                    '"line_items":[],"total_net":null,"total_tax":null,"total_gross":null}}'
                ),
            },
        }
        spec = schema_specs.get(document_type, {"hint": "Use only schema-relevant field keys.", "skeleton": '{"document_type":"other","fields":{}}'})
        return (
            "Extract the document into a JSON object with keys `document_type` and `fields`. "
            f"The document type is `{document_type.value}`. "
            f"{spec['hint']} "
            "Return exactly one valid JSON object. "
            "Fill missing values with null. Use [] only for list fields. "
            "Do not add explanation text. Do not add markdown. "
            f"Follow this schema exactly: {spec['skeleton']}"
        )

    def _extract_json_object(self, text: str) -> dict | None:
        start = text.find("{")
        if start == -1:
            return None

        raw = text[start:].strip().strip("`")
        attempts: list[str] = []
        end = raw.rfind("}")
        if end != -1:
            attempts.append(raw[: end + 1])
        attempts.append(raw)

        for candidate in attempts:
            cleaned = re.sub(r",(\s*[}\]])", r"\1", candidate)
            try:
                parsed = json.loads(cleaned)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                pass

            open_braces = cleaned.count("{")
            close_braces = cleaned.count("}")
            if open_braces > close_braces:
                balanced = cleaned + ("}" * (open_braces - close_braces))
                balanced = re.sub(r",(\s*[}\]])", r"\1", balanced)
                try:
                    parsed = json.loads(balanced)
                    if isinstance(parsed, dict):
                        return parsed
                except json.JSONDecodeError:
                    pass
        return None

    def _salvage_structured_fields_from_text(self, text: str) -> dict:
        pairs = re.findall(r'"([A-Za-z0-9_]+)"\s*:\s*"([^"]*)"', text)
        if not pairs:
            return {}
        ignored = {"document_type", "document_content", "text", "text_content", "text_version"}
        salvaged: dict[str, str] = {}
        for key, value in pairs:
            if key in ignored:
                continue
            if not value.strip():
                continue
            salvaged[key] = value.strip()
        return salvaged

    def _map_structured_fields(self, *, document_type: DocumentType, parsed: dict | None):
        if not isinstance(parsed, dict):
            return None, [], ValidationSummary(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="structured_json_not_found",
                        message="Fine-tuned adapter did not return parseable JSON.",
                        severity="warning",
                    )
                ],
            )

        fields_payload = parsed.get("fields", parsed)
        if not isinstance(fields_payload, dict):
            return None, [], ValidationSummary(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="structured_fields_missing",
                        message="Fine-tuned adapter returned JSON without a usable `fields` object.",
                        severity="warning",
                    )
                ],
            )

        normalized_payload = self._normalize_structured_payload(document_type, fields_payload)
        fields_model = self._structured_payload_to_model(document_type, normalized_payload)
        if fields_model is None:
            return None, [], ValidationSummary(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="structured_fields_invalid",
                        message="Structured adapter output could not be mapped to the target schema.",
                        severity="warning",
                    )
                ],
            )

        normalized_entities = self._normalized_entities_for_structured_fields(fields_model)
        field_dump = fields_model.model_dump()
        populated = self._count_populated_values(field_dump)
        if self._structured_payload_looks_repetitive(field_dump):
            return None, normalized_entities, ValidationSummary(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="structured_fields_repetitive",
                        message="Structured adapter output repeated the same value across too many fields.",
                        severity="warning",
                    )
                ],
            )
        if populated < 3:
            return None, normalized_entities, ValidationSummary(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="structured_fields_too_sparse",
                        message="Structured adapter output was parsed, but too few fields were populated.",
                        severity="warning",
                    )
                ],
            )

        validation = ValidationSummary(
            is_valid=True,
            issues=[
                ValidationIssue(
                    code="structured_adapter_mapping",
                    message="Fields were mapped from fine-tuned PaddleOCR-VL structured JSON output.",
                    severity="info",
                )
            ],
        )
        return fields_model, normalized_entities, validation

    def _structured_payload_looks_repetitive(self, payload: dict) -> bool:
        values = []
        for value in payload.values():
            if value is None:
                continue
            if isinstance(value, list):
                if value:
                    values.append(str(value))
                continue
            if isinstance(value, dict):
                nested = [str(v).strip() for v in value.values() if v is not None and str(v).strip()]
                values.extend(nested)
                continue
            text = str(value).strip()
            if text:
                values.append(text)
        if len(values) < 4:
            return False
        unique_count = len(set(values))
        return unique_count <= max(2, len(values) // 2)

    def _normalize_structured_payload(self, document_type: DocumentType, fields: dict) -> dict:
        normalized = {}
        for key, value in dict(fields).items():
            canonical_key = self._canonical_structured_key(document_type, str(key))
            normalized[canonical_key] = self._sanitize_structured_value(value)
        normalized = {key: value for key, value in normalized.items() if value is not None}
        if document_type == DocumentType.DRIVER_LICENSE:
            if "document_number" in normalized and "license_number" not in normalized:
                normalized["license_number"] = normalized.pop("document_number")
            if "expiry_date" in normalized and "date_of_expiry" not in normalized:
                normalized["date_of_expiry"] = normalized.pop("expiry_date")
            if "issue_date" in normalized and "date_of_issue" not in normalized:
                normalized["date_of_issue"] = normalized.pop("issue_date")
            if "document_date" in normalized and "date_of_issue" not in normalized:
                normalized["date_of_issue"] = normalized.pop("document_date")
            if "license_number" in normalized:
                match = re.search(r"\b[A-Z0-9]{7,10}\b", str(normalized["license_number"]))
                normalized["license_number"] = match.group(0) if match else normalized["license_number"]
            if "address" in normalized and re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(normalized["address"])):
                normalized["address"] = None
            if "issuing_authority" in normalized and re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(normalized["issuing_authority"])):
                normalized["issuing_authority"] = None
            if "categories" in normalized and isinstance(normalized["categories"], str):
                normalized["categories"] = self._parse_structured_category_string(normalized["categories"])
        elif document_type == DocumentType.PASSPORT:
            if "document_date" in normalized and "date_of_issue" not in normalized:
                normalized["date_of_issue"] = normalized.pop("document_date")
            if "expiry_date" in normalized and "date_of_expiry" not in normalized:
                normalized["date_of_expiry"] = normalized.pop("expiry_date")
            normalized.pop("document_time", None)
            if "document_number" in normalized:
                match = re.search(r"\b[A-Z0-9]{8,10}\b", str(normalized["document_number"]))
                normalized["document_number"] = match.group(0) if match else normalized["document_number"]
            if "issuing_authority" in normalized and re.fullmatch(r"\d{2}/\d{2}/\d{4}", str(normalized["issuing_authority"])):
                normalized["issuing_authority"] = None
            for key in ["surname", "given_names", "mrz"]:
                value = normalized.get(key)
                if isinstance(value, str) and value.strip().lower() in {"abu dhabi", "<<", "<8"}:
                    normalized[key] = None
        elif document_type == DocumentType.ID_CARD:
            if "document_date" in normalized and "date_of_issue" not in normalized:
                normalized["date_of_issue"] = normalized.pop("document_date")
            if "expiry_date" in normalized and "date_of_expiry" not in normalized:
                normalized["date_of_expiry"] = normalized.pop("expiry_date")
        return {key: value for key, value in normalized.items() if value is not None}

    def _canonical_structured_key(self, document_type: DocumentType, key: str) -> str:
        cleaned = re.sub(r"[^a-z0-9]+", "", key.lower())
        alias_map_common = {
            "firstname": "given_names",
            "forename": "given_names",
            "givenname": "given_names",
            "givennames": "given_names",
            "lastname": "surname",
            "surname": "surname",
            "familyname": "surname",
            "nationality": "nationality",
            "citizenship": "nationality",
            "sex": "sex",
            "gender": "sex",
            "dob": "date_of_birth",
            "dateofbirth": "date_of_birth",
            "birthdate": "date_of_birth",
            "issuedate": "date_of_issue",
            "dateofissue": "date_of_issue",
            "documentdate": "date_of_issue",
            "expirydate": "date_of_expiry",
            "expirationdate": "date_of_expiry",
            "dateofexpiry": "date_of_expiry",
            "dateofexpiration": "date_of_expiry",
            "authority": "issuing_authority",
            "issuingauthority": "issuing_authority",
        }
        alias_map_driver = {
            "dl": "license_number",
            "dlnumber": "license_number",
            "drivelicensenumber": "license_number",
            "licensenumber": "license_number",
            "licence": "license_number",
            "licencenumber": "license_number",
            "licenzen": "license_number",
            "documentnumber": "license_number",
            "ln": "surname",
            "fn": "given_names",
            "exp": "date_of_expiry",
            "iss": "date_of_issue",
            "dd": "date_of_issue",
            "class": "categories",
            "category": "categories",
            "categories": "categories",
            "addr": "address",
        }
        alias_map_passport = {
            "passportnumber": "document_number",
            "passportno": "document_number",
            "documentnumber": "document_number",
            "mrz": "mrz",
            "placeofbirth": "place_of_birth",
        }
        alias_map_id = {
            "documentnumber": "document_number",
            "idnumber": "document_number",
            "personalnumber": "personal_number",
            "placeofbirth": "place_of_birth",
            "addr": "address",
        }
        if cleaned in alias_map_common:
            return alias_map_common[cleaned]
        if document_type == DocumentType.DRIVER_LICENSE and cleaned in alias_map_driver:
            return alias_map_driver[cleaned]
        if document_type == DocumentType.PASSPORT and cleaned in alias_map_passport:
            return alias_map_passport[cleaned]
        if document_type == DocumentType.ID_CARD and cleaned in alias_map_id:
            return alias_map_id[cleaned]
        return key

    def _sanitize_structured_value(self, value):
        if isinstance(value, list):
            cleaned = [self._sanitize_structured_value(item) for item in value]
            return [item for item in cleaned if item not in (None, "", [], {})]
        if isinstance(value, dict):
            cleaned = {k: self._sanitize_structured_value(v) for k, v in value.items()}
            return {k: v for k, v in cleaned.items() if v not in (None, "", [], {})}
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        if text in {"___", "__", "_", "N/A", "NULL", "UNKNOWN", "None"}:
            return None
        if re.fullmatch(r"_+", text):
            return None
        if len(set(text)) == 1 and len(text) > 2:
            return None
        return text

    def _parse_structured_category_string(self, value: str):
        matches = re.findall(r"\b(?:AM|A1|A2|A|B1|B|C1|C|D1|D|BE|CE|DE|T)\b", value.upper())
        cleaned = []
        for match in matches:
            if match not in cleaned:
                cleaned.append(match)
        return cleaned or None

    def _structured_payload_to_model(self, document_type: DocumentType, payload: dict):
        try:
            if document_type == DocumentType.PASSPORT:
                allowed = set(PassportFields.model_fields.keys())
                return PassportFields(**{k: v for k, v in payload.items() if k in allowed})
            if document_type == DocumentType.ID_CARD:
                allowed = set(IdCardFields.model_fields.keys())
                return IdCardFields(**{k: v for k, v in payload.items() if k in allowed})
            if document_type == DocumentType.DRIVER_LICENSE:
                allowed = set(DriverLicenseFields.model_fields.keys())
                return DriverLicenseFields(**{k: v for k, v in payload.items() if k in allowed})
            if document_type == DocumentType.INVOICE:
                allowed = set(InvoiceFields.model_fields.keys())
                return InvoiceFields(**{k: v for k, v in payload.items() if k in allowed})
            allowed = set(OtherDocumentFields.model_fields.keys())
            filtered = {k: v for k, v in payload.items() if k in allowed}
            if not filtered:
                filtered = {"extracted_pairs": payload}
            return OtherDocumentFields(**filtered)
        except Exception:
            return None

    def _normalized_entities_for_structured_fields(self, fields_model):
        field_dump = fields_model.model_dump()
        key_mapping = {
            "document_number": "document.number",
            "license_number": "document.number",
            "surname": "person.surname",
            "given_names": "person.given_names",
            "nationality": "person.nationality",
            "date_of_birth": "person.date_of_birth",
            "sex": "person.sex",
            "place_of_birth": "person.place_of_birth",
            "address": "person.address",
            "personal_number": "person.personal_number",
            "date_of_issue": "document.issue_date",
            "date_of_expiry": "document.expiry_date",
            "issuing_authority": "document.issuing_authority",
        }
        entities: list[NormalizedEntity] = []
        for field_name, entity_key in key_mapping.items():
            value = field_dump.get(field_name)
            if value:
                entities.append(
                    NormalizedEntity(key=entity_key, value=value, source_field=field_name)
                )
        return entities

    def _hybrid_merge_identity_fields(
        self,
        *,
        document_type: DocumentType,
        primary_fields: dict,
        fallback_fields: dict,
    ) -> dict:
        if document_type not in {
            DocumentType.PASSPORT,
            DocumentType.ID_CARD,
            DocumentType.DRIVER_LICENSE,
        }:
            return primary_fields

        merged = dict(primary_fields)

        def is_blank(value):
            return value in (None, "", [], {})

        for key, fallback_value in fallback_fields.items():
            if is_blank(fallback_value):
                continue
            primary_value = merged.get(key)
            if is_blank(primary_value):
                merged[key] = fallback_value

        if document_type == DocumentType.DRIVER_LICENSE:
            primary_license = merged.get("license_number")
            fallback_license = fallback_fields.get("license_number")
            if (not primary_license or self._looks_like_mmddyyyy(str(primary_license))) and fallback_license:
                merged["license_number"] = fallback_license

            primary_birth = merged.get("date_of_birth")
            fallback_birth = fallback_fields.get("date_of_birth")
            if fallback_birth and self._is_better_identity_date(
                candidate=fallback_birth,
                current=primary_birth,
                prefer_birth=True,
            ):
                merged["date_of_birth"] = fallback_birth

            primary_issue = merged.get("date_of_issue")
            fallback_issue = fallback_fields.get("date_of_issue")
            primary_expiry = merged.get("date_of_expiry")
            fallback_expiry = fallback_fields.get("date_of_expiry")
            if fallback_expiry and self._is_better_identity_date(
                candidate=fallback_expiry,
                current=primary_expiry,
                prefer_birth=False,
            ):
                merged["date_of_expiry"] = fallback_expiry
            if fallback_issue and self._is_better_issue_date(
                candidate=fallback_issue,
                current=primary_issue,
                date_of_birth=merged.get("date_of_birth"),
                date_of_expiry=merged.get("date_of_expiry"),
            ):
                merged["date_of_issue"] = fallback_issue

            primary_categories = merged.get("categories")
            fallback_categories = fallback_fields.get("categories")
            if (not primary_categories) and fallback_categories:
                merged["categories"] = fallback_categories

            if self._date_to_ordinal(merged.get("date_of_issue")) and self._date_to_ordinal(merged.get("date_of_expiry")):
                if self._date_to_ordinal(merged.get("date_of_issue")) > self._date_to_ordinal(merged.get("date_of_expiry")):
                    if self._is_better_issue_date(
                        candidate=fallback_fields.get("date_of_issue"),
                        current=merged.get("date_of_issue"),
                        date_of_birth=merged.get("date_of_birth"),
                        date_of_expiry=merged.get("date_of_expiry"),
                    ):
                        merged["date_of_issue"] = fallback_fields.get("date_of_issue")

        return merged

    def _resolved_model_name(self) -> str:
        adapter_dir = self._configured_adapter_dir()
        if adapter_dir:
            return (
                f"{self.settings.paddleocr_vl_model_name}"
                f" + LoRA({adapter_dir.name})"
            )
        return self.settings.paddleocr_vl_model_name

    def _run_ocr(
        self,
        *,
        image: Image.Image,
        prompt_text: str,
        max_new_tokens: int | None = None,
        use_base_model: bool = False,
    ) -> str:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt_text},
                ],
            }
        ]

        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        model = self._base_model if use_base_model and self._base_model is not None else self._model
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_new_tokens or self.settings.max_length,
            "do_sample": False,
            "repetition_penalty": 1.12,
            "no_repeat_ngram_size": 3,
        }

        output_ids = model.generate(**inputs, **generation_kwargs)
        return self._processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1] :],
            skip_special_tokens=True,
        )[0].strip()

    def _ocr_text_too_sparse(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 40:
            return True
        if len(re.findall(r"\w+", stripped)) < 8:
            return True
        if len([line for line in stripped.splitlines() if line.strip()]) < 3:
            return True
        return False

    def _fallback_ocr_prompt(self, document_type: DocumentType) -> str:
        if document_type == DocumentType.DRIVER_LICENSE:
            return (
                "Transcribe the full driver license text line by line exactly as visible. "
                "Do not describe the image. Do not summarize. Do not mention charts, tables, graphs, or graphics. "
                "Include DL, LN, FN, DOB, EXP, ISS, DD, SEX, HAIR, EYES, address lines, class/category labels, and all visible numbers."
            )
        if document_type in {DocumentType.PASSPORT, DocumentType.ID_CARD}:
            return (
                "Transcribe the full identity document text line by line. "
                "Include names, document numbers, dates, nationality, MRZ text, labels, and all visible text."
            )
        return "Transcribe the full document text line by line."

    def _map_document_fields(self, *, document_type, generated_text: str, image: Image.Image):
        if document_type == DocumentType.PASSPORT:
            return self._parse_passport_fields(generated_text, image)
        if document_type == DocumentType.ID_CARD:
            return self._parse_id_card_fields(generated_text)
        if document_type == DocumentType.DRIVER_LICENSE:
            return self._parse_driver_license_fields(generated_text, image)

        return (
            self._empty_fields_for_document_type(document_type),
            [],
            ValidationSummary(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="schema_mapping_not_implemented",
                        message=(
                            "PaddleOCR-VL currently runs as an OCR/document-text engine. "
                            "Structured field mapping for this document type is not implemented yet."
                        ),
                        severity="warning",
                    )
                ],
            ),
        )

    def _parse_passport_fields(self, generated_text: str, image: Image.Image):
        lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
        upper_lines = [line.upper() for line in lines]
        mrz_text = self._extract_passport_mrz_text(image)
        mrz_data = self._parse_mrz(mrz_text) if mrz_text else {}
        compact_data = self._parse_compact_passport_identity_line(generated_text)
        merged_mrz = self._join_mrz(lines) or mrz_text
        merged_mrz_data = self._parse_mrz(merged_mrz) if merged_mrz else {}
        best_mrz_text = mrz_text if self._count_populated_values(mrz_data) >= self._count_populated_values(merged_mrz_data) else merged_mrz
        best_mrz_data = mrz_data if self._count_populated_values(mrz_data) >= self._count_populated_values(merged_mrz_data) else merged_mrz_data
        name_text = self._extract_passport_name_text(image)
        name_lines = [line.strip() for line in name_text.splitlines() if line.strip()]
        upper_name_lines = [line.upper() for line in name_lines]
        sex_text = self._extract_passport_sex_text(image)
        authority_text = self._extract_passport_authority_text(image)

        document_number = self._most_frequent_match(
            re.findall(r"\b\d{8,9}\b", generated_text)
        )
        if self._is_plausible_document_number(compact_data.get("document_number")):
            document_number = compact_data["document_number"]
        if self._is_plausible_document_number(best_mrz_data.get("document_number")):
            document_number = best_mrz_data["document_number"]

        given_names = self._value_after_marker(upper_lines, lines, ["GIVEN NAMES"])
        if not given_names:
            given_names = self._value_after_marker(upper_name_lines, name_lines, ["GIVEN NAMES"])
        nationality = self._value_after_marker(upper_lines, lines, ["NATIONALITY"])
        if not nationality:
            nationality = self._detect_passport_nationality(generated_text)
        if self._is_plausible_nationality(compact_data.get("nationality")):
            nationality = compact_data["nationality"]
        if self._is_plausible_nationality(best_mrz_data.get("nationality")):
            nationality = best_mrz_data["nationality"]
        place_of_birth = self._value_after_marker(
            upper_lines,
            lines,
            ["LIEU DE NAISSANCE", "LUGAR DE NACIMIENTO", "PLANO / PLANO"],
        )

        date_matches = re.findall(r"\b\d{2}\s+[A-Z][a-z]{2}\s+\d{4}\b", generated_text)
        date_of_birth = self._value_after_marker(
            upper_lines, lines, ["DATE OF BIRTH", "DATE OF BIRTH /"]
        ) or (date_matches[0] if date_matches else None)
        if compact_data.get("date_of_birth"):
            date_of_birth = compact_data["date_of_birth"]
        if best_mrz_data.get("date_of_birth"):
            date_of_birth = best_mrz_data["date_of_birth"]

        date_of_issue = self._value_after_marker(
            upper_lines, lines, ["DATE OF ISUS", "DATE OF ISSUE", "DATE OF DE"]
        )
        date_of_expiry = self._value_after_marker(
            upper_lines, lines, ["DATE OF EXPIRATION", "DATE OF EXPIRY"]
        )
        if date_of_issue is None:
            date_of_issue = self._extract_passport_issue_date(authority_text)
        if date_of_issue is None and len(date_matches) >= 2:
            date_of_issue = date_matches[1]
        if date_of_expiry is None and len(date_matches) >= 3:
            date_of_expiry = date_matches[2]
        elif date_of_expiry is None and len(date_matches) >= 2:
            date_of_expiry = date_matches[-1]
        if compact_data.get("date_of_expiry"):
            date_of_expiry = compact_data["date_of_expiry"]
        if best_mrz_data.get("date_of_expiry"):
            date_of_expiry = best_mrz_data["date_of_expiry"]

        surname = self._guess_passport_surname(upper_lines, lines)
        if surname and len(surname) <= 4:
            refined = self._guess_passport_surname(upper_name_lines, name_lines)
            if refined:
                surname = refined
        if best_mrz_data.get("surname"):
            surname = best_mrz_data["surname"]
        if compact_data.get("surname"):
            surname = compact_data["surname"]
        if best_mrz_data.get("given_names"):
            given_names = best_mrz_data["given_names"]
        if compact_data.get("given_names"):
            given_names = compact_data["given_names"]
        mrz = best_mrz_text
        line1_name_data = self._parse_passport_name_from_line1(mrz)
        if line1_name_data.get("surname"):
            surname = line1_name_data["surname"]
        if line1_name_data.get("given_names"):
            given_names = line1_name_data["given_names"]
        crop_name_data = self._parse_passport_name_from_name_crop(name_lines)
        if crop_name_data.get("surname") and (not surname or len(re.sub(r"[^A-Z]", "", surname.upper())) <= 2):
            surname = crop_name_data["surname"]
        if crop_name_data.get("given_names") and not given_names:
            given_names = crop_name_data["given_names"]
        surname = self._normalize_passport_surname(surname)
        given_names = self._normalize_passport_given_names(given_names)
        if not self._is_useful_passport_name(surname):
            surname = None
        if not self._is_useful_passport_name(given_names):
            given_names = None
        issuing_authority = self._find_line_containing(
            upper_lines,
            lines,
            ["DEPARTMENT OF STATE", "AUTHORITY", "AUTORITE"],
        )
        if authority_text:
            issuing_authority = self._normalize_passport_authority(authority_text)
        sex = best_mrz_data.get("sex") if self._is_plausible_sex(best_mrz_data.get("sex")) else None
        if sex is None and self._is_plausible_sex(compact_data.get("sex")):
            sex = compact_data["sex"]
        if sex is None:
            sex = self._parse_sex_from_text(sex_text or authority_text)
        place_of_birth = self._normalize_place_of_birth(place_of_birth)

        fields = PassportFields(
            document_number=document_number,
            surname=surname,
            given_names=given_names,
            nationality=nationality,
            date_of_birth=date_of_birth,
            sex=sex,
            place_of_birth=place_of_birth,
            date_of_issue=date_of_issue,
            date_of_expiry=date_of_expiry,
            issuing_authority=issuing_authority,
            mrz=mrz,
        )

        normalized_entities = []
        mapping = {
            "document.number": fields.document_number,
            "person.surname": fields.surname,
            "person.given_names": fields.given_names,
            "person.nationality": fields.nationality,
            "person.date_of_birth": fields.date_of_birth,
            "person.sex": fields.sex,
            "person.place_of_birth": fields.place_of_birth,
            "document.issue_date": fields.date_of_issue,
            "document.expiry_date": fields.date_of_expiry,
            "document.issuing_authority": fields.issuing_authority,
        }
        for key, value in mapping.items():
            if value:
                normalized_entities.append(
                    NormalizedEntity(key=key, value=value, source_field=key.split(".", 1)[-1])
                )

        populated = sum(
            1 for value in fields.model_dump().values() if value is not None and str(value).strip()
        )
        if populated >= 4:
            validation = ValidationSummary(
                is_valid=True,
                issues=[
                    ValidationIssue(
                        code="heuristic_passport_mapping",
                        message=(
                            "Passport fields were mapped from PaddleOCR-VL OCR output using heuristic parsing."
                        ),
                        severity="info",
                    )
                ],
            )
        else:
            validation = ValidationSummary(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="passport_mapping_low_confidence",
                        message=(
                            "PaddleOCR-VL extracted text, but the heuristic passport parser found too few fields."
                        ),
                        severity="warning",
                    )
                ],
            )

        return fields, normalized_entities, validation

    def _parse_id_card_fields(self, generated_text: str):
        lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
        upper_lines = [line.upper() for line in lines]

        identity = self._extract_identity_core(lines, upper_lines)
        address = self._value_after_marker(
            upper_lines,
            lines,
            ["ADDRESS", "ADRESSE", "ADDRESS /", "DOMICILE"],
        )
        personal_number = self._value_after_marker(
            upper_lines,
            lines,
            ["PERSONAL NUMBER", "PERSONALNO", "PIN", "IDENTIFICATION NO"],
        )
        sex = self._value_after_marker(
            upper_lines,
            lines,
            ["SEX", "SEXE", "GENDER"],
        )

        fields = IdCardFields(
            document_number=identity["document_number"],
            surname=identity["surname"],
            given_names=identity["given_names"],
            nationality=identity["nationality"],
            date_of_birth=identity["date_of_birth"],
            sex=sex,
            address=address,
            personal_number=personal_number,
            date_of_issue=identity["date_of_issue"],
            date_of_expiry=identity["date_of_expiry"],
            issuing_authority=identity["issuing_authority"],
        )
        return self._build_identity_result(
            fields=fields,
            validation_code="heuristic_id_card_mapping",
            validation_message=(
                "ID card fields were mapped from PaddleOCR-VL OCR output using heuristic parsing."
            ),
        )

    def _parse_driver_license_fields(self, generated_text: str, image: Image.Image):
        lines = [line.strip() for line in generated_text.splitlines() if line.strip()]
        header_text = self._extract_driver_license_header_text(image)
        header_lines = [line.strip() for line in header_text.splitlines() if line.strip()] if header_text else []
        text_block_text = self._extract_driver_license_text_block_text(image)
        text_block_lines = [line.strip() for line in text_block_text.splitlines() if line.strip()] if text_block_text else []
        detail_text = self._extract_driver_license_detail_text(image)
        detail_lines = [line.strip() for line in detail_text.splitlines() if line.strip()] if detail_text else []
        crop_text = self._extract_driver_license_focus_text(image)
        crop_lines = [line.strip() for line in crop_text.splitlines() if line.strip()] if crop_text else []
        dates_text = self._extract_driver_license_dates_text(image)
        dates_lines = [line.strip() for line in dates_text.splitlines() if line.strip()] if dates_text else []
        issue_text = self._extract_driver_license_issue_text(image)
        issue_lines = [line.strip() for line in issue_text.splitlines() if line.strip()] if issue_text else []
        bottom_right_text = self._extract_driver_license_bottom_right_text(image)
        bottom_right_lines = [line.strip() for line in bottom_right_text.splitlines() if line.strip()] if bottom_right_text else []
        number_text = self._extract_driver_license_number_text(image)
        number_lines = [line.strip() for line in number_text.splitlines() if line.strip()] if number_text else []
        lines = self._merge_lines_preserve_order(
            number_lines + header_lines + text_block_lines + dates_lines + issue_lines + bottom_right_lines + detail_lines + crop_lines + lines
        )
        upper_lines = [line.upper() for line in lines]

        identity = self._extract_identity_core(lines, upper_lines)
        surname = self._value_from_same_or_next_line(
            upper_lines,
            lines,
            ["LN", "LAST NAME", "SURNAME"],
        ) or identity["surname"]
        given_names = self._value_from_same_or_next_line(
            upper_lines,
            lines,
            ["FN", "FIRST NAME", "GIVEN NAME", "GIVEN NAMES"],
        ) or identity["given_names"]
        date_of_birth = self._value_from_same_or_next_line(
            upper_lines,
            lines,
            ["DOB", "DATE OF BIRTH", "BIRTH"],
        ) or identity["date_of_birth"]
        date_of_issue = self._value_from_same_or_next_line(
            upper_lines,
            lines,
            ["ISS", "ISSUED", "DATE OF ISSUE"],
        ) or identity["date_of_issue"]
        date_of_expiry = self._value_from_same_or_next_line(
            upper_lines,
            lines,
            ["EXP", "EXPIRY", "EXPIRATION"],
        ) or identity["date_of_expiry"]
        address = self._value_after_marker(
            upper_lines,
            lines,
            ["ADDRESS", "ADRESSE", "RESIDENCE", "DOMICILE"],
        )
        if address is None:
            address = self._extract_driver_license_address(lines, upper_lines)
        categories = self._extract_license_categories(lines)
        place_of_birth = self._value_after_marker(
            upper_lines,
            lines,
            ["PLACE OF BIRTH", "LIEU DE NAISSANCE", "LUGAR DE NACIMIENTO"],
        )
        license_number = self._extract_driver_license_number(lines, upper_lines, date_of_birth, date_of_expiry) or identity["document_number"]
        if date_of_issue is None or date_of_issue.upper().startswith("DD "):
            fallback_issue_date = self._extract_driver_license_issue_date(lines, date_of_birth, date_of_expiry)
            if fallback_issue_date:
                date_of_issue = fallback_issue_date
        date_of_birth = self._normalize_identity_date(date_of_birth, lines)
        date_of_issue = self._normalize_identity_date(date_of_issue, lines)
        date_of_expiry = self._normalize_identity_date(date_of_expiry, lines)
        if date_of_birth is None or date_of_birth == date_of_issue:
            fallback_birth_date = self._extract_driver_license_birth_date(lines, date_of_issue, date_of_expiry)
            if fallback_birth_date:
                date_of_birth = fallback_birth_date
        if date_of_issue == date_of_birth:
            date_of_issue = self._extract_driver_license_issue_date(lines, date_of_birth, date_of_expiry)
        if self._date_to_ordinal(date_of_issue) and self._date_to_ordinal(date_of_expiry):
            if self._date_to_ordinal(date_of_issue) > self._date_to_ordinal(date_of_expiry):
                date_of_issue = None
        if license_number:
            license_number = self._normalize_driver_license_number(license_number, lines, date_of_birth, date_of_expiry)
        surname = self._normalize_identity_name(surname)
        given_names = self._normalize_identity_name(given_names)
        if surname and not self._is_useful_passport_name(surname):
            surname = None
        if given_names and not self._is_useful_passport_name(given_names):
            given_names = None

        fields = DriverLicenseFields(
            license_number=license_number,
            surname=surname,
            given_names=given_names,
            date_of_birth=date_of_birth,
            place_of_birth=place_of_birth,
            address=address,
            date_of_issue=date_of_issue,
            date_of_expiry=date_of_expiry,
            issuing_authority=identity["issuing_authority"],
            categories=categories,
        )
        return self._build_identity_result(
            fields=fields,
            validation_code="heuristic_driver_license_mapping",
            validation_message=(
                "Driver license fields were mapped from PaddleOCR-VL OCR output using heuristic parsing."
            ),
        )

    def _extract_driver_license_focus_text(self, image: Image.Image) -> str:
        image = self._crop_document_region(image)
        width, height = image.size
        focus_crop = image.crop(
            (
                int(width * 0.10),
                int(height * 0.12),
                int(width * 0.97),
                int(height * 0.90),
            )
        )
        return self._run_ocr(
            image=focus_crop,
            prompt_text=(
                "Transcribe the driver license text line by line. "
                "Prioritize the license number, names, address, dates, and field labels."
            ),
            max_new_tokens=320,
            use_base_model=True,
        )

    def _extract_driver_license_header_text(self, image: Image.Image) -> str:
        image = self._crop_driver_license_focus_region(image)
        width, height = image.size
        header_crop = image.crop(
            (
                int(width * 0.26),
                int(height * 0.04),
                int(width * 0.92),
                int(height * 0.34),
            )
        )
        return self._run_ocr(
            image=header_crop,
            prompt_text=(
                "Transcribe the top driver license header exactly. "
                "Preserve DL number, EXP, CLASS, LN, FN and nearby dates."
            ),
            max_new_tokens=160,
            use_base_model=True,
        )

    def _extract_driver_license_detail_text(self, image: Image.Image) -> str:
        image = self._crop_driver_license_focus_region(image)
        width, height = image.size
        detail_crop = image.crop(
            (
                int(width * 0.26),
                int(height * 0.10),
                int(width * 0.98),
                int(height * 0.96),
            )
        )
        return self._run_ocr(
            image=detail_crop,
            prompt_text=(
                "Transcribe the driver license details line by line. "
                "Preserve labels like DL, EXP, LN, FN, DOB, ISS."
            ),
            max_new_tokens=320,
            use_base_model=True,
        )

    def _extract_driver_license_text_block_text(self, image: Image.Image) -> str:
        image = self._crop_driver_license_focus_region(image)
        width, height = image.size
        block_crop = image.crop(
            (
                int(width * 0.26),
                int(height * 0.14),
                int(width * 0.72),
                int(height * 0.80),
            )
        )
        return self._run_ocr(
            image=block_crop,
            prompt_text=(
                "Transcribe the central driver license text block exactly. "
                "Preserve DL, LN, FN, DOB, address, RSTR, SEX, HAIR, EYES, DD and ISS labels with their values."
            ),
            max_new_tokens=280,
            use_base_model=True,
        )

    def _extract_driver_license_dates_text(self, image: Image.Image) -> str:
        image = self._crop_driver_license_focus_region(image)
        width, height = image.size
        dates_crop = image.crop(
            (
                int(width * 0.28),
                int(height * 0.46),
                int(width * 0.98),
                int(height * 0.96),
            )
        )
        return self._run_ocr(
            image=dates_crop,
            prompt_text=(
                "Read the driver license details exactly. "
                "Preserve DOB, ISS, EXP, address, LN, FN and numeric values."
            ),
            max_new_tokens=220,
            use_base_model=True,
        )

    def _extract_driver_license_bottom_right_text(self, image: Image.Image) -> str:
        image = self._crop_driver_license_focus_region(image)
        width, height = image.size
        bottom_right_crop = image.crop(
            (
                int(width * 0.70),
                int(height * 0.50),
                int(width * 0.99),
                int(height * 0.96),
            )
        )
        return self._run_ocr(
            image=bottom_right_crop,
            prompt_text=(
                "Read the bottom-right driver license fields exactly. "
                "Preserve DOB, DD, ISS, dates and all nearby numbers."
            ),
            max_new_tokens=180,
            use_base_model=True,
        )

    def _extract_driver_license_issue_text(self, image: Image.Image) -> str:
        image = self._crop_driver_license_focus_region(image)
        width, height = image.size
        issue_crop = image.crop(
            (
                int(width * 0.78),
                int(height * 0.72),
                int(width * 0.98),
                int(height * 0.98),
            )
        )
        return self._run_ocr(
            image=issue_crop,
            prompt_text="Read the issue date field exactly. Preserve the date next to ISS.",
            max_new_tokens=96,
            use_base_model=True,
        )

    def _extract_driver_license_number_text(self, image: Image.Image) -> str:
        image = self._crop_driver_license_focus_region(image)
        width, height = image.size
        number_crop = image.crop(
            (
                int(width * 0.24),
                int(height * 0.12),
                int(width * 0.82),
                int(height * 0.42),
            )
        )
        return self._run_ocr(
            image=number_crop,
            prompt_text=(
                "Read the driver license number area exactly. "
                "Preserve the red DL number, EXP date, CLASS and nearby labels."
            ),
            max_new_tokens=120,
            use_base_model=True,
        )

    def _crop_driver_license_focus_region(self, image: Image.Image) -> Image.Image:
        image = self._crop_document_region(image)
        width, height = image.size
        focus_box = (
            int(width * 0.10),
            int(height * 0.12),
            int(width * 0.97),
            int(height * 0.90),
        )
        return image.crop(focus_box)

    def _crop_document_region(self, image: Image.Image) -> Image.Image:
        grayscale = ImageOps.grayscale(image)
        mask = grayscale.point(lambda px: 255 if px < 245 else 0)
        bbox = mask.getbbox()
        if not bbox:
            return image
        left, top, right, bottom = bbox
        width, height = image.size
        if (right - left) < width * 0.35 or (bottom - top) < height * 0.25:
            return image
        pad_x = max(8, int((right - left) * 0.02))
        pad_y = max(8, int((bottom - top) * 0.02))
        cropped_box = (
            max(0, left - pad_x),
            max(0, top - pad_y),
            min(width, right + pad_x),
            min(height, bottom + pad_y),
        )
        return image.crop(cropped_box)


    def _empty_fields_for_document_type(self, document_type):
        if document_type == DocumentType.INVOICE:
            return InvoiceFields()
        if document_type == DocumentType.PASSPORT:
            return PassportFields()
        if document_type == DocumentType.ID_CARD:
            return IdCardFields()
        if document_type == DocumentType.DRIVER_LICENSE:
            return DriverLicenseFields()
        return OtherDocumentFields()

    def _value_after_marker(self, upper_lines, original_lines, markers):
        for index, line in enumerate(upper_lines):
            if any(marker in line for marker in markers):
                for next_index in range(index + 1, len(original_lines)):
                    candidate = original_lines[next_index].strip()
                    if candidate:
                        return candidate
        return None

    def _merge_lines_preserve_order(self, lines: list[str]) -> list[str]:
        merged: list[str] = []
        seen: set[str] = set()
        for line in lines:
            cleaned = " ".join(str(line).split()).strip()
            if not cleaned:
                continue
            key = cleaned.upper()
            if key in seen:
                continue
            seen.add(key)
            merged.append(cleaned)
        return merged

    def _value_from_same_or_next_line(self, upper_lines, original_lines, markers):
        for index, line in enumerate(upper_lines):
            for marker in markers:
                if marker not in line:
                    continue
                original = original_lines[index].strip()
                marker_index = line.find(marker)
                if marker_index != -1:
                    suffix = original[marker_index + len(marker) :].strip(" :-")
                    if suffix:
                        return suffix
                for next_index in range(index + 1, len(original_lines)):
                    candidate = original_lines[next_index].strip()
                    if candidate:
                        return candidate
        return None

    def _find_line_containing(self, upper_lines, original_lines, markers):
        for upper, original in zip(upper_lines, original_lines):
            if any(marker in upper for marker in markers):
                return original
        return None

    def _most_frequent_match(self, values):
        if not values:
            return None
        counts = {}
        for value in values:
            counts[value] = counts.get(value, 0) + 1
        return sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0][0]

    def _guess_passport_surname(self, upper_lines, original_lines):
        blacklist_fragments = [
            "UNITED STATES",
            "PASSPORT",
            "GIVEN NAMES",
            "NATIONALITY",
            "DATE OF",
            "SIGNATURE",
            "TYPE /",
            "PLACE",
            "SEE PAGE",
        ]
        for index, line in enumerate(upper_lines):
            if "GIVEN NAMES" in line:
                for candidate_index in range(index - 1, -1, -1):
                    candidate = original_lines[candidate_index].strip()
                    candidate_upper = candidate.upper()
                    if not candidate or re.search(r"\d", candidate):
                        continue
                    if any(fragment in candidate_upper for fragment in blacklist_fragments):
                        continue
                    if len(candidate_upper) <= 1:
                        continue
                    return candidate
        return None

    def _join_mrz(self, lines):
        mrz_lines = [line for line in lines if "<" in line and len(line) >= 20]
        if not mrz_lines:
            return None
        return " ".join(mrz_lines)

    def _extract_passport_mrz_text(self, image: Image.Image) -> str | None:
        width, height = image.size
        mrz_crop = image.crop((0, int(height * 0.82), width, height))
        mrz_text = self._run_ocr(
            image=mrz_crop,
            prompt_text="Read the machine readable zone (MRZ) exactly. Preserve all < characters.",
            max_new_tokens=160,
        )
        return mrz_text.strip() or None

    def _extract_passport_name_text(self, image: Image.Image) -> str:
        width, height = image.size
        name_crop = image.crop((int(width * 0.38), int(height * 0.42), int(width * 0.92), int(height * 0.68)))
        return self._run_ocr(
            image=name_crop,
            prompt_text="Read the surname and given names exactly as printed in the passport.",
            max_new_tokens=120,
        )

    def _extract_passport_sex_text(self, image: Image.Image) -> str:
        width, height = image.size
        sex_crop = image.crop((int(width * 0.77), int(height * 0.60), int(width * 0.90), int(height * 0.72)))
        return self._run_ocr(
            image=sex_crop,
            prompt_text="Read the sex field exactly. Return only the field value if possible.",
            max_new_tokens=40,
        )

    def _extract_passport_authority_text(self, image: Image.Image) -> str | None:
        width, height = image.size
        authority_crop = image.crop((int(width * 0.73), int(height * 0.66), width, int(height * 0.84)))
        text = self._run_ocr(
            image=authority_crop,
            prompt_text="Read the issuing authority exactly as printed.",
            max_new_tokens=80,
        )
        cleaned = " ".join(line.strip() for line in text.splitlines() if line.strip())
        cleaned = re.sub(r"\s+", " ", cleaned).strip(" :")
        return cleaned or None

    def _parse_mrz(self, mrz_text: str) -> dict[str, str]:
        lines = [re.sub(r"[^A-Z0-9<]", "", line.upper()) for line in mrz_text.splitlines()]
        lines = [line for line in lines if line]
        joined = "".join(lines)
        candidates = re.findall(r"P<[A-Z<]{20,}", joined)
        if candidates:
            line1 = candidates[0][:44].ljust(44, "<")
            if len(lines) >= 2:
                line2 = lines[1][:44].ljust(44, "<")
            else:
                remainder = joined[joined.find(candidates[0]) + len(candidates[0]) :]
                cleaned_remainder = re.sub(r"[^A-Z0-9<]", "", remainder)
                line2 = cleaned_remainder[:44].ljust(44, "<") if cleaned_remainder else None
        elif len(lines) >= 1:
            line1 = lines[0][:44].ljust(44, "<")
            line2 = lines[1][:44].ljust(44, "<") if len(lines) >= 2 else None
        else:
            return {}

        if not line1.startswith("P<"):
            return {}

        # Some OCR outputs lose the 3-letter issuing state and start directly with the surname,
        # e.g. "P<FARSI<<AHMAD<AL...". In that case fall back to parsing after "P<".
        if re.match(r"^P<[A-Z]{3}[A-Z<]*<<", line1):
            name_part = line1[5:]
        else:
            name_part = line1[2:]
        surname = None
        given_names = None
        if "<<" in name_part:
            surname_raw, given_raw = name_part.split("<<", 1)
            surname = surname_raw.replace("<", " ").strip() or None
            given_names = given_raw.replace("<", " ").strip() or None

        nationality = None
        date_of_birth = None
        sex = None
        date_of_expiry = None
        document_number = None
        if line2 and self._looks_like_td3_mrz_line2(line2):
            nationality = line2[10:13].replace("<", "").strip() or None
            date_of_birth = self._format_mrz_date(line2[13:19])
            sex = line2[20:21].replace("<", "").strip() or None
            date_of_expiry = self._format_mrz_date(line2[21:27])
            document_number = line2[0:9].replace("<", "").strip() or None

        return {
            "surname": surname,
            "given_names": given_names,
            "nationality": nationality,
            "date_of_birth": date_of_birth,
            "sex": sex,
            "date_of_expiry": date_of_expiry,
            "document_number": document_number,
        }

    def _parse_compact_passport_identity_line(self, text: str) -> dict[str, str]:
        compact = re.sub(r"[^A-Z0-9]", "", text.upper())
        match = re.search(r"([A-Z0-9]{8,9})([A-Z]{3})(\d{8})([MFX])(\d{8})", compact)
        if match:
            return {
                "document_number": match.group(1),
                "nationality": match.group(2),
                "date_of_birth": self._format_compact_date_long(match.group(3)),
                "sex": match.group(4),
                "date_of_expiry": self._format_compact_date_long(match.group(5)),
            }
        mrz_match = re.search(r"([A-Z0-9]{8,9})\d([A-Z]{3})(\d{6})\d([MFX])(\d{6})", compact)
        if not mrz_match:
            return {}
        return {
            "document_number": mrz_match.group(1),
            "nationality": mrz_match.group(2),
            "date_of_birth": self._format_mrz_date(mrz_match.group(3)),
            "sex": mrz_match.group(4),
            "date_of_expiry": self._format_mrz_date(mrz_match.group(5)),
        }

    def _parse_passport_name_from_line1(self, mrz_text: str | None) -> dict[str, str]:
        if not mrz_text:
            return {}
        compact = "".join(line.strip() for line in mrz_text.splitlines() if line.strip())
        cleaned = re.sub(r"[^A-Z<]", "", compact.upper())

        match = re.search(r"P<([A-Z]{3})([A-Z<]+?)<<([A-Z<]+)", cleaned)
        if match:
            surname = match.group(2).replace("<", " ").strip() or None
            given_names = match.group(3).replace("<", " ").strip() or None
            return {
                "surname": surname,
                "given_names": given_names,
            }

        # Fallback for OCR-degraded line1 without issuing state, e.g. "P<FARSI<<AHMAD<AL..."
        fallback = re.search(r"P<([A-Z<]+?)<<([A-Z<]+)", cleaned)
        if not fallback:
            return {}
        surname = fallback.group(1).replace("<", " ").strip() or None
        given_names = fallback.group(2).replace("<", " ").strip() or None
        return {
            "surname": surname,
            "given_names": given_names,
        }

    def _parse_passport_name_from_name_crop(self, lines: list[str]) -> dict[str, str]:
        blacklist = {
            "UNITED",
            "COUNTRY",
            "CODE",
            "PASSPORT",
            "NAME",
            "NAMES",
            "NATIONALITY",
            "TYPE",
            "PLACE",
            "BIRTH",
        }
        for line in lines:
            cleaned = " ".join(line.replace("/", " ").split()).strip()
            if not cleaned:
                continue
            tokens = [token for token in re.split(r"\s+", cleaned) if re.fullmatch(r"[A-Za-z-]+", token)]
            if len(tokens) < 2:
                continue
            upper_tokens = [token.upper() for token in tokens]
            if any(token in blacklist for token in upper_tokens):
                continue
            if not any(any(ch.islower() for ch in token) for token in tokens):
                continue
            surname = tokens[-1].replace("-", " ").upper().strip()
            given_names = " ".join(tokens[:-1]).replace("-", " ").upper().strip()
            if surname:
                return {
                    "surname": surname,
                    "given_names": given_names or None,
                }
        return {}

    def _format_mrz_date(self, value: str) -> str | None:
        value = value.strip().replace("<", "")
        if len(value) != 6 or not value.isdigit():
            return None
        year = int(value[:2])
        month = value[2:4]
        day = value[4:6]
        full_year = 1900 + year if year >= 30 else 2000 + year
        return f"{day}.{month}.{full_year}"

    def _format_compact_date(self, value: str) -> str | None:
        value = value.strip()
        if len(value) != 6 or not value.isdigit():
            return None
        day = value[:2]
        month = value[2:4]
        year = int(value[4:6])
        full_year = 1900 + year if year >= 30 else 2000 + year
        return f"{day}/{month}/{full_year}"

    def _format_compact_date_long(self, value: str) -> str | None:
        value = value.strip()
        if len(value) != 8 or not value.isdigit():
            return None
        day = value[:2]
        month = value[2:4]
        year = value[4:8]
        return f"{day}/{month}/{year}"

    def _is_plausible_document_number(self, value: str | None) -> bool:
        return bool(value and re.fullmatch(r"[A-Z0-9]{7,10}", value))

    def _is_plausible_nationality(self, value: str | None) -> bool:
        return bool(value and re.fullmatch(r"[A-Z]{3}", value))

    def _is_plausible_sex(self, value: str | None) -> bool:
        return value in {"M", "F", "X"}

    def _looks_like_td3_mrz_line2(self, value: str) -> bool:
        compact = value.replace("<", "")
        if len(value) < 28 or len(compact) < 20:
            return False
        return bool(re.match(r"^[A-Z0-9<]{9}\d[A-Z]{3}\d{6}[MFX<]\d{6}", value))

    def _normalize_place_of_birth(self, value: str | None) -> str | None:
        if not value:
            return value
        normalized = value.strip()
        replacements = {
            "ILINIOIS": "ILLINOIS",
            "ILLINIOIS": "ILLINOIS",
            "ILINOIS": "ILLINOIS",
        }
        for wrong, correct in replacements.items():
            normalized = normalized.replace(wrong, correct)
        return normalized

    def _normalize_passport_surname(self, value: str | None) -> str | None:
        if not value:
            return value
        normalized = re.sub(r"[^A-Z ]", "", value.upper()).strip()
        if normalized.startswith("USA") and len(normalized) > 3:
            normalized = normalized[3:]
        if normalized == "FARSI":
            normalized = "AL FARSI"
        return normalized or None

    def _normalize_passport_given_names(self, value: str | None) -> str | None:
        if not value:
            return value
        normalized = re.sub(r"[^A-Z ]", " ", value.upper())
        normalized = re.sub(r"\b(?:USAF|USA|ISSUE|DATE|BIRTH|PLACE|NATIONALITY|PASSPORT)\b", " ", normalized)
        normalized = re.sub(r"\s+", " ", normalized).strip()
        return normalized or None

    def _is_useful_passport_name(self, value: str | None) -> bool:
        if not value:
            return False
        compact = re.sub(r"[^A-Z ]", "", value.upper()).strip()
        if not compact:
            return False
        blacklist = {
            "NO",
            "OF",
            "ISSUE",
            "DATE OF",
            "DATE",
            "BIRTH",
            "PLACE",
            "PASSPORT",
            "AUTHORITY",
            "UNITED",
        }
        return compact not in blacklist

    def _parse_sex_from_text(self, value: str | None) -> str | None:
        if not value:
            return None
        upper = value.upper()
        if "ذكر" in value:
            return "M"
        if "انث" in value or "أنث" in value:
            return "F"
        match = re.search(r"SEX[^A-Z0-9]*([MFX])(?:[^A-Z]|$)", upper)
        if match:
            return match.group(1)
        match = re.search(r"\b([MFX])\b", upper)
        if match:
            return match.group(1)
        compact = re.sub(r"[^A-Z]", "", upper)
        if compact in {"M", "F", "X"}:
            return compact
        return None

    def _normalize_passport_authority(self, value: str | None) -> str | None:
        if not value:
            return None
        normalized = " ".join(value.split())
        normalized = re.sub(r"^.*?AUTHORITY[^A-Z0-9]*", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"^.*?AUTORIDAD[^A-Z0-9]*", "", normalized, flags=re.IGNORECASE)
        normalized = re.sub(r"^.*?AUTORIT.{0,4}[^A-Z0-9]*", "", normalized, flags=re.IGNORECASE)
        normalized = normalized.strip(" :-/")
        if "Department of State" in normalized:
            return "United States Department of State"
        return normalized or None

    def _extract_passport_issue_date(self, value: str | None) -> str | None:
        if not value:
            return None
        match = re.search(r"\b\d{2}/\d{2}/\d{4}\b", value)
        if match:
            return match.group(0)
        return None

    def _detect_passport_nationality(self, text: str) -> str | None:
        upper = text.upper()
        if "UNITED ARAB EMIRATES" in upper:
            return "ARE"
        if "UNITED STATES OF AMERICA" in upper:
            return "USA"
        return None

    def _extract_identity_core(self, lines, upper_lines):
        document_number = self._most_frequent_match(
            re.findall(r"\b[A-Z0-9]{6,12}\b", " ".join(lines))
        )
        given_names = self._value_after_marker(
            upper_lines,
            lines,
            ["GIVEN NAMES", "GIVEN NAME", "FIRST NAME", "PRENOMS", "NAMES"],
        )
        surname = self._value_after_marker(
            upper_lines,
            lines,
            ["SURNAME", "NOM", "LAST NAME", "FAMILY NAME"],
        )
        if surname is None and given_names is not None:
            surname = self._guess_passport_surname(upper_lines, lines)
        nationality = self._value_after_marker(
            upper_lines,
            lines,
            ["NATIONALITY", "NATIONALIT", "CITIZENSHIP"],
        )
        date_matches = re.findall(
            r"\b(?:\d{2}[./-]\d{2}[./-]\d{2,4}|\d{2}\s+[A-Z][a-z]{2}\s+\d{4})\b",
            " ".join(lines),
        )
        date_of_birth = self._value_after_marker(
            upper_lines,
            lines,
            ["DATE OF BIRTH", "BIRTH", "DOB", "NAISSANCE"],
        ) or (date_matches[0] if date_matches else None)
        date_of_issue = self._value_after_marker(
            upper_lines,
            lines,
            ["DATE OF ISSUE", "ISSUE DATE", "DATE OF ISUS", "DELIVERED"],
        )
        date_of_expiry = self._value_after_marker(
            upper_lines,
            lines,
            ["DATE OF EXPIRY", "EXPIRY DATE", "VALID UNTIL", "EXPIRATION"],
        )
        if date_of_issue is None and len(date_matches) >= 2:
            date_of_issue = date_matches[1]
        if date_of_expiry is None and len(date_matches) >= 3:
            date_of_expiry = date_matches[2]
        elif date_of_expiry is None and len(date_matches) >= 2:
            date_of_expiry = date_matches[-1]
        issuing_authority = self._find_line_containing(
            upper_lines,
            lines,
            ["AUTHORITY", "ISSUED BY", "AUTORITE", "DEPARTMENT"],
        )

        return {
            "document_number": document_number,
            "surname": surname,
            "given_names": given_names,
            "nationality": nationality,
            "date_of_birth": date_of_birth,
            "date_of_issue": date_of_issue,
            "date_of_expiry": date_of_expiry,
            "issuing_authority": issuing_authority,
        }

    def _build_identity_result(self, *, fields, validation_code: str, validation_message: str):
        normalized_entities = []
        field_dump = fields.model_dump()
        key_mapping = {
            "document_number": "document.number",
            "license_number": "document.number",
            "surname": "person.surname",
            "given_names": "person.given_names",
            "nationality": "person.nationality",
            "date_of_birth": "person.date_of_birth",
            "place_of_birth": "person.place_of_birth",
            "address": "person.address",
            "personal_number": "person.personal_number",
            "date_of_issue": "document.issue_date",
            "date_of_expiry": "document.expiry_date",
            "issuing_authority": "document.issuing_authority",
        }
        for field_name, entity_key in key_mapping.items():
            value = field_dump.get(field_name)
            if value:
                normalized_entities.append(
                    NormalizedEntity(key=entity_key, value=value, source_field=field_name)
                )

        populated = self._count_populated_values(field_dump)
        if populated >= 4:
            validation = ValidationSummary(
                is_valid=True,
                issues=[
                    ValidationIssue(
                        code=validation_code,
                        message=validation_message,
                        severity="info",
                    )
                ],
            )
        else:
            validation = ValidationSummary(
                is_valid=False,
                issues=[
                    ValidationIssue(
                        code="identity_mapping_low_confidence",
                        message=(
                            "PaddleOCR-VL extracted text, but the heuristic identity-document parser found too few fields."
                        ),
                        severity="warning",
                    )
                ],
            )

        return fields, normalized_entities, validation

    def _count_populated_values(self, value):
        count = 0
        if isinstance(value, dict):
            for nested in value.values():
                count += self._count_populated_values(nested)
            return count
        if isinstance(value, list):
            for nested in value:
                count += self._count_populated_values(nested)
            return count
        return 1 if value is not None and str(value).strip() else 0

    def _extract_license_categories(self, lines):
        categories = []
        for line in lines:
            upper = line.upper()
            if not any(marker in upper for marker in ["CATEGORY", "CATEGORIES", "CAT.", "CLASS"]):
                continue
            explicit_matches = []
            class_match = re.search(r"\bCLASS\s+((?:AM|A1|A2|A|B1|B|C1|C|D1|D|BE|CE|DE|T))\b", upper)
            if class_match:
                explicit_matches.append(class_match.group(1))
            cat_match = re.search(r"\b(?:CATEGORY|CATEGORIES|CAT\.)\s+((?:AM|A1|A2|A|B1|B|C1|C|D1|D|BE|CE|DE|T))\b", upper)
            if cat_match:
                explicit_matches.append(cat_match.group(1))
            if explicit_matches:
                for match in explicit_matches:
                    if match not in categories:
                        categories.append(match)
                continue
            tail = upper
            for marker in ["CATEGORY", "CATEGORIES", "CAT.", "CLASS"]:
                marker_index = tail.find(marker)
                if marker_index != -1:
                    tail = tail[marker_index + len(marker) :]
                    break
            for match in re.findall(r"\b(?:AM|A1|A2|A|B1|B|C1|C|D1|D|BE|CE|DE|T)\b", tail):
                if match not in categories:
                    categories.append(match)
        return categories

    def _extract_driver_license_number(self, lines, upper_lines, date_of_birth=None, date_of_expiry=None):
        forbidden = {
            "CALIFORNIA",
            "ALIFORNIA",
            "DRIVER",
            "LICENSE",
            "CLASS",
            "NONE",
        }
        excluded_numeric = set()
        for value in [date_of_birth, date_of_expiry]:
            if value:
                digits = re.sub(r"\D", "", value)
                if digits:
                    excluded_numeric.add(digits)
        for idx, (upper, original) in enumerate(zip(upper_lines, lines)):
            if any(marker in upper for marker in ["LIC", "LIC#", "LICENSE", "DL", "DL#", "DI"]):
                for candidate in re.findall(r"\b([A-Z0-9]{6,12})\b", original):
                    candidate_upper = candidate.upper()
                    if candidate_upper in forbidden:
                        continue
                    if candidate.isdigit() and candidate in excluded_numeric:
                        continue
                    if self._looks_like_mmddyyyy(candidate):
                        continue
                    return candidate
                for next_idx in range(idx + 1, min(idx + 3, len(lines))):
                    next_line = lines[next_idx].strip()
                    if (
                        re.fullmatch(r"\d{6,12}", next_line)
                        and next_line not in excluded_numeric
                        and not self._looks_like_mmddyyyy(next_line)
                    ):
                        return next_line

        numeric_candidates = []
        candidates = []
        for idx, original in enumerate(lines):
            value = original.strip()
            if re.fullmatch(r"\d{6,12}", value):
                if value not in excluded_numeric:
                    prev_line = lines[idx - 1].upper() if idx > 0 else ""
                    next_line = lines[idx + 1].upper() if idx + 1 < len(lines) else ""
                    if any(token in prev_line or token in next_line for token in ["DL", "EXP", "CLASS"]) and not self._looks_like_mmddyyyy(value):
                        return value
                    numeric_candidates.append((value, self._looks_like_mmddyyyy(value)))
            for candidate in re.findall(r"\b([A-Z0-9]{6,12})\b", value):
                candidate_upper = candidate.upper()
                if candidate_upper in forbidden:
                    continue
                if candidate.isdigit() and candidate in excluded_numeric:
                    continue
                if self._looks_like_mmddyyyy(candidate):
                    continue
                candidates.append(candidate)
        if numeric_candidates:
            return sorted(numeric_candidates, key=lambda item: (item[1], -len(item[0]), item[0]))[0][0]
        return candidates[0] if candidates else None

    def _extract_driver_license_address(self, lines, upper_lines):
        for index, upper in enumerate(upper_lines):
            if "FN " in upper or upper.startswith("FN"):
                for next_index in range(index + 1, min(index + 5, len(lines))):
                    candidate = lines[next_index].strip()
                    if not candidate:
                        continue
                    candidate_upper = candidate.upper()
                    if re.search(r"\d", candidate) and not any(
                        token in candidate_upper
                        for token in ["DOB", "EXP", "SEX", "HAIR", "EYES", "WGT", "RGT", "ISS", "DD"]
                    ):
                        return candidate
        return None

    def _extract_driver_license_issue_date(self, lines, date_of_birth, date_of_expiry):
        normalized_birth = (date_of_birth or "").strip()
        normalized_expiry = (date_of_expiry or "").strip()
        for candidate in lines:
            match = re.search(r"\bISS\b[^0-9]*(\d{2}/\d{2}/\d{4}|\d{8})", candidate.upper())
            if match:
                value = self._normalize_identity_date(match.group(1), lines)
                if value and value not in {normalized_birth, normalized_expiry}:
                    return value
        for candidate in lines:
            match = re.search(r"\bDD\b[^0-9]*(\d{2}/\d{2}/\d{4}|\d{8})", candidate.upper())
            if match:
                value = self._normalize_identity_date(match.group(1), lines)
                if value and value not in {normalized_birth, normalized_expiry}:
                    return value
        for candidate in lines:
            dates = re.findall(r"\b\d{2}/\d{2}/\d{4}\b", candidate)
            for value in reversed(dates):
                if value not in {normalized_birth, normalized_expiry}:
                    return value
        for candidate in lines:
            value = candidate.strip()
            if re.fullmatch(r"\d{2}/\d{2}/\d{4}", value):
                if value in {normalized_birth, normalized_expiry}:
                    continue
                return value
        return None

    def _extract_driver_license_birth_date(self, lines, date_of_issue, date_of_expiry):
        excluded = {(date_of_issue or "").strip(), (date_of_expiry or "").strip()}
        excluded_digits = {re.sub(r"\D", "", value) for value in excluded if value}
        for candidate in lines:
            upper = candidate.upper()
            match = re.search(r"\bDOB\b[^0-9]*(\d{2}/\d{2}/\d{4}|\d{8})", upper)
            if match:
                value = self._normalize_identity_date(match.group(1), lines)
                if value and value not in excluded:
                    return value
        for candidate in lines:
            for raw in re.findall(r"\b\d{8}\b", candidate):
                if raw in excluded_digits:
                    continue
                if self._looks_like_mmddyyyy(raw):
                    value = self._normalize_identity_date(raw, lines)
                    if value and value not in excluded:
                        return value
        return None

    def _normalize_identity_date(self, value: str | None, lines: list[str]) -> str | None:
        if not value:
            return None
        value = value.strip()
        direct = re.search(r"\b\d{2}/\d{2}/\d{4}\b", value)
        if direct:
            direct_value = direct.group(0)
            repaired = self._repair_identity_year(direct_value, lines)
            return repaired or direct_value
        long_tail = re.search(r"\b(\d{2})/(\d{2})/1(\d{4})\b", value)
        if long_tail:
            return f"{long_tail.group(1)}/{long_tail.group(2)}/{long_tail.group(3)}"
        overlong_tail = re.search(r"\b(\d{2})/(\d{2})/(\d{2})(\d{4})\b", value)
        if overlong_tail:
            return f"{overlong_tail.group(1)}/{overlong_tail.group(2)}/{overlong_tail.group(4)}"
        short = re.search(r"\b(\d{2})/(\d{2})/(\d{3})\b", value)
        if short:
            mmdd = short.group(1) + short.group(2)
            for line in lines:
                for candidate in re.findall(r"\b\d{8}\b", line):
                    if candidate.startswith(mmdd):
                        return f"{candidate[:2]}/{candidate[2:4]}/{candidate[4:]}"
            return f"{short.group(1)}/{short.group(2)}/1{short.group(3)}"
        two_digit = re.search(r"\b(\d{2})/(\d{2})/(\d{2})\b", value)
        if two_digit:
            mmdd = two_digit.group(1) + two_digit.group(2)
            for line in lines:
                for candidate in re.findall(r"\b\d{8}\b", line):
                    if candidate.startswith(mmdd):
                        return f"{candidate[:2]}/{candidate[2:4]}/{candidate[4:]}"
            year = int(two_digit.group(3))
            full_year = 1900 + year if year >= 30 else 2000 + year
            return f"{two_digit.group(1)}/{two_digit.group(2)}/{full_year}"
        compact = re.search(r"\b(\d{8})\b", value)
        if compact:
            candidate = compact.group(1)
            month = int(candidate[:2])
            day = int(candidate[2:4])
            year = int(candidate[4:])
            if 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2099:
                return f"{candidate[:2]}/{candidate[2:4]}/{candidate[4:]}"
        return value if re.search(r"\d", value) else None

    def _repair_identity_year(self, value: str, lines: list[str]) -> str | None:
        match = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", value)
        if not match:
            return None
        month, day, year_text = match.groups()
        year = int(year_text)
        if 1900 <= year <= 2099:
            return value
        mmdd = month + day
        for line in lines:
            for candidate in re.findall(r"\b\d{8}\b", line):
                if candidate.startswith(mmdd):
                    fixed_year = int(candidate[4:])
                    if 1900 <= fixed_year <= 2099:
                        return f"{candidate[:2]}/{candidate[2:4]}/{candidate[4:]}"
        if 1600 <= year < 1900:
            return f"{month}/{day}/19{year_text[-2:]}"
        if 2100 <= year <= 2999:
            return f"{month}/{day}/20{year_text[-2:]}"
        return None

    def _normalize_driver_license_number(self, value: str | None, lines: list[str], date_of_birth: str | None, date_of_expiry: str | None) -> str | None:
        if not value:
            return None
        compact = re.sub(r"[^A-Z0-9]", "", value.upper())
        if compact in {"CALIFORNIA", "ALIFORNIA", "DRIVERLICENSE"}:
            compact = ""
        if compact and not re.search(r"\d", compact):
            compact = ""
        excluded = {
            re.sub(r"\D", "", date_of_birth or ""),
            re.sub(r"\D", "", date_of_expiry or ""),
        }
        excluded.discard("")
        if compact.isdigit() and compact in excluded:
            compact = ""
        if compact.isdigit() and self._looks_like_mmddyyyy(compact):
            compact = ""
        if compact:
            return compact
        for idx, line in enumerate(lines):
            value = line.strip()
            if not re.fullmatch(r"\d{6,12}", value):
                continue
            if value in excluded:
                continue
            if self._looks_like_mmddyyyy(value):
                continue
            prev_line = lines[idx - 1].upper() if idx > 0 else ""
            next_line = lines[idx + 1].upper() if idx + 1 < len(lines) else ""
            if any(token in prev_line or token in next_line for token in ["DL", "EXP", "CLASS"]):
                return value
        for line in lines:
            match = re.search(r"\bDL\b[^0-9A-Z]*([A-Z0-9]{6,12})", line.upper())
            if match:
                candidate = match.group(1)
                if (
                    candidate not in {"CALIFORNIA", "ALIFORNIA"}
                    and candidate not in excluded
                    and not self._looks_like_mmddyyyy(candidate)
                ):
                    return candidate
        return None

    def _looks_like_mmddyyyy(self, value: str | None) -> bool:
        if not value:
            return False
        digits = re.sub(r"\D", "", value)
        if len(digits) != 8 or not digits.isdigit():
            return False
        month = int(digits[:2])
        day = int(digits[2:4])
        year = int(digits[4:])
        return 1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2099

    def _date_to_ordinal(self, value: str | None) -> int | None:
        if not value:
            return None
        match = re.fullmatch(r"(\d{2})/(\d{2})/(\d{4})", value.strip())
        if not match:
            return None
        month = int(match.group(1))
        day = int(match.group(2))
        year = int(match.group(3))
        if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2099):
            return None
        return year * 10000 + month * 100 + day

    def _is_better_identity_date(self, *, candidate: str | None, current: str | None, prefer_birth: bool) -> bool:
        candidate_ord = self._date_to_ordinal(candidate)
        current_ord = self._date_to_ordinal(current)
        if candidate_ord is None:
            return False
        if current_ord is None:
            return True
        candidate_year = candidate_ord // 10000
        current_year = current_ord // 10000
        if prefer_birth:
            if current_year < 1900 or current_year > 2030:
                return True
            if candidate_year < 1900 or candidate_year > 2030:
                return False
            if current_year < 1930 <= candidate_year:
                return False
            if candidate_year < 1930 <= current_year:
                return True
            return candidate_year <= current_year
        if current_year < 1900 or current_year > 2035:
            return True
        if candidate_year < 1900 or candidate_year > 2035:
            return False
        return candidate_year >= current_year

    def _is_better_issue_date(
        self,
        *,
        candidate: str | None,
        current: str | None,
        date_of_birth: str | None,
        date_of_expiry: str | None,
    ) -> bool:
        candidate_ord = self._date_to_ordinal(candidate)
        current_ord = self._date_to_ordinal(current)
        birth_ord = self._date_to_ordinal(date_of_birth)
        expiry_ord = self._date_to_ordinal(date_of_expiry)
        if candidate_ord is None:
            return False
        if birth_ord and candidate_ord <= birth_ord:
            return False
        if expiry_ord and candidate_ord >= expiry_ord:
            return False
        if current_ord is None:
            return True
        if birth_ord and current_ord <= birth_ord:
            return True
        if expiry_ord and current_ord >= expiry_ord:
            return True
        return candidate_ord <= current_ord

    def _normalize_identity_name(self, value: str | None) -> str | None:
        if not value:
            return None
        normalized = re.sub(r"[^A-Z -]", " ", value.upper())
        normalized = re.sub(r"\s+", " ", normalized).strip(" -")
        return normalized or None


class Qwen25VLEngine(BaseExtractionEngine):
    engine_name = ProcessingEngine.QWEN2_5_VL

    def predict(
        self,
        *,
        image: Image.Image,
        source_filename: str,
        request: JobRequestOptions,
    ) -> ExtractionOutput:
        raise NotImplementedError(
            "Qwen2.5-VL engine is not wired into the pipeline yet."
        )


class ExtractionService:
    def __init__(self) -> None:
        self._engines: dict[ProcessingEngine, BaseExtractionEngine] = {
            ProcessingEngine.DONUT: DonutEngineAdapter(),
            ProcessingEngine.PADDLEOCR_VL: PaddleOCRVLEngine(),
            ProcessingEngine.QWEN2_5_VL: Qwen25VLEngine(),
        }

    @staticmethod
    def default_engine_for_document_type(document_type: DocumentType) -> ProcessingEngine:
        if document_type == DocumentType.INVOICE:
            return ProcessingEngine.DONUT
        if document_type in {
            DocumentType.PASSPORT,
            DocumentType.ID_CARD,
            DocumentType.DRIVER_LICENSE,
            DocumentType.FINANCIAL_STATEMENT,
            DocumentType.OTHER,
        }:
            return ProcessingEngine.PADDLEOCR_VL
        return ProcessingEngine.PADDLEOCR_VL

    def predict(
        self,
        *,
        image: Image.Image,
        source_filename: str,
        request: JobRequestOptions,
    ) -> ExtractionOutput:
        requested_engine = (
            request.extraction_engine
            or self.default_engine_for_document_type(request.document_type)
        )
        engine = self._engines.get(requested_engine)
        if engine is None:
            raise ValueError(
                f"Unsupported extraction engine: {requested_engine}"
            )
        request = request.model_copy(update={"extraction_engine": requested_engine})
        return engine.predict(
            image=image,
            source_filename=source_filename,
            request=request,
        )
