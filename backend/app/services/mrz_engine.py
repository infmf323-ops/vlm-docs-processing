"""Чтение машиночитаемой зоны паспорта дообученной моделью PaddleOCR-VL + LoRA.

Инкапсулирует загрузку модели с корректной связкой версий (проверенной в
исследовательской части): transformers 4.55, загрузка через AutoModelForCausalLM
с trust_remote_code, минимальный шим совместимости для create_causal_mask и
жадное декодирование без подавления повторов (последнее принципиально для
сохранения символов-заполнителей «<» машиночитаемой зоны).

Модель загружается лениво при первом обращении. Тяжёлые зависимости
импортируются внутри методов, поэтому модуль безопасно импортировать без них.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from threading import Lock

from app.core.config import get_settings


class PassportMrzReader:
    """Синглтон-обёртка над дообученной моделью чтения MRZ."""

    _instance: "PassportMrzReader | None" = None
    _lock = Lock()

    def __init__(self) -> None:
        self.settings = get_settings()
        self._processor = None
        self._model = None
        self._device = "cpu"
        self._loaded = False

    @classmethod
    def instance(cls) -> "PassportMrzReader":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @property
    def adapter_dir(self) -> Path | None:
        raw = self.settings.passport_mrz_adapter_dir
        if not raw:
            return None
        candidate = Path(raw)
        if str(candidate).strip() in {"", "."}:
            return None
        return candidate

    def enabled(self) -> bool:
        directory = self.adapter_dir
        return directory is not None and directory.exists()

    @property
    def device(self) -> str:
        return self._device

    def _apply_causal_mask_shim(self) -> None:
        """Согласует наименование аргумента create_causal_mask (4.55 vs код модели)."""
        try:
            import transformers.masking_utils as masking_utils

            true_fn = masking_utils.create_causal_mask

            def shim(*args, inputs_embeds=None, input_embeds=None, **kwargs):
                actual = input_embeds if input_embeds is not None else inputs_embeds
                return true_fn(*args, input_embeds=actual, **kwargs)

            for name, module in list(sys.modules.items()):
                if "PaddleOCR" in name and hasattr(module, "create_causal_mask"):
                    module.create_causal_mask = shim
        except Exception:
            pass

    def _resolve_device(self) -> str:
        import torch

        configured = self.settings.device
        if configured == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return configured

    def _load(self) -> None:
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            import torch
            from peft import PeftModel
            from transformers import AutoModelForCausalLM, AutoProcessor

            settings = self.settings
            cache_root = Path(settings.huggingface_home)
            cache_root.mkdir(parents=True, exist_ok=True)
            os.environ.setdefault("HF_HOME", str(cache_root))

            device = self._resolve_device()
            dtype = torch.float16 if device.startswith("cuda") else torch.float32

            self._processor = AutoProcessor.from_pretrained(
                settings.paddleocr_vl_model_name,
                trust_remote_code=True,
                cache_dir=str(cache_root),
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                settings.paddleocr_vl_model_name,
                trust_remote_code=True,
                torch_dtype=dtype,
                cache_dir=str(cache_root),
            )
            self._apply_causal_mask_shim()
            model = PeftModel.from_pretrained(
                base_model, str(self.adapter_dir), local_files_only=True
            )
            self._model = model.to(device).eval()
            self._device = device
            self._loaded = True

    def read(self, crop) -> str:
        """Распознаёт MRZ на изображении локализованной полосы, возвращает текст."""
        self._load()
        import torch

        prompt = self.settings.passport_mrz_prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": crop},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(text=text, images=crop, return_tensors="pt").to(self._device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.settings.passport_mrz_max_new_tokens,
                do_sample=False,
                num_beams=1,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return self._processor.decode(
            output_ids[0][prompt_len:], skip_special_tokens=True
        ).strip()
