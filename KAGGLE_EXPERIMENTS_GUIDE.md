# Инструкция по запуску экспериментов на Kaggle

Полное пошаговое руководство: от сборки бандла до интерпретации результатов.

---

## Шаг 1. Сборка бандла локально

Запусти в PowerShell (из любой папки):

```powershell
E:\thesis\.conda311\python.exe E:\thesis\backend\scripts\build_kaggle_bundle.py
```

Это создаст файл:

```
E:\thesis\artifacts\kaggle_bundle.zip
```

Бандл включает: весь код бэкенда, скрипты, JSONL-датасеты и изображения для обучения и оценки. Всё, что нужно для запуска на Kaggle.

> **Если добавил новые данные** (например, augmented split), обязательно пересобери бандл перед загрузкой.

---

## Шаг 2. Загрузка на Kaggle

1. Открой [https://www.kaggle.com/datasets](https://www.kaggle.com/datasets) → **New Dataset**
2. Загрузи `E:\thesis\artifacts\kaggle_bundle.zip`
3. Назови датасет, например: `thesis-bundle` (запомни имя — оно понадобится)
4. Видимость: Private
5. Нажми **Create**

---

## Шаг 3. Создание ноутбука на Kaggle

1. Перейди в [https://www.kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. В настройках ноутбука (справа):
   - **Accelerator**: `GPU P100` (или T4 x2 если доступен)
   - **Internet**: `On` (нужен для скачивания весов PaddleOCR-VL с Hugging Face)
3. Добавь датасет: нажми **Add Data** → найди свой `thesis-bundle`

---

## Шаг 4. Ячейка инициализации (запускать первой в каждом ноутбуке)

```python
# ── Инициализация ──────────────────────────────────────────────────────────────
import os, shutil, zipfile, pathlib, subprocess

BUNDLE = "/kaggle/input/thesis-bundle"   # ← замени на своё имя датасета если другое
DEST   = "/kaggle/working/thesis_bundle"

if not pathlib.Path(DEST).exists():
    zip_candidates = list(pathlib.Path(BUNDLE).glob("*.zip"))
    if zip_candidates:
        print(f"Распаковываем {zip_candidates[0]} ...")
        with zipfile.ZipFile(zip_candidates[0]) as zf:
            zf.extractall(DEST)
    else:
        # Датасет загружен как папка — копируем в working (input read-only)
        print(f"Копируем {BUNDLE} → {DEST} ...")
        shutil.copytree(BUNDLE, DEST)
else:
    print(f"Папка {DEST} уже существует, пропускаем копирование")

THESIS_ROOT = DEST
os.chdir(THESIS_ROOT)
print(f"THESIS_ROOT = {THESIS_ROOT}")
print(f"Рабочая папка: {os.getcwd()}")

# Установка зависимостей
subprocess.run(
    ["pip", "install", "-q", "-r", "kaggle/requirements-kaggle.txt"],
    check=True
)
print("✓ Зависимости установлены")
```

Первый запуск занимает ~3–5 минут (скачивание базовой модели ~6 GB).

---

## Эксперимент 1 (ГЛАВНЫЙ): Transfer Learning v3 — полные поля

**Что делает:** обучает модель на ~30 паспортах разных стран (без русских и без печатных) → адаптирует на 3 русских паспортах → оценивает извлечение всех полей.

**Почему это главное:** самая чистая версия transfer learning, без утечки данных.

### 1a. Pretrain — многонациональные паспорта v3

```python
# Pretrain: ~30 паспортов без русских, без печатных.
# Ожидаемое время: ~60–90 минут на P100.
!python kaggle/run_train_on_kaggle.py \
    --mode passport_pretrain_hf_v3 \
    --epochs 6

print("✓ Pretrain v3 завершён")
print(f"Адаптер сохранён в: {THESIS_ROOT}/outputs/paddleocr_vl_passport_pretrain_hf_v3_kaggle")
```

### 1b. Finetune — адаптация на русские паспорта v3

```python
# Finetune: 3 русских паспорта, LR=5e-5 (низкий, чтобы не забыть pretrain).
# Ожидаемое время: ~15–25 минут на P100.
!python kaggle/run_train_on_kaggle.py \
    --mode passport_russian_finetune_v3 \
    --epochs 6 \
    --resume-from-adapter {THESIS_ROOT}/outputs/paddleocr_vl_passport_pretrain_hf_v3_kaggle

print("✓ Finetune v3 завершён")
```

### 1c. Оценка — полные поля

```python
# Оценка на тестовой выборке русских паспортов.
import os
env = os.environ.copy()
env["PASSPORT_ADAPTER_DIR"] = f"{THESIS_ROOT}/outputs/paddleocr_vl_passport_russian_finetune_v3_kaggle"
env["PASSPORT_EVAL_OUTPUT"] = "/kaggle/working/eval_v3_flat.json"

import subprocess
result = subprocess.run(
    ["python", "kaggle/eval_passport_flat_on_kaggle.py"],
    cwd=THESIS_ROOT, env=env, capture_output=False
)
print("Результаты сохранены в: /kaggle/working/eval_v3_flat.json")
```

### Альтернатива: запустить всё одной командой

```python
# Pipeline: pretrain + finetune + eval в одной команде.
# Используй если хочешь запустить и уйти — всё сделает само.
!python kaggle/run_passport_transfer_on_kaggle.py \
    --pipeline-version v3 \
    --pretrain-epochs 6 \
    --finetune-epochs 6 \
    --eval-output /kaggle/working/eval_v3_pipeline.json
```

---

## Эксперимент 2: Transfer Learning v2 — для сравнения

**Что делает:** то же самое, но pretrain включает печатные паспорта. Сравниваем с v3 чтобы понять, помогают ли печатные образцы или мешают.

```python
# Pretrain v2 (~30 мин)
!python kaggle/run_train_on_kaggle.py \
    --mode passport_pretrain_hf_v2 \
    --epochs 6

# Finetune v2 (~20 мин)  
!python kaggle/run_train_on_kaggle.py \
    --mode passport_russian_finetune_v2 \
    --epochs 6 \
    --resume-from-adapter {THESIS_ROOT}/outputs/paddleocr_vl_passport_pretrain_hf_v2_kaggle

# Оценка v2
import os, subprocess
env = os.environ.copy()
env["PASSPORT_ADAPTER_DIR"] = f"{THESIS_ROOT}/outputs/paddleocr_vl_passport_russian_finetune_v2_kaggle"
env["PASSPORT_EVAL_OUTPUT"] = "/kaggle/working/eval_v2_flat.json"
subprocess.run(["python", "kaggle/eval_passport_flat_on_kaggle.py"],
               cwd=THESIS_ROOT, env=env)
print("Результаты: /kaggle/working/eval_v2_flat.json")
```

---

## Эксперимент 3: Augmented Russian Finetune (v3_aug)

**Что делает:** использует увеличенный датасет русских паспортов (3 оригинала → 21 образец с аугментацией), finetune поверх pretrain v3.

**Зачем:** проверяем, помогает ли аугментация при таком маленьком наборе данных.

> **Предварительно:** аугментированный файл уже создан и лежит в бандле как  
> `data/multidoc/passport_russian_finetune_train_v3_aug.jsonl` (21 строка).

```python
# Убедись, что pretrain v3 уже обучен (Эксперимент 1a)
# Затем finetune на аугментированных данных:
!python kaggle/run_train_on_kaggle.py \
    --mode passport_russian_finetune_v3_aug \
    --epochs 8 \
    --resume-from-adapter {THESIS_ROOT}/outputs/paddleocr_vl_passport_pretrain_hf_v3_kaggle

# Оценка
import os, subprocess
env = os.environ.copy()
env["PASSPORT_ADAPTER_DIR"] = f"{THESIS_ROOT}/outputs/paddleocr_vl_passport_russian_finetune_v3_aug_kaggle"
env["PASSPORT_EVAL_OUTPUT"] = "/kaggle/working/eval_v3_aug_flat.json"
subprocess.run(["python", "kaggle/eval_passport_flat_on_kaggle.py"],
               cwd=THESIS_ROOT, env=env)
print("Результаты: /kaggle/working/eval_v3_aug_flat.json")
```

---

## Эксперимент 4: MRZ-first Branch

**Что делает:** вместо всех полей модель учится выдавать только строки MRZ (машиночитаемую зону), из которых поля потом парсятся детерминированно по стандарту ICAO 9303.

**Зачем:** MRZ короче и структурированнее полного flat-формата. Гипотеза: модели проще генерировать 2 строки MRZ, чем 10+ полей.

### 4a. MRZ Pretrain v3

```python
!python kaggle/run_train_on_kaggle.py \
    --mode passport_pretrain_hf_mrz_v3 \
    --epochs 6
```

### 4b. MRZ Finetune v3

```python
!python kaggle/run_train_on_kaggle.py \
    --mode passport_russian_finetune_mrz_v3 \
    --epochs 6 \
    --resume-from-adapter {THESIS_ROOT}/outputs/paddleocr_vl_passport_pretrain_hf_mrz_v3_kaggle
```

### 4c. MRZ Оценка

```python
import os, subprocess
env = os.environ.copy()
env["PASSPORT_ADAPTER_DIR"] = f"{THESIS_ROOT}/outputs/paddleocr_vl_passport_russian_finetune_mrz_v3_kaggle"
env["PASSPORT_EVAL_OUTPUT"] = "/kaggle/working/eval_v3_mrz.json"
subprocess.run(["python", "kaggle/eval_passport_mrz_on_kaggle.py"],
               cwd=THESIS_ROOT, env=env)
print("Результаты: /kaggle/working/eval_v3_mrz.json")
```

---

## Чтение и сравнение результатов

После запуска оценок — прочитай и сравни JSON-файлы прямо в ноутбуке:

```python
import json

def show_eval(path, label):
    try:
        data = json.load(open(path))
    except FileNotFoundError:
        print(f"{label}: файл не найден")
        return
    
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    
    # Точные метрики
    exact = data.get("exact_metrics", data)
    n = exact.get("n_samples", data.get("n_samples", "?"))
    print(f"  Образцов: {n}")
    
    fields = ["document_number", "surname", "given_names", "nationality",
              "date_of_birth", "sex", "date_of_issue", "date_of_expiry",
              "issuing_authority", "mrz"]
    
    print(f"\n  {'Поле':<22} {'Exact%':>8} {'Norm%':>8}")
    print(f"  {'-'*40}")
    
    for f in fields:
        exact_acc = exact.get(f"acc_{f}", data.get(f"acc_{f}", None))
        norm_acc = data.get("normalized_metrics", {}).get(f"acc_{f}", None)
        
        e_str = f"{exact_acc*100:.1f}%" if exact_acc is not None else "—"
        n_str = f"{norm_acc*100:.1f}%" if norm_acc is not None else "—"
        print(f"  {f:<22} {e_str:>8} {n_str:>8}")
    
    # Общая точность
    overall_exact = exact.get("overall_exact_match", data.get("overall_exact_match", None))
    overall_norm = data.get("normalized_metrics", {}).get("overall_exact_match", None)
    print(f"\n  {'ИТОГО (все поля)':<22} {f'{overall_exact*100:.1f}%' if overall_exact else '—':>8} {f'{overall_norm*100:.1f}%' if overall_norm else '—':>8}")

# Показываем все результаты
show_eval("/kaggle/working/eval_v3_flat.json",    "Transfer v3 (полные поля)")
show_eval("/kaggle/working/eval_v2_flat.json",    "Transfer v2 (для сравнения)")
show_eval("/kaggle/working/eval_v3_aug_flat.json","Transfer v3 + Aug (3→21 образцов)")
show_eval("/kaggle/working/eval_v3_mrz.json",     "MRZ-first v3")
```

---

## Как сохранить результаты перед завершением сессии

Kaggle удаляет `/kaggle/working/` после завершения ноутбука. Сохрани важное:

```python
import shutil, os

# Собираем все eval JSON в одну папку
os.makedirs("/kaggle/working/results", exist_ok=True)

for fname in ["eval_v3_flat.json", "eval_v2_flat.json",
              "eval_v3_aug_flat.json", "eval_v3_mrz.json"]:
    src = f"/kaggle/working/{fname}"
    if os.path.exists(src):
        shutil.copy(src, f"/kaggle/working/results/{fname}")
        print(f"✓ {fname}")

# Выведем сводку для копирования вручную если надо
print("\nВсе результаты в /kaggle/working/results/")
print("Скачай папку через File → Download кнопку справа в файл-менеджере Kaggle.")
```

Также можно сохранить обученные адаптеры как Kaggle Dataset (через **Save Version**), чтобы переиспользовать в следующих сессиях без переобучения.

---

## Порядок запуска (рекомендуемый)

Если у тебя одна сессия (≈12 часов на GPU), запускай в таком порядке:

| # | Что | Время | Приоритет |
|---|-----|-------|-----------|
| 1 | Pretrain v3 | ~80 мин | Обязательно |
| 2 | Finetune v3 | ~20 мин | Обязательно |
| 3 | Eval v3 flat | ~5 мин | Обязательно |
| 4 | Finetune v3_aug | ~30 мин | Важно |
| 5 | Eval v3_aug flat | ~5 мин | Важно |
| 6 | MRZ pretrain v3 | ~80 мин | Желательно |
| 7 | MRZ finetune v3 | ~20 мин | Желательно |
| 8 | MRZ eval v3 | ~5 мин | Желательно |
| 9 | Pretrain v2 | ~80 мин | Если осталось время |
| 10 | Finetune v2 + eval | ~25 мин | Если осталось время |

**Итого обязательная программа:** ~3 часа. Полная программа: ~6–7 часов.

---

## Часто встречающиеся проблемы

### Out of Memory (CUDA OOM)
Уменьши размер изображения и длину последовательности:
```python
!python kaggle/run_train_on_kaggle.py \
    --mode passport_pretrain_hf_v3 \
    --epochs 6 \
    --max-image-side 640 \
    --max-length 1536 \
    --max-new-tokens 160
```

### Модель не скачивается (нет интернета)
Убедись что в настройках ноутбука включён **Internet: On**. Без этого базовая модель не скачается.

### "THESIS_ROOT not defined" в ячейках после инициализации
Переменные Python не сохраняются между ячейками если ядро перезапустилось. Запусти ячейку инициализации заново.

### Адаптер не найден при finetune
Проверь что претрейн завершился без ошибок и папка с адаптером существует:
```python
import os
adapter_path = f"{THESIS_ROOT}/outputs/paddleocr_vl_passport_pretrain_hf_v3_kaggle"
print("Существует:", os.path.exists(adapter_path))
print("Файлы:", os.listdir(adapter_path) if os.path.exists(adapter_path) else "нет папки")
```

---

## Что означают результаты

- **Exact%** — процент образцов где поле совпало дословно (с учётом регистра и пробелов)
- **Norm%** — процент после нормализации: нижний регистр, лишние пробелы, замена символов-заменителей (`<` → ` `) и т.д.
- **ИТОГО** — процент образцов где **все** поля правильные одновременно

Хороший результат для данного объёма обучающих данных (3–21 образец):
- `date_of_birth`, `sex`, `document_number` — должны быть ≥ 80% при наличии MRZ
- `surname`, `given_names` — сложнее, ожидаем 50–80%
- `issuing_authority` — самое сложное поле, часто < 50%
- `ИТОГО` — даже 20–30% это хороший знак при таком малом датасете

Если MRZ-first (Эксперимент 4) даёт лучше `document_number`, `date_of_birth`, `nationality` чем flat — это подтверждает гипотезу о преимуществе структурированного промежуточного представления.
