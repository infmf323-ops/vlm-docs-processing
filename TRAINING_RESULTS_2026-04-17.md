# Результаты обучения и сравнения моделей (2026-04-17)

## Текущая базовая модель

`Bennet1996/donut-small`

Локальные артефакты:

- full fine-tuning checkpoint: `E:\thesis\outputs\Bennet1996_donut-small_ft_best`
- DoRA checkpoint: `E:\thesis\outputs\Bennet1996_donut-small_dora_best`
- итоговое сравнение на validation: `E:\thesis\validation_model_comparison.json`

## Среда и датасет

- Датасет: `katanaml-org/invoices-donut-data-v1`
- Устройство: `NVIDIA GeForce RTX 3080`
- Обучение и инференс выполнялись локально на CUDA
- Для стабильности пайплайн был переведён на локальные `.arrow`-файлы датасета, а не на загрузку через Hugging Face Hub во время запуска

После очистки битых примеров использовалось:

- train: `424`
- validation: `49`
- пропущено: `1` train и `1` validation пример с пустым `ground_truth` после сериализации

## Full Fine-Tuning

Это обычный полный fine-tuning всей модели, без PEFT-адаптеров.

Конфигурация:

- epochs: `8`
- image size: `640x480`
- max length: `256`
- gradient accumulation steps: `4`
- learning rate: `1.5e-5`
- scheduler: `cosine`
- warmup ratio: `0.08`
- weight decay: `0.01`

Лучший результат:

- best val loss: `0.7832700318219711`
- best epoch: `8`

Динамика по эпохам:

| Эпоха | Train Loss | Val Loss |
|---|---:|---:|
| 1 | 6.8848 | 3.6028 |
| 2 | 3.1864 | 2.2377 |
| 3 | 1.5537 | 1.2654 |
| 4 | 1.0692 | 0.9869 |
| 5 | 0.8178 | 0.8566 |
| 6 | 0.6814 | 0.8004 |
| 7 | 0.6246 | 0.7852 |
| 8 | 0.6102 | 0.7833 |

## DoRA

Дополнительно был реализован и запущен DoRA-вариант дообучения для decoder-части модели.

Конфигурация:

- epochs requested: `8`
- epochs completed: `6`
- early stopping: сработал
- image size: `640x480`
- max length: `256`
- gradient accumulation steps: `4`
- learning rate: `1.5e-4`
- DoRA rank: `8`
- DoRA alpha: `16`
- DoRA dropout: `0.05`

Лучший результат обучения:

- best val loss: `0.5118634661241453`
- best epoch: `4`

Динамика по эпохам:

| Эпоха | Train Loss | Val Loss |
|---|---:|---:|
| 1 | 4.7801 | 0.9936 |
| 2 | 0.7478 | 0.6251 |
| 3 | 0.4355 | 0.5368 |
| 4 | 0.2665 | 0.5119 |
| 5 | 0.1626 | 0.5290 |
| 6 | 0.0878 | 0.5190 |

Важно:

- по `val loss` DoRA выглядит лучше, чем полный fine-tuning;
- однако на текущий момент инференс DoRA ещё требует дополнительной отладки;
- из-за этого field-level метрики DoRA ниже не отражают её обучающий потенциал напрямую.

## Сравнение моделей на validation

Сравнивались три варианта:

- out of the box: `Bennet1996/donut-small`
- full fine-tuning: `E:\thesis\outputs\Bennet1996_donut-small_ft_best`
- DoRA: `E:\thesis\outputs\Bennet1996_donut-small_dora_best`

Оценка выполнена на `49` валидных документах.

### Краткий вывод

- `out of the box` модель практически не извлекает нужную invoice-структуру на этой выборке
- `full fine-tuning` даёт сильный и уже практически полезный результат по ключевым полям
- `DoRA` обучилась по loss очень хорошо, но текущая генерация пока возвращает пустые поля, поэтому её inference-часть ещё требует исправления

### Document-Level Exact Match

| Модель | Exact Match |
|---|---:|
| Out of the box | 0.0000 |
| Full fine-tuning | 0.0000 |
| DoRA | 0.0000 |

### Field-Level Exact Match Accuracy

| Поле | Out of the box | Full fine-tuning | DoRA |
|---|---:|---:|---:|
| `invoice_no` | 0.0000 | 1.0000 | 0.0000 |
| `invoice_date` | 0.0000 | 1.0000 | 0.0000 |
| `seller` | 0.0000 | 0.0000 | 0.0000 |
| `client` | 0.0000 | 0.0000 | 0.0000 |
| `seller_tax_id` | 0.0000 | 0.9592 | 0.0000 |
| `client_tax_id` | 0.0000 | 1.0000 | 0.0000 |
| `iban` | 0.0000 | 0.3878 | 0.0000 |
| `total_net_worth` | 0.0000 | 0.9184 | 0.0000 |
| `total_vat` | 0.0000 | 0.9184 | 0.0000 |
| `total_gross_worth` | 0.0000 | 0.8571 | 0.0000 |

### Интерпретация

- `full fine-tuning` уже отлично восстанавливает номер счёта и дату
- `full fine-tuning` почти идеально восстанавливает tax IDs
- `full fine-tuning` очень хорошо работает по суммам
- слабые места full fine-tuning: длинные поля `seller`, `client` и пока ещё нестабильный `IBAN`
- `DoRA` нельзя считать проигравшей по качеству обучения только из-за текущих нулевых field-level метрик, потому что сейчас проблема сосредоточена именно в inference-конфигурации

## Quick Qualitative Comparison

Ниже оставлены качественные примеры, на которых видно отличие `out of the box` и `full fine-tuning`.

- base model: часто генерирует нерелевантные или пустые поля
- full fine-tuning: восстанавливает нужную структуру, идентификаторы и суммы заметно лучше
- DoRA: qualitative-сравнение будет добавлено отдельно после исправления инференса

### Пример 1

![Документ 41](report_assets/val_41.png)

- Dataset index: `41`
- Ground truth:
  - `invoice_no = 32530472`
  - `invoice_date = 08/27/2015`
  - `total_gross_worth = $ 187,00`
- Base model:
  - сгенерировала нерелевантные поля вроде `Straße`, `Geburtsdatum`, `Gesamtbrutto`
- Full fine-tuning:
  - правильно восстановила `invoice_no`
  - правильно восстановила `invoice_date`
  - правильно или почти правильно восстановила итоговую сумму
  - всё ещё искажает длинный seller/client текст

### Пример 2

![Документ 7](report_assets/val_7.png)

- Dataset index: `7`
- Ground truth:
  - `invoice_no = 67583819`
  - `invoice_date = 06/01/2012`
  - `total_net_worth = $ 889,67`
  - `total_gross_worth = 978,64`
- Base model:
  - выдала нерелевантный текст и неверную структуру
- Full fine-tuning:
  - правильно восстановила номер и дату
  - правильно восстановила `seller_tax_id`
  - правильно восстановила `client_tax_id`
  - итоговая сумма получилась близкой к правильной

### Пример 3

![Документ 1](report_assets/val_1.png)

- Dataset index: `1`
- Ground truth:
  - `invoice_no = 16220332`
  - `invoice_date = 05/15/2017`
  - `total_gross_worth = $ 69138,73`
- Base model:
  - выдала в основном нерелевантную последовательность
- Full fine-tuning:
  - правильно восстановила номер и дату
  - правильно восстановила tax IDs
  - правильно восстановила `IBAN`
  - сумма очень близка к правильной

### Пример 4

![Документ 48](report_assets/val_48.png)

- Dataset index: `48`
- Ground truth:
  - `invoice_no = 37959814`
  - `invoice_date = 07/12/2013`
  - `total_net_worth = $6623,62`
  - `total_gross_worth = $7285,98`
- Base model:
  - практически развалилась в повторяющийся нерелевантный текст
- Full fine-tuning:
  - правильно восстановила номер и дату
  - правильно восстановила seller/client tax IDs
  - правильно восстановила `IBAN`
  - правильно восстановила gross total

### Пример 5

![Документ 17](report_assets/val_17.png)

- Dataset index: `17`
- Ground truth:
  - `invoice_no = 64281058`
  - `invoice_date = 11/08/2019`
  - `total_net_worth = $ 14 014,99`
  - `total_gross_worth = $ 15 416,49`
- Base model:
  - выдала в основном нерелевантный шаблонный текст
- Full fine-tuning:
  - правильно восстановила номер и дату
  - правильно восстановила оба tax ID
  - правильно восстановила `IBAN`
  - gross amount получился очень близким

## Практический вывод

На данный момент наиболее полезная и практически рабочая модель — это `full fine-tuning`.

Почему:

- она уже стабильно извлекает основные invoice-поля
- у неё хорошие field-level метрики на всей validation-выборке
- её инференс работает корректно и воспроизводимо

По DoRA текущий статус такой:

- обучение успешно запущено и завершено
- адаптер показывает сильный `val loss`
- но inference-часть ещё не доведена до корректной генерации

## Следующий шаг

Следующий технически важный шаг:

- починить DoRA inference до нормальной генерации

После этого можно будет:

- повторно прогнать full comparison
- добавить честное qualitative-сравнение `out of the box vs full fine-tuning vs DoRA`
- понять, действительно ли DoRA даёт преимущество над full fine-tuning не только по loss, но и по реальному качеству извлечения полей
