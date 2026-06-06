# Identity-Document Dataset Shortlist

Дата фиксации: 14 мая 2026

## Цель

Подобрать источники данных для следующей итерации `multi-document extraction`, где нам нужны:

- `passport`
- `id_card`
- `driver_license`
- пригодность для извлечения полей
- возможность привести данные к нашей `JSON`-схеме

## Итоговый выбор

Для проекта беру такой стек источников:

1. `DocXPand-25k` - основной supervised source для новой `LoRA`-итерации
2. `MIDV-500` - основной real-world benchmark и внешний evaluation-set
3. `ud-biometrics` passport / driver-license datasets - bootstrap-источник изображений для очереди разметки

`IDNet` пока не тяну в extraction-ветку. Его лучше держать как отдельный резерв под `fraud / tamper detection`.

## 1. DocXPand-25k

Статус: `выбран как основной train source`

Почему подходит:

- В статье заявлены `24,994` richly labeled images identity-документов.
- В набор входят `identity cards`, `residence permits` и `passports`.
- В официальном README прямо указано, что labels хранятся в `JSON`-формате.
- Структура датасета уже ориентирована на document analysis и text recognition.

Почему это важно для нас:

- Нам нужен не просто OCR, а именно `field extraction`.
- У DocXPand есть и масштаб, и явные field-level labels.
- Это наиболее естественный кандидат для `PaddleOCR-VL + LoRA`.

Ограничения:

- Лицензия датасета: `CC-BY-NC-SA 4.0`.
- Данные синтетические, поэтому real-world перенос нужно отдельно проверять.

Практическая роль в проекте:

- использовать как главный supervised source для `passport / id_card / identity-doc extraction`

Источники:

- [DocXPand-25k paper](https://arxiv.org/abs/2407.20662)
- [DocXPand GitHub release](https://github.com/QuickSign/docxpand/releases)

## 2. MIDV-500

Статус: `выбран как основной real-world benchmark`

Почему подходит:

- Это публичный dataset именно для `identity document analysis`.
- В статье заявлены `500` video clips для `50` document types.
- В наборе есть `ID cards`, `passports`, `driving licenses` и другие документы.
- Датасет часто используют как benchmark для реальных мобильных сценариев.

Почему это важно для нас:

- После обучения на synthetic-источнике нужен внешний real-world benchmark.
- MIDV-500 подходит для проверки generalization новой extraction-ветки.
- Мы уже знакомы с ним по classification-части проекта.

Ограничения:

- Это video dataset, а не готовый `image + fields JSON` train-set.
- Для extraction-пайплайна его нужно конвертировать: выбирать кадры и поднимать нужные annotations.

Практическая роль в проекте:

- использовать как внешний benchmark
- частично использовать как supplemental evaluation / annotation source

Источники:

- [MIDV-500 paper](https://arxiv.org/abs/1807.05786)
- [Smart Engines MIDV-500 overview](https://smartengines.com/wp-content/uploads/2020/04/datasets-of-id-documents-midv-500.pdf)

## 3. ud-biometrics passport / driver-license datasets

Статус: `выбран как bootstrap-source для очереди разметки`

Что именно интересно:

- `ud-biometrics/passport-dataset`
- `ud-biometrics/synthetic-usa-driver-license`
- synthetic passport datasets того же автора

Что удалось проверить:

- `passport-dataset` на Hugging Face в текущем виде содержит в основном изображения.
- synthetic passport / driver-license наборы содержат `image + label`, где `label` - это класс шаблона, а не полноценная extraction-разметка полей.

Почему это все равно полезно:

- Эти наборы легко скачать и быстро завести в локальный pipeline.
- Их удобно использовать как `bootstrap-source` для ручной разметки.
- Это хороший способ быстро расширить `passport / driver_license` очередь для annotation.

Почему это не основной train-source:

- В текущем виде там нет полноценной field-level extraction-разметки.
- Для supervised extraction они слабее DocXPand.

Практическая роль в проекте:

- быстрое расширение очереди разметки
- дополнительный identity-source для ручной annotation

Источники:

- [ud-biometrics/passport-dataset](https://huggingface.co/datasets/ud-biometrics/passport-dataset)
- [ud-biometrics/synthetic-usa-driver-license](https://huggingface.co/datasets/ud-biometrics/synthetic-usa-driver-license)

## 4. IDNet

Статус: `отложен`

Почему интересен:

- Современный identity-document dataset с акцентом на fraud detection.
- Полезен для задач подделок, tamper analysis и synthetic identity.

Почему пока не берем:

- Для текущей extraction-ветки он избыточен по цели.
- Нам сейчас важнее стабилизировать `field extraction`, а не fraud branch.

Практическая роль в проекте:

- резерв под следующий этап: `tamper / fraud detection`

Источники:

- [IDNet paper](https://arxiv.org/abs/2408.01690)
- [IDNet Zenodo](https://zenodo.org/records/10570622)

## Что реально тянем в проект сейчас

### Берем сразу

1. `DocXPand-25k` - как основной supervised source
2. `MIDV-500` - как benchmark и внешний validation source
3. `ud-biometrics` - как bootstrap-source для annotation queue

### Не берем пока

1. `IDNet` - откладываем на отдельную fraud/tamper ветку

## Практический план интеграции

### Этап 1

- подготовить `import / conversion` pipeline под `DocXPand-25k`
- привести его к нашей `multi-document JSON`-схеме

### Этап 2

- подготовить `MIDV-500` как evaluation-source
- выбрать подмножество `passport / id_card / driver_license`

### Этап 3

- подтянуть `ud-biometrics` в локальный staging
- использовать эти изображения для расширения очереди разметки

## Почему этот выбор оптимален

- `DocXPand-25k` дает масштаб и field labels
- `MIDV-500` дает real-world benchmark
- `ud-biometrics` дает быстрый приток identity-изображений
- `IDNet` не смешивает текущую extraction-задачу с отдельной fraud-веткой
