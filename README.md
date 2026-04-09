# News-to-Trade Relevance Scorer

Демо-сервис для анализа коротких финансовых текстов: заголовков новостей, твитов, кратких news items и фрагментов пресс-релизов.

Приложение предсказывает:
- `sentiment`: `bullish` / `bearish` / `neutral`
- `actionability`: `actionable` / `non_actionable`
- `event_type`: тип события
- `horizon`: эвристический торговый горизонт
- confidence scores
- короткое текстовое объяснение

## Как это устроено

Сервис использует три отдельно обученные модели на базе FinBERT:
- sentiment
- actionability
- event type

Для деплоя используется `ONNX Runtime`.  
Поле `horizon` сейчас вычисляется эвристически из `actionability + event_type`.

## Проведённые эксперименты

В рамках проекта были рассмотрены и обучены несколько отдельных моделей на базе FinBERT:
- **sentiment model**: классификация `bullish / bearish / neutral`
- **actionability model**: классификация `actionable / non_actionable`
- **event type model**: многоклассовая классификация типа события

Для `event_type` также тестировалась двухэтапная схема обучения:
- сначала на расширенном weak-labeled train set,
- затем дополнительная доадаптация только на `FiQA` тренировочной выборке

Для `actionability` дополнительно рассматривались:
- обычный full fine-tuning
- более мягкий fine-tuning с уменьшенным learning rate
- вариант с заморозкой encoder
- PEFT (LoRA) с разными гиперпараметрами

На практике лучшими оказались:
- отдельная sentiment-модель на Financial PhraseBank
- отдельная actionability-модель на вручную проверенном и расширенном датасете
- отдельная event type модель с дообучением только на `FiQA` после внешнего weak-labeled pretrain'а

## Работа с данными

Для обучения использовалась комбинированная data strategy:
- **Financial PhraseBank** для задачи sentiment
- **FiQA** как основное ядро для actionability и event type
- внешний набор финансовых headline'ов для weak labeling и расширения train set

Что было сделано с данными:
- нормализация текстов и схемы полей
- ручная разметка и проверка части actionability-меток
- weak labeling по keyword/rule engine
- объединение ручных, weak и dataset-based labels
- выделение отдельных train / val / test split'ов
- построение отдельного `FiQA-only` train split для второй стадии обучения event type модели

В текущем demo-runtime используются только финальные ONNX-артефакты, а не training checkpoints.

## Структура runtime-артефактов

В рантайме используются только ONNX-артефакты:
- `models/onnx_artifacts/tokenizer`
- `models/onnx_artifacts/sentiment`
- `models/onnx_artifacts/actionability`
- `models/onnx_artifacts/event_type`

## Локальный запуск

```bash
streamlit run app.py
```

## Зависимости

Основные зависимости:
- `streamlit`
- `transformers`
- `onnx`
- `onnxruntime`
- `numpy`

Полный список указан в `requirements.txt`.

## Ограничения

- Это исследовательский/демо-проект, а не инвестиционный продукт.
- Сервис лучше всего работает на коротких англоязычных финансовых текстах.
- `horizon` не является рыночной ground-truth меткой и задаётся эвристически.

## Дисклеймер

Приложение не является инвестиционной рекомендацией и не должно использоваться как готовая торговая система.
