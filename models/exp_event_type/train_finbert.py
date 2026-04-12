import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed
)


def parse_args() -> argparse.Namespace:
    model_dir = Path(__file__).resolve().parent
    project_dir = model_dir.parent.parent
    default_data_dir = project_dir / 'data' / 'processed_event_type'
    default_output_dir = project_dir / 'models' / 'artifacts' / 'finbert_event_type'

    parser = argparse.ArgumentParser(description='Fine-tune FinBERT on event type labels.')
    parser.add_argument('--train-path', type=Path, default=default_data_dir / 'event_type_train.csv')
    parser.add_argument('--val-path', type=Path, default=default_data_dir / 'event_type_val.csv')
    parser.add_argument('--label-mapping-path', type=Path, default=default_data_dir / 'event_type_label_mapping.json')
    parser.add_argument('--model-name', default='ProsusAI/finbert')
    parser.add_argument('--output-dir', type=Path, default=default_output_dir)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--train-batch-size', type=int, default=16)
    parser.add_argument('--eval-batch-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-ratio', type=float, default=0.1)
    parser.add_argument('--logging-steps', type=int, default=20)
    parser.add_argument('--early-stopping-patience', type=int, default=2)
    parser.add_argument('--early-stopping-threshold', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)

    return parser.parse_args()


def load_label_mapping(path: Path) -> tuple[dict[str, int], dict[int, str]]:
    payload = json.loads(path.read_text(encoding='utf-8'))
    label2id = {str(label): int(index) for label, index in payload['label2id'].items()}
    id2label = {int(index): str(label) for index, label in payload['id2label'].items()}
    return label2id, id2label


def load_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)

    df = df.dropna(subset=['text', 'label_id']).copy()
    df['text'] = df['text'].astype(str).str.strip()
    df = df[df['text'] != ''].reset_index(drop=True)
    df['label_id'] = df['label_id'].astype(int)

    return df


def dataframe_to_dataset(df: pd.DataFrame) -> Dataset:
    return Dataset.from_pandas(df[['text', 'label_id']], preserve_index=False)


def tokenize_batch(batch: dict[str, list[str]],
                   tokenizer: AutoTokenizer, max_length: int) -> dict[str, list[int]]:
    return tokenizer(batch['text'], truncation=True, max_length=max_length)


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        'accuracy': float(accuracy_score(labels, predictions)),
        'macro_f1': float(f1_score(labels, predictions, average='macro'))
    }


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    train_path = args.train_path.resolve()
    val_path = args.val_path.resolve()
    label_mapping_path = args.label_mapping_path.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    label2id, id2label = load_label_mapping(label_mapping_path)
    train_df = load_split(train_path)
    val_df = load_split(val_path)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(label2id), label2id=label2id,
        id2label=id2label, ignore_mismatched_sizes=True)

    train_dataset = dataframe_to_dataset(train_df).rename_column('label_id', 'labels')
    val_dataset = dataframe_to_dataset(val_df).rename_column('label_id', 'labels')

    train_dataset = train_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer=tokenizer, max_length=args.max_length),
        batched=True, desc='Tokenizing train split')
    val_dataset = val_dataset.map(
        lambda batch: tokenize_batch(batch, tokenizer=tokenizer, max_length=args.max_length),
        batched=True, desc='Tokenizing validation split')

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        overwrite_output_dir=True,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        logging_strategy='steps',
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.epochs,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        load_best_model_at_end=True,
        metric_for_best_model='macro_f1',
        greater_is_better=True,
        save_total_limit=2,
        report_to='none',
        seed=args.seed
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset,
        tokenizer=tokenizer, data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics, callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold
            )
        ]
    )

    trainer.train()

    eval_metrics = trainer.evaluate(eval_dataset=val_dataset)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(output_dir)

    metrics_path = output_dir / 'val_metrics.json'
    metrics_path.write_text(json.dumps(eval_metrics, indent=2), encoding='utf-8')

    print(f'Saved model and tokenizer to: {output_dir}')
    print(f'Validation metrics saved to: {metrics_path}')
    print(json.dumps(eval_metrics, indent=2))


if __name__ == '__main__':
    main()
