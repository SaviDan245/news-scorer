import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import pipeline


def parse_args() -> argparse.Namespace:
    model_dir = Path(__file__).resolve().parent
    project_dir = model_dir.parent.parent
    default_data_dir = project_dir / "data" / "processed_pb"
    default_artifact_dir = project_dir / "models" / "artifacts" / "finbert_phrasebank"

    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned FinBERT checkpoint on test.csv.")
    parser.add_argument("--test-path", type=Path, default=default_data_dir / "test.csv")
    parser.add_argument("--label-mapping-path", type=Path, default=default_data_dir / "label_mapping.json")
    parser.add_argument("--artifact-dir", type=Path, default=default_artifact_dir)
    parser.add_argument("--output-path", type=Path, default=default_artifact_dir / "test_metrics.json")
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def load_label_mapping(path: Path) -> tuple[dict[str, int], dict[int, str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    label2id = {str(label): int(index) for label, index in payload["label2id"].items()}
    id2label = {int(index): str(label) for index, label in payload["id2label"].items()}
    ordered_labels = [id2label[index] for index in sorted(id2label)]
    return label2id, id2label, ordered_labels


def load_test_split(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["text", "label", "label_id"]).copy()

    df["text"] = df["text"].astype(str).str.strip()
    df = df[df["text"] != ""].reset_index(drop=True)

    df["label_id"] = df["label_id"].astype(int)
    return df


def normalize_prediction_label(raw_label: str) -> str:
    return raw_label.strip().lower()


def evaluate(test_df: pd.DataFrame, artifact_dir: Path,
             ordered_labels: list[str], batch_size: int) -> dict[str, object]:
    clf = pipeline(task="text-classification", model=str(artifact_dir),
                   tokenizer=str(artifact_dir), truncation=True, top_k=1)

    predictions_raw = clf(test_df["text"].tolist(), batch_size=batch_size)
    predicted_labels = [normalize_prediction_label(item[0]["label"]) for item in predictions_raw]
    true_labels = test_df["label"].str.strip().str.lower().tolist()

    accuracy = accuracy_score(true_labels, predicted_labels)
    macro_f1 = f1_score(true_labels, predicted_labels, average="macro")
    report = classification_report(true_labels, predicted_labels, labels=ordered_labels,
                                   output_dict=True, zero_division=0)
    matrix = confusion_matrix(true_labels, predicted_labels, labels=ordered_labels)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "labels": ordered_labels,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "num_examples": len(test_df),
    }


def main() -> None:
    args = parse_args()
    test_path = args.test_path.resolve()
    label_mapping_path = args.label_mapping_path.resolve()
    artifact_dir = args.artifact_dir.resolve()
    output_path = args.output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _, _, ordered_labels = load_label_mapping(label_mapping_path)
    test_df = load_test_split(test_path)

    metrics = evaluate(
        test_df=test_df,
        artifact_dir=artifact_dir,
        ordered_labels=ordered_labels,
        batch_size=args.batch_size,
    )

    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved test metrics to: {output_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
