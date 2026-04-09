from __future__ import annotations

import argparse
from pathlib import Path

from transformers import pipeline


class FinBERTInference:
    def __init__(self, artifact_dir: str | Path | None = None) -> None:
        model_dir = Path(__file__).resolve().parent
        default_artifact_dir = model_dir.parent / "artifacts" / "finbert_phrasebank"
        self.artifact_dir = Path(artifact_dir).resolve() if artifact_dir is not None else default_artifact_dir.resolve()
        self.classifier = pipeline(
            task="text-classification",
            model=str(self.artifact_dir),
            tokenizer=str(self.artifact_dir),
            truncation=True,
            top_k=None,
        )

    def predict(self, text: str) -> dict[str, object]:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Input text must not be empty.")

        outputs = self.classifier(cleaned_text)
        probs = {
            item["label"].strip().lower(): float(item["score"])
            for item in outputs
        }
        label = max(probs, key=probs.get)
        confidence = probs[label]

        return {
            "text": cleaned_text,
            "label": label,
            "confidence": confidence,
            "probs": probs,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run single-text inference with fine-tuned FinBERT.")
    parser.add_argument("--text", required=True, help="Input text to classify.")
    parser.add_argument("--artifact-dir", type=Path, default=None, help="Path to saved model artifact.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = FinBERTInference(artifact_dir=args.artifact_dir)
    result = predictor.predict(args.text)
    print(result)


if __name__ == "__main__":
    main()
