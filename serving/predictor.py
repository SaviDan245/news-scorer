from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
from transformers import AutoTokenizer


SENTIMENT_MAP = {
    "positive": "bullish",
    "negative": "bearish",
    "neutral": "neutral",
}

HORIZON_MAP = {
    "earnings_guidance": "intraday",
    "corporate_action_mna": "mft",
    "management_change": "intraday",
    "legal_regulatory": "intraday",
    "financing_restructuring": "intraday",
    "other_actionable": "mft",
}

EVENT_TYPE_DISPLAY = {
    "earnings_guidance": "earnings / guidance",
    "corporate_action_mna": "corporate action / M&A",
    "management_change": "management change",
    "legal_regulatory": "legal / regulatory",
    "financing_restructuring": "financing / restructuring",
    "other_actionable": "other actionable",
}

ALLOWED_BACKENDS = {"onnx", "torch"}


def normalize_id2label(id2label: dict[int | str, str]) -> dict[int, str]:
    return {int(index): str(label) for index, label in id2label.items()}


def format_confidence(score: float) -> str:
    return f"{score:.1%}"


class BaseBackend:
    def __init__(self, tokenizer_dir: Path, max_length: int) -> None:
        self.tokenizer_dir = tokenizer_dir
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
        self.device_name = "cpu"
        self.runtime_name = "unknown"

    def encode(self, text: str) -> Any:
        raise NotImplementedError

    def predict_head(
        self,
        task_name: str,
        encoded: Any,
        label_map: dict[str, str] | None = None,
    ) -> dict[str, object]:
        raise NotImplementedError


class TorchBackend(BaseBackend):
    def __init__(
        self,
        sentiment_dir: Path,
        actionability_dir: Path,
        event_type_dir: Path,
        tokenizer_dir: Path,
        max_length: int,
    ) -> None:
        import torch
        import torch.nn.functional as F
        from transformers import AutoModelForSequenceClassification

        super().__init__(tokenizer_dir=tokenizer_dir, max_length=max_length)

        self._torch = torch
        self._F = F
        self.runtime_name = "torch"
        self.device = self._detect_device()
        self.device_name = str(self.device)

        self.models = {
            "sentiment": self._load_model(sentiment_dir, AutoModelForSequenceClassification),
            "actionability": self._load_model(actionability_dir, AutoModelForSequenceClassification),
            "event_type": self._load_model(event_type_dir, AutoModelForSequenceClassification),
        }
        self.id2label = {
            name: normalize_id2label(model.config.id2label)
            for name, model in self.models.items()
        }

    def _detect_device(self):
        if self._torch.cuda.is_available():
            return self._torch.device("cuda")
        if self._torch.backends.mps.is_available():
            return self._torch.device("mps")
        return self._torch.device("cpu")

    def _load_model(self, artifact_dir: Path, model_cls):
        model = model_cls.from_pretrained(str(artifact_dir))
        model.to(self.device)
        model.eval()
        return model

    def encode(self, text: str) -> dict[str, Any]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def predict_head(
        self,
        task_name: str,
        encoded: dict[str, Any],
        label_map: dict[str, str] | None = None,
    ) -> dict[str, object]:
        model = self.models[task_name]
        id2label = self.id2label[task_name]

        with self._torch.no_grad():
            logits = model(**encoded).logits
            probs = self._F.softmax(logits, dim=-1).squeeze(0).detach().cpu().numpy()

        return build_prediction_payload(probs=probs, id2label=id2label, label_map=label_map)


class OnnxBackend(BaseBackend):
    def __init__(
        self,
        sentiment_dir: Path,
        actionability_dir: Path,
        event_type_dir: Path,
        tokenizer_dir: Path,
        max_length: int,
    ) -> None:
        import onnxruntime as ort

        super().__init__(tokenizer_dir=tokenizer_dir, max_length=max_length)

        self._ort = ort
        self.runtime_name = "onnx"
        self.device_name = "cpu"
        self.sessions = {
            "sentiment": self._load_session(sentiment_dir),
            "actionability": self._load_session(actionability_dir),
            "event_type": self._load_session(event_type_dir),
        }
        self.id2label = {
            "sentiment": normalize_id2label(json.loads((sentiment_dir / "config.json").read_text(encoding="utf-8"))["id2label"]),
            "actionability": normalize_id2label(json.loads((actionability_dir / "config.json").read_text(encoding="utf-8"))["id2label"]),
            "event_type": normalize_id2label(json.loads((event_type_dir / "config.json").read_text(encoding="utf-8"))["id2label"]),
        }
        self.input_names = {
            name: [input_meta.name for input_meta in session.get_inputs()]
            for name, session in self.sessions.items()
        }

    def _load_session(self, artifact_dir: Path):
        providers = ["CPUExecutionProvider"]
        return self._ort.InferenceSession(str(artifact_dir / "model.onnx"), providers=providers)

    def encode(self, text: str) -> dict[str, np.ndarray]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="np",
        )
        return {key: value.astype(np.int64) for key, value in encoded.items()}

    def predict_head(
        self,
        task_name: str,
        encoded: dict[str, np.ndarray],
        label_map: dict[str, str] | None = None,
    ) -> dict[str, object]:
        session = self.sessions[task_name]
        id2label = self.id2label[task_name]
        feed = {
            name: encoded[name]
            for name in self.input_names[task_name]
            if name in encoded
        }
        logits = session.run(None, feed)[0]
        logits = np.asarray(logits)[0]
        probs = softmax_np(logits)
        return build_prediction_payload(probs=probs, id2label=id2label, label_map=label_map)


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - np.max(logits)
    exp_logits = np.exp(logits)
    return exp_logits / exp_logits.sum()


def build_prediction_payload(
    probs: np.ndarray,
    id2label: dict[int, str],
    label_map: dict[str, str] | None = None,
) -> dict[str, object]:
    label_probs = {
        id2label[index]: float(probs[index])
        for index in range(len(id2label))
    }
    best_index = int(np.argmax(probs))
    raw_label = id2label[best_index]
    label = label_map.get(raw_label, raw_label) if label_map else raw_label

    if label_map:
        mapped_probs = {
            label_map.get(raw_key, raw_key): score
            for raw_key, score in label_probs.items()
        }
    else:
        mapped_probs = label_probs

    return {
        "label": label,
        "confidence": float(probs[best_index]),
        "probs": mapped_probs,
        "raw_label": raw_label,
    }


def resolve_backend(explicit_backend: str | None) -> str:
    backend = (explicit_backend or os.getenv("NEWS_SCORER_BACKEND") or "onnx").strip().lower()
    if backend not in ALLOWED_BACKENDS:
        raise ValueError(f"Unsupported backend: {backend}. Allowed values: {sorted(ALLOWED_BACKENDS)}")
    return backend


class NewsScorerPredictor:
    def __init__(
        self,
        backend: str | None = None,
        sentiment_dir: Path | None = None,
        actionability_dir: Path | None = None,
        event_type_dir: Path | None = None,
        onnx_root: Path | None = None,
        max_length: int = 128,
    ) -> None:
        project_dir = Path(__file__).resolve().parent.parent
        torch_artifacts_dir = project_dir / "models" / "artifacts"
        onnx_artifacts_dir = (onnx_root or project_dir / "models" / "onnx_artifacts").resolve()

        self.backend_name = resolve_backend(backend)
        self.max_length = max_length

        self.sentiment_dir = (sentiment_dir or torch_artifacts_dir / "finbert_phrasebank").resolve()
        self.actionability_dir = (actionability_dir or torch_artifacts_dir / "finbert_actionability").resolve()
        self.event_type_dir = (event_type_dir or torch_artifacts_dir / "finbert_event_type_stage2").resolve()

        if self.backend_name == "onnx":
            self.onnx_root = onnx_artifacts_dir
            self.tokenizer_dir = (onnx_artifacts_dir / "tokenizer").resolve()
            self.runtime_artifacts = {
                "tokenizer": str(self.tokenizer_dir),
                "sentiment": str((onnx_artifacts_dir / "sentiment").resolve()),
                "actionability": str((onnx_artifacts_dir / "actionability").resolve()),
                "event_type": str((onnx_artifacts_dir / "event_type").resolve()),
            }
            self.backend = OnnxBackend(
                sentiment_dir=(onnx_artifacts_dir / "sentiment").resolve(),
                actionability_dir=(onnx_artifacts_dir / "actionability").resolve(),
                event_type_dir=(onnx_artifacts_dir / "event_type").resolve(),
                tokenizer_dir=self.tokenizer_dir,
                max_length=max_length,
            )
        else:
            self.onnx_root = onnx_artifacts_dir
            self.tokenizer_dir = self.sentiment_dir
            self.runtime_artifacts = {
                "tokenizer": str(self.tokenizer_dir),
                "sentiment": str(self.sentiment_dir),
                "actionability": str(self.actionability_dir),
                "event_type": str(self.event_type_dir),
            }
            self.backend = TorchBackend(
                sentiment_dir=self.sentiment_dir,
                actionability_dir=self.actionability_dir,
                event_type_dir=self.event_type_dir,
                tokenizer_dir=self.tokenizer_dir,
                max_length=max_length,
            )

        self.device = self.backend.device_name

    def _infer_horizon(self, actionability_label: str, event_type_label: str | None) -> str:
        if actionability_label == "non_actionable":
            return "not_actionable"
        if event_type_label is None:
            return "intraday"
        return HORIZON_MAP.get(event_type_label, "intraday")

    def _build_rationale(
        self,
        sentiment_label: str,
        actionability_label: str,
        event_type_label: str | None,
        horizon_label: str,
    ) -> str:
        if actionability_label == "non_actionable":
            return (
                "The text looks more like commentary or background information than a fresh tradeable event, "
                "so the service marks it as non-actionable."
            )

        if event_type_label is None:
            return (
                f"The text is classified as an actionable {sentiment_label} event with an expected {horizon_label} reaction window."
            )

        event_name = EVENT_TYPE_DISPLAY.get(event_type_label, event_type_label.replace("_", " "))
        return (
            f"The text is classified as an actionable {event_name} event with {sentiment_label} tone, "
            f"so the expected trading horizon is {horizon_label}."
        )

    def predict(self, text: str) -> dict[str, object]:
        text = str(text).strip()
        if not text:
            raise ValueError("text must be a non-empty string")

        started_at = time.perf_counter()
        encoded = self.backend.encode(text)

        sentiment = self.backend.predict_head(
            task_name="sentiment",
            encoded=encoded,
            label_map=SENTIMENT_MAP,
        )
        actionability = self.backend.predict_head(
            task_name="actionability",
            encoded=encoded,
        )

        event_type: dict[str, object] | None = None
        if actionability["label"] == "actionable":
            event_type = self.backend.predict_head(
                task_name="event_type",
                encoded=encoded,
            )

        latency_ms = (time.perf_counter() - started_at) * 1000.0
        horizon = self._infer_horizon(
            actionability_label=str(actionability["label"]),
            event_type_label=None if event_type is None else str(event_type["label"]),
        )
        rationale = self._build_rationale(
            sentiment_label=str(sentiment["label"]),
            actionability_label=str(actionability["label"]),
            event_type_label=None if event_type is None else str(event_type["label"]),
            horizon_label=horizon,
        )

        return {
            "text": text,
            "sentiment": sentiment,
            "actionability": actionability,
            "event_type": event_type,
            "horizon": horizon,
            "rationale": rationale,
            "meta": {
                "backend": self.backend_name,
                "device": self.backend.device_name,
                "latency_ms": round(latency_ms, 2),
                "max_length": self.max_length,
            },
        }

    def predict_for_ui(self, text: str) -> dict[str, str]:
        result = self.predict(text)
        sentiment = result["sentiment"]
        actionability = result["actionability"]
        event_type = result["event_type"]

        return {
            "sentiment": f"{sentiment['label']} ({format_confidence(sentiment['confidence'])})",
            "actionability": f"{actionability['label']} ({format_confidence(actionability['confidence'])})",
            "event_type": (
                "not_applicable"
                if event_type is None
                else f"{event_type['label']} ({format_confidence(event_type['confidence'])})"
            ),
            "horizon": str(result["horizon"]),
            "rationale": str(result["rationale"]),
            "raw_json": json.dumps(result, indent=2, ensure_ascii=False),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run unified local inference for news_scorer.")
    parser.add_argument("--text", required=True)
    parser.add_argument("--backend", choices=sorted(ALLOWED_BACKENDS), default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictor = NewsScorerPredictor(backend=args.backend)
    result = predictor.predict(args.text)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
