# News-to-Trade Relevance Scorer

Transformer-based decision-support demo for financial text triage.

HF Spaces target:
- primary backend: `ONNX Runtime`
- public entrypoint: `Gradio`
- local/dev fallback: `FastAPI + PyTorch`

The app returns:
- sentiment: `bullish` / `bearish` / `neutral`
- actionability: `actionable` / `non_actionable`
- event type
- heuristic trading horizon
- confidence scores
- a short rationale

## Runtime Layout

The service uses three separately trained FinBERT-based classifiers:
- PyTorch source artifacts in [models/artifacts](/Users/savidan/JupyterLabProjects/ysda/ml-2/hws/hw4/news_scorer/models/artifacts)
- compact ONNX export in `models/onnx_artifacts`

`horizon` is currently derived heuristically from `actionability + event_type`.

## ONNX Export

Export final serving artifacts:

```bash
cd hws/hw4/news_scorer
/opt/homebrew/anaconda3/envs/ml_env/bin/python serving/export_onnx.py
```

This creates:
- `models/onnx_artifacts/tokenizer`
- `models/onnx_artifacts/sentiment`
- `models/onnx_artifacts/actionability`
- `models/onnx_artifacts/event_type`

Only these compact ONNX artifacts should be used for HF Spaces runtime.

## Local Run

```bash
cd hws/hw4/news_scorer
/opt/homebrew/anaconda3/envs/ml_env/bin/python app.py
```

The Gradio app uses the ONNX backend by default.

## API

There is also a small FastAPI entrypoint in [serving/api.py](/Users/savidan/JupyterLabProjects/ysda/ml-2/hws/hw4/news_scorer/serving/api.py).

Local run:

```bash
cd hws/hw4/news_scorer/serving
/opt/homebrew/anaconda3/envs/ml_env/bin/python api.py
```

By default the local API uses the PyTorch backend. You can override it with:

```bash
export NEWS_SCORER_BACKEND=onnx
```

The local PyTorch path assumes `torch` is available in your local environment. It is not required for the HF Spaces ONNX runtime path.

## Notes

- This is a research/demo project, not an investment recommendation system.
- The service is optimized for short English financial text such as headlines, tweets, short news items, and press release excerpts.
- For HF Spaces, the main entrypoint is [app.py](/Users/savidan/JupyterLabProjects/ysda/ml-2/hws/hw4/news_scorer/app.py).
- For deployment, do not include training checkpoint folders; only final ONNX artifacts are needed at runtime.
