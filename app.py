from __future__ import annotations

from functools import lru_cache

import gradio as gr

from serving.predictor import NewsScorerPredictor


EXAMPLES = [
    "Company cuts full-year guidance amid weak demand",
    "CEO resigns effective immediately after accounting probe",
    "Analysts discuss whether tech may outperform next year",
    "Drugmaker wins FDA approval for new cancer treatment",
]


@lru_cache(maxsize=1)
def get_predictor() -> NewsScorerPredictor:
    return NewsScorerPredictor(backend="onnx")


def run_inference(text: str) -> tuple[str, str, str, str, str, str]:
    try:
        result = get_predictor().predict_for_ui(text)
    except ValueError as error:
        message = str(error)
        return message, message, message, message, message, "{}"

    return (
        result["sentiment"],
        result["actionability"],
        result["event_type"],
        result["horizon"],
        result["rationale"],
        result["raw_json"],
    )


with gr.Blocks(title="News-to-Trade Relevance Scorer") as demo:
    gr.Markdown(
        """
        # News-to-Trade Relevance Scorer
        Compact ONNX-powered financial NLP demo for:
        - sentiment: bullish / bearish / neutral
        - actionability: actionable / non_actionable
        - event type
        - heuristic trading horizon

        This is a decision-support demo, not investment advice.
        """
    )

    text_input = gr.Textbox(
        label="Financial text",
        lines=4,
        placeholder="Paste a headline, tweet, short news item, or press release excerpt...",
    )
    submit_btn = gr.Button("Score")

    with gr.Row():
        sentiment_output = gr.Textbox(label="Sentiment")
        actionability_output = gr.Textbox(label="Actionability")

    with gr.Row():
        event_type_output = gr.Textbox(label="Event Type")
        horizon_output = gr.Textbox(label="Trading Horizon")

    rationale_output = gr.Textbox(label="Rationale", lines=3)
    raw_json_output = gr.Code(label="Raw JSON", language="json")

    gr.Examples(examples=EXAMPLES, inputs=text_input)

    submit_btn.click(
        fn=run_inference,
        inputs=text_input,
        outputs=[
            sentiment_output,
            actionability_output,
            event_type_output,
            horizon_output,
            rationale_output,
            raw_json_output,
        ],
    )
    text_input.submit(
        fn=run_inference,
        inputs=text_input,
        outputs=[
            sentiment_output,
            actionability_output,
            event_type_output,
            horizon_output,
            rationale_output,
            raw_json_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
