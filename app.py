#!/usr/bin/env python3
"""app.py - small local web app for the crop advisory pipeline.

Run (from ~/agri, with ~/agri-env active):
    python app.py
then open http://127.0.0.1:7860 in the browser on the same laptop.

Everything runs locally. After the first model download there is no network
traffic (set HF_HUB_OFFLINE=1 to be sure). Gradio is only the screen; all the
decisions are made by pipeline.py and classify.py.
"""
import json
import types
from pathlib import Path

import classify
import pipeline

HERE = Path(__file__).resolve().parent
DEFAULT_REGION = "Karnataka"

INTRO = """# Crop disease advisor - offline prototype
Upload **one leaf** photographed in daylight, in focus, on a plain background.
The photo is analysed on this device. Advice text comes from a checked knowledge
base; the on-device Llama model only writes a short explanation.

*Prototype for decision support, not a diagnosis.*"""

_MODEL = None  # (model, torch), loaded once on first use


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = classify.load_model()
    return _MODEL


def run_card(image_path, language="en", use_llm=True, region=DEFAULT_REGION, kb_path=None):
    """Photo path -> (advice card text, technical details JSON text)."""
    if not image_path:
        return "Please upload a leaf photo first.", ""
    kb = pipeline.load_kb(kb_path or HERE / "kb.json")
    model, torch = get_model()
    result = classify.classify_image(model, torch, Path(image_path), 3, region, language)
    args = types.SimpleNamespace(
        language=language,
        no_llm=not use_llm,
        runner=pipeline.DEFAULT_RUNNER,
        model=pipeline.DEFAULT_MODEL,
        tokenizer=pipeline.DEFAULT_TOKENIZER,
        seq_len=400,
        timeout=180,
    )
    card = pipeline.build_card(result, kb, args)
    details = json.dumps({"classifier": result, "card": card}, indent=2, ensure_ascii=False)
    return pipeline.render_card(card, kb), details


def build_ui():
    import gradio as gr  # imported here so the helpers above can be tested without it

    with gr.Blocks(title="Crop disease advisor (offline prototype)") as demo:
        gr.Markdown(INTRO)
        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="filepath", label="Leaf photo")
                language = gr.Dropdown(["en"], value="en", label="Language")
                use_llm = gr.Checkbox(value=True, label="Explain with on-device Llama (ExecuTorch)")
                button = gr.Button("Analyse photo", variant="primary")
            with gr.Column(scale=2):
                card = gr.Textbox(label="Advice", lines=28)
                details = gr.Textbox(label="Technical details (JSON)", lines=10)
        button.click(
            lambda img, lang, llm: run_card(img, lang, llm),
            inputs=[image, language, use_llm],
            outputs=[card, details],
        )
    return demo


if __name__ == "__main__":
    print("Loading the classifier (first start takes a few seconds)...")
    get_model()
    build_ui().launch(server_name="127.0.0.1", server_port=7860)
