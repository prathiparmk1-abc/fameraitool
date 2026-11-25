# Offline Crop Disease Advisory on Arm (Llama 3.2 1B + ExecuTorch)

Arm Bharat SoC Challenge - Problem Statement 1 - Phase 1 prototype.

A farmer photographs a tomato leaf. The app names the likely problem, estimates
severity and shows a short advice card. Everything runs locally; the Llama model
also runs on an Arm smartphone through ExecuTorch.

**Team:** add names here  |  **Report:** add link or file name  |  **Demo video:** add link

## How it works

```
leaf photo -> classifier (MobileNetV2, PlantVillage) -> problem + confidence
          -> severity rule (OpenCV colour heuristic)
          -> decision rule (not sure / not covered / ok)
          -> advice from kb.json (checked text, shown as written)
          -> Llama 3.2 1B via ExecuTorch writes a short explanation (temperature 0)
          -> code check rejects unsafe explanations, falls back to kb.json text
```

Why the advice text is not written by the model: in our tests a 1B model dropped a
safety step, changed a meaning and could not write usable Hindi (see the report).
Chemical treatment is always deferred to the local agriculture officer and the
product label.

## Files

| File | Purpose |
|---|---|
| `classify.py` | Photo -> JSON (crop, problem, confidence, severity). Preprocessing is written by hand |
| `pipeline.py` | Decision rule, knowledge base lookup, Llama call, output check, advice card |
| `kb.json` | Knowledge base (DRAFT: sources and expert review still to be added) |
| `app.py` | Local web app (Gradio) |
| `evaluate.py` | Accuracy report and variant comparison over labelled photo folders |
| `android/` | `run_phone.sh` and `phone_prompt.txt` used to benchmark Llama on the phone |
| `requirements.txt` | Python packages for the classifier and app |

## Setup (laptop, Ubuntu or WSL2)

1. Build ExecuTorch from source and the Llama runner (`llama_main`) following the
   ExecuTorch Llama README (Step 3). Request access to Llama 3.2 on Hugging Face.
2. Download the pre-exported model and tokenizer:
   `hf download executorch-community/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8-ET --local-dir ~/llama-1b-spinquant`
3. Create a separate environment for the classifier and app:
   ```bash
   uv venv --python 3.11 ~/agri-env
   source ~/agri-env/bin/activate
   uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
   uv pip install -r requirements.txt
   ```
4. Run:
   ```bash
   python classify.py test_images/leaf.jpg          # classifier only
   python pipeline.py test_images/leaf.jpg          # full card on the command line
   python pipeline.py test_images/leaf.jpg --no-llm # verified text only
   python app.py                                    # web app at http://127.0.0.1:7860
   ```
   The default paths for `llama_main`, the `.pte` model and the tokenizer are at the
   top of `pipeline.py`; change them if yours differ.
5. Offline use: after the first run downloads the classifier, set `HF_HUB_OFFLINE=1`.

## Accuracy options and evaluation

`classify.py` has four optional modes (`--variant`, or `export CROP_VARIANT=...`):
`plain` (default), `tta` (average of 4 mirrored/resized views), `leafcrop` (crop to the
leaf region first) and `tta+leafcrop`. They are cheap and often help on real photos,
but they are not guaranteed to, so measure them on labelled photos:

```
eval_images/early_blight/  eval_images/late_blight/  eval_images/bacterial_spot/  eval_images/healthy/
python evaluate.py eval_images --variants all
```

`evaluate.py` prints accuracy, how many confident answers were wrong, a confusion
table, a comparison of the variants and a confidence-threshold table. Use it to choose
the variant and to set `MIN_CONFIDENCE` in `pipeline.py`.

## Llama on an Arm phone (Android)

Cross-compile with Android NDK r28c (`ANDROID_ABI=arm64-v8a`,
`ANDROID_PLATFORM=android-23`, XNNPACK, optimized, quantized and LLM kernels on),
then copy `llama_main`, the `.pte` model and `tokenizer.model` to
`/data/local/tmp/llama` with `adb push`, run `adb shell chmod +x`, and start the
runner with `adb shell`. Details are in the report (Sections 4 and 9).

## Results (same 126-token prompt, 42-token answer, temperature 0)

| | Arm phone (Redmi Note 10 Pro, arm64-v8a) | x86 laptop |
|---|---|---|
| Prefill | 61.6 tokens/s | 148.6 tokens/s |
| Decode | 12.9 tokens/s | 26.1 tokens/s |
| Model load | 4.5 s | about 2.4 s |
| Memory after load | about 1.37 GB | not measured |

The answer text was identical on both devices.

## Limitations

- The classifier was trained on laboratory photos and can be confidently wrong on field photos.
- Severity is a rough colour rule. Advice exists for four tomato conditions only.
- The knowledge base is a draft and must be checked against ICAR, a state agricultural
  university or a Krishi Vigyan Kendra before real use.
- English only. No pest detection and no region-specific content yet.
- Only Llama runs on the phone; the classifier and app run on the laptop.

## Licences

Llama 3.2 Community License (Meta). ExecuTorch and XNNPACK are open source.
Classifier: `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`
(Hugging Face; licence listed as "other", confirm before reuse).
PlantVillage dataset: add citation and licence.
