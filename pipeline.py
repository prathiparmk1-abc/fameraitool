#!/usr/bin/env python3
"""pipeline.py - leaf photo -> advice card (offline).

Arm Bharat SoC Challenge, Phase 1 prototype.

Usage (from ~/agri, with ~/agri-env active):
    python pipeline.py test_images/leaf.jpg
    python pipeline.py test_images/leaf.jpg --no-llm
    python pipeline.py --from-json result.json        # skip the classifier (testing)

Design (say this in the report):
  * The classifier decides WHAT the problem is.
  * Advice text comes from a human-checked knowledge base (kb.json). It is
    shown exactly as written; the LLM never rewrites it.
  * Llama 3.2 1B (ExecuTorch, temperature 0) writes only a short plain-language
    explanation from the knowledge-base facts. A code check rejects it if it
    contains numbers, doses, chemical/treatment words, or does not mention the
    problem; then the verified text is shown instead.
  * Uncertain results (low confidence or two close candidates) never name a
    disease and give only safe, general steps.
"""
import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
HOME = Path.home()
DEFAULT_RUNNER = HOME / "executorch/cmake-out/examples/models/llama/llama_main"
DEFAULT_MODEL = HOME / "llama-1b-spinquant/Llama-3.2-1B-Instruct-SpinQuant_INT4_EO8.pte"
DEFAULT_TOKENIZER = HOME / "llama-1b-spinquant/tokenizer.model"

MIN_CONFIDENCE = 0.60   # below this: do not name a disease
MIN_MARGIN = 0.20       # top-1 minus top-2 score; smaller means two close candidates

FOOTER = ("This tool supports decisions; it is not a diagnosis. For any treatment, "
          "follow your local agriculture officer's advice and the product label.")

SYSTEM_PROMPT = ("You are a careful farm advisor. Use only the facts provided. "
                 "Write two short sentences in simple English. Do not give any numbers, "
                 "doses, product names or chemicals.")

# Words the explanation must not contain (treatment advice comes only from kb.json).
BANNED = re.compile(
    r"\b(mg|ml|kg|gram|grams|litre|litres|liter|liters|dose|doses|dosage|"
    r"spray|sprays|spraying|sprayed|fungicide|fungicides|pesticide|pesticides|"
    r"bactericide|insecticide|mancozeb|chlorothalonil|copper|streptomycin|"
    r"metalaxyl|azoxystrobin|carbendazim|imidacloprid|sulphur|sulfur)\b",
    re.IGNORECASE,
)


# ----------------------------------------------------------------- knowledge base
def load_kb(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_lang(lang, *fields):
    """Use `lang` only if every field has it; otherwise fall back to English."""
    if all(isinstance(f, dict) and lang in f for f in fields):
        return lang
    return "en"


def build_steps(entry, severity, lang):
    steps = list(entry["steps_always"][lang])
    if severity in ("moderate", "severe", "unknown"):
        steps += entry.get("steps_moderate_or_severe", {}).get(lang, [])
    if severity == "severe":
        steps += entry.get("steps_severe", {}).get(lang, [])
    return steps


# ------------------------------------------------------------------ decision rule
def decide_route(result, kb):
    """Return (route, margin). Routes: ok, uncertain, not_covered."""
    top_k = result.get("top_k", [])
    margin = None
    if len(top_k) >= 2:
        margin = round(float(top_k[0]["score"]) - float(top_k[1]["score"]), 4)
    if result["confidence"] < MIN_CONFIDENCE or (margin is not None and margin < MIN_MARGIN):
        return "uncertain", margin
    key = f'{result["crop"]}/{result["problem"]}'
    if key not in kb["entries"]:
        return "not_covered", margin
    return "ok", margin


# ------------------------------------------------------------------------- Llama
def build_prompt(system, user):
    return ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" + system +
            "<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n" + user +
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n")


def extract_answer(raw):
    """Pull the assistant's reply out of llama_main's output (it echoes the prompt)."""
    marker = "<|start_header_id|>assistant<|end_header_id|>"
    idx = raw.rfind(marker)
    if idx < 0:
        return None
    text = raw[idx + len(marker):]
    for stop in ("<|eot_id|>", "PyTorchObserver"):
        cut = text.find(stop)
        if cut >= 0:
            text = text[:cut]
    return text.strip()


def extract_stats(raw):
    m = re.search(r"PyTorchObserver\s*(\{.*\})", raw)
    if not m:
        return {}
    try:
        data = json.loads(m.group(1))
    except json.JSONDecodeError:
        return {}
    return {
        "prefill_tokens_per_sec": data.get("prefill_token_per_sec"),
        "decode_tokens_per_sec": data.get("decode_token_per_sec"),
        "prompt_tokens": data.get("prompt_tokens"),
        "generated_tokens": data.get("generated_tokens"),
    }


def run_llama(runner, model, tokenizer, user_text, seq_len, timeout):
    """Run llama_main once. Returns dict(ok, text, stats, seconds, error)."""
    prompt = build_prompt(SYSTEM_PROMPT, user_text)
    cmd = [str(runner), f"--model_path={model}", f"--tokenizer_path={tokenizer}",
           f"--seq_len={seq_len}", "--temperature=0", f"--prompt={prompt}"]
    start = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "text": None, "stats": {}, "seconds": timeout, "error": "timeout"}
    except OSError as exc:
        return {"ok": False, "text": None, "stats": {}, "seconds": 0.0, "error": str(exc)}
    seconds = round(time.time() - start, 1)
    raw = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        return {"ok": False, "text": None, "stats": {}, "seconds": seconds,
                "error": f"llama_main exited with code {proc.returncode}"}
    return {"ok": True, "text": extract_answer(raw), "stats": extract_stats(raw),
            "seconds": seconds, "error": None}


def clean_answer(text):
    """Drop a one-line preamble such as 'Here are two short sentences:' (small models add it)."""
    if not text:
        return text
    t = text.strip()
    parts = t.split("\n", 1)
    if (len(parts) == 2 and parts[0].rstrip().endswith(":")
            and re.match(r"(?i)^(here|sure|okay|ok)\b", parts[0].strip())):
        t = parts[1].strip()
    return t


def validate_llm_text(text, keywords):
    """Code check on the model's explanation. Returns (accepted, reason)."""
    if not text:
        return False, "empty answer"
    t = text.strip()
    if len(t) < 20 or len(t) > 500:
        return False, "unexpected length"
    if "<|" in t or "|>" in t:
        return False, "contains special tokens"
    if re.search(r"\d", t):
        return False, "contains numbers"
    if ":" in t or re.match(r"(?i)^(here|sure|okay|ok)\b", t):
        return False, "starts with a preamble"
    if len(re.findall(r"[.!?](?:\s|$)", t)) > 3:
        return False, "too many sentences"
    if BANNED.search(t):
        return False, "mentions treatment, product or dose words"
    if keywords and not any(k.lower() in t.lower() for k in keywords):
        return False, "does not mention the problem"
    return True, "ok"


# --------------------------------------------------------------------------- card
def build_card(result, kb, args):
    lang_req = args.language
    route, margin = decide_route(result, kb)
    card = {
        "route": route,
        "confidence": result["confidence"],
        "margin": margin,
        "severity": result.get("severity"),
        "region": result.get("region"),
        "language_requested": lang_req,
        "llm": {"used": False, "accepted": False, "reason": None, "text": None,
                "seconds": None, "stats": {}},
    }

    if route == "ok":
        key = f'{result["crop"]}/{result["problem"]}'
        entry = kb["entries"][key]
        lang = pick_lang(lang_req, entry["display_name"], entry["what_it_is"],
                         entry["steps_always"], entry["when_to_get_help"])
        card.update({
            "language_shown": lang,
            "title": entry["display_name"][lang],
            "explanation": entry["what_it_is"][lang],
            "steps": build_steps(entry, result.get("severity"), lang),
            "prevention": entry.get("prevention", {}).get(lang, []),
            "when_to_get_help": entry["when_to_get_help"][lang],
            "explanation_source": "verified knowledge base",
            "check_leaf": [],
        })
        if entry.get("look_for") and entry.get("confusable_with"):
            keys = [key] + list(entry["confusable_with"])
            for k in keys:
                other = kb["entries"].get(k)
                if other and other.get("look_for") and lang in other["look_for"]:
                    card["check_leaf"].append(f"{other['display_name'][lang]}: {other['look_for'][lang]}")
        is_disease = result["problem"] != "healthy"
        if is_disease and not args.no_llm and lang == "en":
            user_text = (f"Facts: {entry['what_it_is']['en']}\n"
                         f"The plant shows: {entry['display_name']['en']} "
                         f"(severity: {result.get('severity')}).\n"
                         "In two short sentences, explain what this problem is, using only the facts.")
            out = run_llama(args.runner, args.model, args.tokenizer, user_text,
                            args.seq_len, args.timeout)
            card["llm"].update({"used": True, "seconds": out["seconds"], "stats": out["stats"]})
            if out["ok"]:
                cleaned = clean_answer(out["text"])
                accepted, reason = validate_llm_text(cleaned, entry.get("keywords", []))
                card["llm"].update({"accepted": accepted, "reason": reason,
                                    "text": cleaned, "raw_text": out["text"]})
                if accepted:
                    card["explanation"] = cleaned
                    card["explanation_source"] = "Llama 3.2 1B on-device (ExecuTorch), code-checked"
            else:
                card["llm"]["reason"] = out["error"]
    else:
        generic = kb["generic"][route]
        lang = pick_lang(lang_req, generic["title"], generic["message"], generic["steps"])
        top_k = result.get("top_k", [])[:2]
        card.update({
            "language_shown": lang,
            "title": generic["title"][lang],
            "explanation": generic["message"][lang],
            "steps": list(generic["steps"][lang]),
            "prevention": [],
            "when_to_get_help": "",
            "candidates": [t["label"] for t in top_k] if route == "uncertain" else [],
            "explanation_source": "verified knowledge base",
        })
    return card


def render_card(card, kb):
    bar = "=" * 64
    lines = [bar, card["title"]]
    if card["route"] == "ok":
        lines.append(f"Severity: {card['severity']}   Confidence: {card['confidence']:.2f}"
                     f"   Region: {card['region']}")
    else:
        lines.append(f"Confidence: {card['confidence']:.2f}")
    lines += [bar, card["explanation"], ""]
    if card.get("candidates"):
        lines.append("Possible, but not certain: " + " / ".join(card["candidates"]))
        lines.append("")
    if card.get("check_leaf"):
        lines.append("Check the leaf before you act (these two are easily confused):")
        lines += [f"  - {c}" for c in card["check_leaf"]]
        lines.append("")
    if card["steps"]:
        lines.append("What to do:")
        lines += [f"  {i}. {s}" for i, s in enumerate(card["steps"], 1)]
        lines.append("")
    if card.get("prevention"):
        lines.append("Next season:")
        lines += [f"  - {p}" for p in card["prevention"]]
        lines.append("")
    if card.get("when_to_get_help"):
        lines += ["When to get help:", f"  {card['when_to_get_help']}", ""]
    lines.append("-" * 64)
    lines.append(FOOTER)
    note = f"Explanation source: {card['explanation_source']}"
    llm = card["llm"]
    if llm["used"]:
        if llm["accepted"]:
            tps = llm["stats"].get("decode_tokens_per_sec")
            note += f" ({tps:.1f} tokens/s decode, {llm['seconds']} s total)" if tps else ""
        else:
            note += f" (Llama explanation not used: {llm['reason']})"
    lines.append(note)
    if card.get("language_shown") != card["language_requested"]:
        lines.append(f"Note: '{card['language_requested']}' text is not available yet; showing English.")
    if kb.get("meta", {}).get("status") != "REVIEWED":
        lines.append("DRAFT knowledge base: advice text and sources not yet verified.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Leaf photo -> advice card")
    ap.add_argument("image", nargs="?", help="leaf photo")
    ap.add_argument("--from-json", help="use a saved classify.py result instead of a photo")
    ap.add_argument("--kb", default=str(HERE / "kb.json"))
    ap.add_argument("--region", default="Karnataka")
    ap.add_argument("--language", default="en")
    ap.add_argument("--no-llm", action="store_true", help="show verified text only")
    ap.add_argument("--runner", default=str(DEFAULT_RUNNER))
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--tokenizer", default=str(DEFAULT_TOKENIZER))
    ap.add_argument("--seq-len", type=int, default=400)
    ap.add_argument("--timeout", type=int, default=180)
    ap.add_argument("--save", help="write the full result and card as JSON to this file")
    args = ap.parse_args()

    kb = load_kb(args.kb)

    if args.from_json:
        with open(args.from_json, "r", encoding="utf-8") as f:
            result = json.load(f)
        result.setdefault("region", args.region)
        result.setdefault("language", args.language)
    elif args.image:
        import classify  # same folder; imports torch only when the model is loaded
        model, torch = classify.load_model()
        result = classify.classify_image(model, torch, Path(args.image), 3,
                                         args.region, args.language)
    else:
        ap.error("give a photo, or use --from-json")

    card = build_card(result, kb, args)
    print(render_card(card, kb))

    if args.save:
        with open(args.save, "w", encoding="utf-8") as f:
            json.dump({"classifier": result, "card": card}, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
