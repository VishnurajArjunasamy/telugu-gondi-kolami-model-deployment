#!/usr/bin/env python
# coding: utf-8
"""
Telugu-Gondi Translation API

Local models (NLLB-200 1.3B + LoRA):
  v3   — tel → gon
  v3.1 — tel ↔ gon

Bhashaverse api (https://ssmt.iiit.ac.in):
  eng ↔ tel  (direct)
  eng → gon  (cascade: eng→tel via IIIT, tel→gon via v3.1)
  gon → eng  (cascade: gon→tel via v3.1, tel→eng via IIIT)

All endpoints support paragraph-level input — text is split on sentence
boundaries, each sentence translated independently, then rejoined.

Routes:
  POST /v3/translate        tel→gon
  POST /v3.1/translate      tel↔gon, eng↔tel, eng↔gon, gon↔eng
  GET  /v3/health
  GET  /v3.1/health
  GET  /health
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]   = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import re
import requests
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, request, jsonify

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_SRC_LENGTH = 256
MAX_TGT_LENGTH = 128
PORT           = 5000

IIIT_API_URL = "https://ssmt.iiit.ac.in/api/onemtapi/v1/translate"

# NLLB lang code → IIIT lang code
_NLLB_TO_IIIT = {
    "eng_Latn": "eng",
    "tel_Telu": "tel",
}

_MODEL_CONFIGS = {
    "v3": {
        "path":            "./hf_models/nllb-telugu-gondi-v3",
        "supported_pairs": {
            ("tel_Telu", "wsg_Telu"),
            ("eng_Latn", "tel_Telu"),
            ("tel_Telu", "eng_Latn"),
            ("eng_Latn", "wsg_Telu"),
        },
    },
    "v3.1": {
        "path":            "./hf_models/nllb-telugu-gondi-v3.1",
        "supported_pairs": {
            ("tel_Telu", "wsg_Telu"),
            ("wsg_Telu", "tel_Telu"),
            ("eng_Latn", "tel_Telu"),
            ("tel_Telu", "eng_Latn"),
            ("eng_Latn", "wsg_Telu"),
            ("wsg_Telu", "eng_Latn"),
        },
    },
}

# Pairs that go through the IIIT API (directly or as a cascade leg)
_IIIT_DIRECT = {("eng_Latn", "tel_Telu"), ("tel_Telu", "eng_Latn")}
# Pairs that need cascade: eng↔gon via tel as pivot
_CASCADE = {
    ("eng_Latn", "wsg_Telu"): [("eng_Latn", "tel_Telu"), ("tel_Telu", "wsg_Telu")],
    ("wsg_Telu", "eng_Latn"): [("wsg_Telu", "tel_Telu"), ("tel_Telu", "eng_Latn")],
}

# ─────────────────────────────────────────────
# LOAD LOCAL MODELS
# ─────────────────────────────────────────────
MODELS = {}

for version, cfg in _MODEL_CONFIGS.items():
    print(f"\nLoading {version}  ({cfg['path']}) …")
    tok = AutoTokenizer.from_pretrained(cfg["path"])
    mdl = AutoModelForSeq2SeqLM.from_pretrained(
        cfg["path"],
        torch_dtype = torch.float16,
        device_map  = {"": 0},
    )
    mdl.eval()
    MODELS[version] = {"tokenizer": tok, "model": mdl, **cfg}
    print(f"  {version} ready — vocab {len(tok):,}")

print("\nAll models loaded.\n" + "=" * 60)

# ─────────────────────────────────────────────
# SENTENCE SPLITTER  (paragraph → sentence list)
# ─────────────────────────────────────────────
_SENT_SPLIT = re.compile(r'(?<=[।.?!])\s+')

def split_sentences(text: str) -> list[str]:
    parts = _SENT_SPLIT.split(text.strip())
    return [p.strip() for p in parts if p.strip()]

# ─────────────────────────────────────────────
# TRANSLATION PRIMITIVES
# ─────────────────────────────────────────────
_CHUNK_TOKEN_LIMIT = 200  # tokens per chunk sent to local model

def _local_translate_chunk(tok, mdl, chunk: str, tgt_token_id: int) -> str:
    """Translate a single chunk that is guaranteed to fit within token limits."""
    inputs = tok(chunk, return_tensors="pt", truncation=True, max_length=MAX_SRC_LENGTH)
    max_new_tokens = min(int(inputs["input_ids"].shape[1] * 1.5), MAX_TGT_LENGTH)
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    with torch.inference_mode():
        outputs = mdl.generate(
            **inputs,
            forced_bos_token_id  = tgt_token_id,
            max_new_tokens       = max_new_tokens,
            num_beams            = 10,
            no_repeat_ngram_size = 3,
            repetition_penalty   = 1.3,
            early_stopping       = True,
        )
    return tok.decode(outputs[0], skip_special_tokens=True).strip()


def _local_translate(version: str, sentence: str, src_lang: str, tgt_lang: str) -> str:
    tok = MODELS[version]["tokenizer"]
    mdl = MODELS[version]["model"]
    tok.src_lang = src_lang
    tgt_token_id = tok.convert_tokens_to_ids(tgt_lang)

    # Check if the sentence fits within the token budget
    token_ids = tok(sentence, add_special_tokens=True)["input_ids"]
    if len(token_ids) <= MAX_SRC_LENGTH:
        return _local_translate_chunk(tok, mdl, sentence, tgt_token_id)

    # Too long — split by words into token-budget chunks, translate each, rejoin
    words  = sentence.split()
    chunks, buf, buf_len = [], [], 0
    for word in words:
        w_len = len(tok(word, add_special_tokens=False)["input_ids"])
        if buf and buf_len + w_len > _CHUNK_TOKEN_LIMIT:
            chunks.append(" ".join(buf))
            buf, buf_len = [], 0
        buf.append(word)
        buf_len += w_len
    if buf:
        chunks.append(" ".join(buf))

    return " ".join(
        _local_translate_chunk(tok, mdl, chunk, tgt_token_id)
        for chunk in chunks
    )


def _iiit_translate(sentence: str, src_lang: str, tgt_lang: str) -> str:
    payload = {
        "task":            "Translation",
        "domain":          "General",
        "source_language": _NLLB_TO_IIIT[src_lang],
        "target_language": _NLLB_TO_IIIT[tgt_lang],
        "text":            sentence,
        "ttext":           "",
    }
    resp = requests.post(IIIT_API_URL, json=payload, timeout=30)
    resp.raise_for_status()
    result = resp.json()
    if result.get("error"):
        raise RuntimeError(f"IIIT API error: {result}")
    return result["data"]


def _translate_sentence(version: str, sentence: str, src_lang: str, tgt_lang: str) -> str:
    pair = (src_lang, tgt_lang)
    if pair in _CASCADE:
        # two-step cascade
        intermediate = None
        for step_src, step_tgt in _CASCADE[pair]:
            text = sentence if intermediate is None else intermediate
            if (step_src, step_tgt) in _IIIT_DIRECT:
                intermediate = _iiit_translate(text, step_src, step_tgt)
            else:
                intermediate = _local_translate(version, text, step_src, step_tgt)
        return intermediate
    elif pair in _IIIT_DIRECT:
        return _iiit_translate(sentence, src_lang, tgt_lang)
    else:
        return _local_translate(version, sentence, src_lang, tgt_lang)


def translate_text(version: str, text: str, src_lang: str, tgt_lang: str) -> str:
    """Split paragraph into sentences, translate each, rejoin."""
    sentences = split_sentences(text)
    if not sentences:
        return ""
    translated = [_translate_sentence(version, s, src_lang, tgt_lang) for s in sentences]
    return " ".join(translated)

# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)

def _translate_endpoint(version: str):
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    src_text = data.get("src_sentence", "").strip()
    src_lang  = data.get("src_lang", "").strip()
    tgt_lang  = data.get("tgt_lang", "").strip()

    if not src_text:
        return jsonify({"error": "src_sentence is required"}), 400
    if not src_lang:
        return jsonify({"error": "src_lang is required"}), 400
    if not tgt_lang:
        return jsonify({"error": "tgt_lang is required"}), 400

    supported = MODELS[version]["supported_pairs"]
    if (src_lang, tgt_lang) not in supported:
        return jsonify({
            "error": f"Unsupported pair for {version}: {src_lang} → {tgt_lang}",
            "supported_pairs": [{"src_lang": s, "tgt_lang": t} for s, t in sorted(supported)],
        }), 400

    try:
        tgt_text = translate_text(version, src_text, src_lang, tgt_lang)
        return jsonify({
            "model":        version,
            "src_lang":     src_lang,
            "tgt_lang":     tgt_lang,
            "src_sentence": src_text,
            "tgt_sentence": tgt_text,
        }), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _health_endpoint(version: str):
    cfg = MODELS[version]
    return jsonify({
        "status":          "ok",
        "model":           version,
        "path":            cfg["path"],
        "supported_pairs": [{"src_lang": s, "tgt_lang": t}
                            for s, t in sorted(cfg["supported_pairs"])],
    }), 200


@app.route("/v3/translate", methods=["POST"])
def v3_translate():
    return _translate_endpoint("v3")

@app.route("/v3.1/translate", methods=["POST"])
def v31_translate():
    return _translate_endpoint("v3.1")

@app.route("/v3/health", methods=["GET"])
def v3_health():
    return _health_endpoint("v3")

@app.route("/v3.1/health", methods=["GET"])
def v31_health():
    return _health_endpoint("v3.1")

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            v: {
                "path":            cfg["path"],
                "supported_pairs": [{"src_lang": s, "tgt_lang": t}
                                    for s, t in sorted(cfg["supported_pairs"])],
            }
            for v, cfg in MODELS.items()
        },
    }), 200


if __name__ == "__main__":
    print(f"Starting API on http://0.0.0.0:{PORT}")
    print("Endpoints:")
    print(f"  POST http://0.0.0.0:{PORT}/v3/translate      tel→gon")
    print(f"  POST http://0.0.0.0:{PORT}/v3.1/translate    tel↔gon, eng↔tel, eng↔gon, gon↔eng")
    print(f"  GET  http://0.0.0.0:{PORT}/v3/health")
    print(f"  GET  http://0.0.0.0:{PORT}/v3.1/health")
    print(f"  GET  http://0.0.0.0:{PORT}/health\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
