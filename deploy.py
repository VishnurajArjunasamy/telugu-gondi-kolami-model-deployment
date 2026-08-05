#!/usr/bin/env python
# coding: utf-8
"""
Two-model NLLB translation API.

  v3   — vishnu-vizz/nllb-telugu-gondi-v3          tel→gon only
  v3.1 — nllb_telgon_exp1_bidir (local)             tel↔gon both directions

Routes:
  POST /v3/translate
  POST /v3.1/translate
  GET  /v3/health
  GET  /v3.1/health
  GET  /health
"""

import os
os.environ["CUDA_VISIBLE_DEVICES"]   = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"))

HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise RuntimeError("HF_TOKEN not found in .env")
login(token=HF_TOKEN)

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
MAX_SRC_LENGTH = 256
MAX_TGT_LENGTH = 128
PORT           = 5000

_MODEL_CONFIGS = {
    "v3": {
        "path":            "./hf_models/nllb-telugu-gondi-v3",
        "supported_pairs": {("tel_Telu", "wsg_Telu")},
    },
    "v3.1": {
        "path":            "./hf_models/nllb-telugu-gondi-v3.1",
        "supported_pairs": {("tel_Telu", "wsg_Telu"), ("wsg_Telu", "tel_Telu")},
    },
}

# ─────────────────────────────────────────────
# LOAD MODELS
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
# TRANSLATE
# ─────────────────────────────────────────────
def translate(version: str, src_sentence: str, src_lang: str, tgt_lang: str) -> str:
    tok = MODELS[version]["tokenizer"]
    mdl = MODELS[version]["model"]

    tok.src_lang = src_lang
    tgt_token_id = tok.convert_tokens_to_ids(tgt_lang)

    inputs = tok(
        src_sentence,
        return_tensors = "pt",
        truncation     = True,
        max_length     = MAX_SRC_LENGTH,
    )
    src_len        = inputs["input_ids"].shape[1]
    max_new_tokens = min(int(src_len * 1.5), MAX_TGT_LENGTH)
    inputs         = {k: v.to("cuda") for k, v in inputs.items()}

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

# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)

def _translate_endpoint(version: str):
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    src_sentence = data.get("src_sentence", "").strip()
    src_lang     = data.get("src_lang", "").strip()
    tgt_lang     = data.get("tgt_lang", "").strip()

    if not src_sentence:
        return jsonify({"error": "src_sentence is required"}), 400
    if not src_lang:
        return jsonify({"error": "src_lang is required"}), 400
    if not tgt_lang:
        return jsonify({"error": "tgt_lang is required"}), 400

    supported = MODELS[version]["supported_pairs"]
    if (src_lang, tgt_lang) not in supported:
        return jsonify({
            "error": f"Unsupported pair for {version}: {src_lang} → {tgt_lang}",
            "supported_pairs": [{"src_lang": s, "tgt_lang": t} for s, t in supported],
        }), 400

    tok = MODELS[version]["tokenizer"]
    if tok.convert_tokens_to_ids(src_lang) == tok.unk_token_id:
        return jsonify({"error": f"Unknown src_lang token: {src_lang}"}), 400
    if tok.convert_tokens_to_ids(tgt_lang) == tok.unk_token_id:
        return jsonify({"error": f"Unknown tgt_lang token: {tgt_lang}"}), 400

    try:
        tgt_sentence = translate(version, src_sentence, src_lang, tgt_lang)
        return jsonify({
            "model":        version,
            "src_lang":     src_lang,
            "tgt_lang":     tgt_lang,
            "src_sentence": src_sentence,
            "tgt_sentence": tgt_sentence,
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
                            for s, t in cfg["supported_pairs"]],
    }), 200


# per-version routes
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

# overall health
@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "models": {
            v: {
                "path":            cfg["path"],
                "supported_pairs": [{"src_lang": s, "tgt_lang": t}
                                    for s, t in cfg["supported_pairs"]],
            }
            for v, cfg in MODELS.items()
        },
    }), 200


if __name__ == "__main__":
    print(f"Starting API on http://0.0.0.0:{PORT}")
    print("Endpoints:")
    print(f"  POST http://0.0.0.0:{PORT}/v3/translate      (tel→gon)")
    print(f"  POST http://0.0.0.0:{PORT}/v3.1/translate    (tel↔gon)")
    print(f"  GET  http://0.0.0.0:{PORT}/v3/health")
    print(f"  GET  http://0.0.0.0:{PORT}/v3.1/health")
    print(f"  GET  http://0.0.0.0:{PORT}/health\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
