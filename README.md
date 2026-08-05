# NLLB Telugu-Gondi MT 

## Requirements

- Python 3.10
- CUDA compatible GPU (≥ 8 GB VRAM single model · ≥ 12 GB for both)
- ~10 GB disk space for downloaded models

---

## Models

| Version | HF repo | Directions | chrF++ | BLEU |
|---|---|---|---:|---:|
| v3 | `vishnu-vizz/nllb-telugu-gondi-v3` | tel → gon | 42.15 | 21.88 |
| v3.1 | `vishnu-vizz/nllb-telugu-gondi-v3.1` | tel → gon | 41.18 | 21.14 |
| v3.1 | `vishnu-vizz/nllb-telugu-gondi-v3.1` | gon → tel | 53.17 | 31.56 |

---

## 1. Clone the repo

```bash
git clone https://github.com/VishnurajArjunasamy/telugu-gondi-kolami-model-deployment
cd telugu-gondi-kolami-model-deployment
```

---

## 2. Create and activate the venv

```bash
python3.10 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install packages

```bash
# PyTorch with CUDA — adjust the cu12x suffix to match your driver
pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu124

pip install -r requirements.txt
```

---

## 4. Download models

Run once — pulls both models from HuggingFace Hub to local disk:

```bash
python download_models.py
```

Downloads to:

```
hf_models/
├── nllb-telugu-gondi-v3/
└── nllb-telugu-gondi-v3.1/
```

Re-running is safe — already-downloaded models are skipped automatically.

---

## 5. Start the API server

```bash
python deploy.py
```

Starts on `http://0.0.0.0:5000`.

---

## 6. API

### Language codes

| Language | Code |
|---|---|
| Telugu | `tel_Telu` |
| Gondi (Adilabad) | `wsg_Telu` |

---

### `POST /v3/translate` — Telugu → Gondi only

```bash
curl -X POST http://localhost:5000/v3/translate \
  -H "Content-Type: application/json" \
  -d '{"src_sentence": "నీళ్ళు తాగండి", "src_lang": "tel_Telu", "tgt_lang": "wsg_Telu"}'
```

---

### `POST /v3.1/translate` — Telugu ↔ Gondi

```bash
# Telugu → Gondi
curl -X POST http://localhost:5000/v3.1/translate \
  -H "Content-Type: application/json" \
  -d '{"src_sentence": "నీళ్ళు తాగండి", "src_lang": "tel_Telu", "tgt_lang": "wsg_Telu"}'

# Gondi → Telugu
curl -X POST http://localhost:5000/v3.1/translate \
  -H "Content-Type: application/json" \
  -d '{"src_sentence": "నీళ్ తాగ్వల్", "src_lang": "wsg_Telu", "tgt_lang": "tel_Telu"}'
```

**Response:**

```json
{
  "model": "v3.1",
  "src_lang": "tel_Telu",
  "tgt_lang": "wsg_Telu",
  "src_sentence": "నీళ్ళు తాగండి",
  "tgt_sentence": "..."
}
```

---

### `GET /health`

```bash
curl http://localhost:5000/health          # both models
curl http://localhost:5000/v3/health       # v3 only
curl http://localhost:5000/v3.1/health     # v3.1 only
```
