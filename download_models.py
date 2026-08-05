#!/usr/bin/env python
# coding: utf-8
"""
Download HuggingFace models to local disk.
Run once before starting deploy.py.
"""

import os
from huggingface_hub import snapshot_download



MODELS = {
    "vishnu-vizz/nllb-telugu-gondi-v3":   "./hf_models/nllb-telugu-gondi-v3",
    "vishnu-vizz/nllb-telugu-gondi-v3.1": "./hf_models/nllb-telugu-gondi-v3.1",
}

for repo_id, local_dir in MODELS.items():
    if os.path.isdir(local_dir) and os.listdir(local_dir):
        print(f"[skip] {repo_id} — already at {local_dir}")
        continue
    print(f"\nDownloading {repo_id} → {local_dir} …")
    snapshot_download(
        repo_id   = repo_id,
        local_dir = local_dir,
    )
    print(f"  Done — {local_dir}")

print("\nAll models downloaded.")
