"""
Task embeddings of a list of instructions with a SentenceTransformer model, saved for
task_embedding_format=precomputed (libero/lifelong/utils.py get_task_embs, cfg.task_embedding_file).
Needs sentence-transformers, which the training env does not have; run it in a separate env, e.g.
  uv venv /tmp/st5 && uv pip install --python /tmp/st5/bin/python torch --index-url https://download.pytorch.org/whl/cpu
  uv pip install --python /tmp/st5/bin/python sentence-transformers
  /tmp/st5/bin/python scripts/precompute_task_embeddings.py descriptions.json out.pt
descriptions.json: list of instructions. Output: {instruction: embedding (dim,) float32}.
Memory Anchors (LiberoContinualLearning) embeds LIBERO instructions with sentence-transformers/sentence-t5-base.
"""
import json
import sys

import torch
from sentence_transformers import SentenceTransformer

descriptions_file, out_file = sys.argv[1], sys.argv[2]
model_name = sys.argv[3] if len(sys.argv) > 3 else "sentence-transformers/sentence-t5-base"
descriptions = json.load(open(descriptions_file))
model = SentenceTransformer(model_name, device="cpu", model_kwargs={"dtype": torch.float32})
embs = model.encode(descriptions, convert_to_tensor=True).float()
torch.save({d: e.clone() for d, e in zip(descriptions, embs)}, out_file)
print(f"{len(descriptions)} embeddings of size {embs.shape[1]} from {model_name} -> {out_file}")
