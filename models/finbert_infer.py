#!/usr/bin/env python3
"""FinBERT inference helper. Runs under ~/models/.venv with transformers+torch."""

import json
import sys

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

import platform
import glob
import os

if platform.system() == "Windows":
    _base = r"Q:\models\finbert"
else:
    _base = "/home/deck/models/finbert"
# Find the snapshot directory (hash varies)
_snapshots = glob.glob(
    os.path.join(_base, "models--ProsusAI--finbert", "snapshots", "*")
)
MODEL_PATH = _snapshots[0] if _snapshots else "ProsusAI/finbert"
LABELS = ["positive", "negative", "neutral"]


def infer(texts):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.train(False)

    inputs = tokenizer(
        texts, padding=True, truncation=True, return_tensors="pt", max_length=512
    )
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)

    results = []
    for i in range(len(texts)):
        scores = {LABELS[j]: round(probs[i][j].item(), 4) for j in range(3)}
        best = max(scores, key=scores.get)
        results.append(
            {
                "text": texts[i],
                "sentiment": best,
                "confidence": scores[best],
                "scores": scores,
            }
        )
    return results


if __name__ == "__main__":
    texts = json.loads(sys.stdin.read())
    results = infer(texts)
    print(json.dumps(results))
