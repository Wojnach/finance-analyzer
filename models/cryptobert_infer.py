#!/usr/bin/env python3
"""CryptoBERT inference helper. Reads JSON texts from stdin, outputs JSON results."""

import json
import sys
import platform

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_NAME = "ElKulako/cryptobert"
if platform.system() == "Windows":
    CACHE_DIR = r"Q:\models\cryptobert"
else:
    CACHE_DIR = "/home/deck/models/cryptobert"

LABELS = {0: "Bearish", 1: "Neutral", 2: "Bullish"}
SENTIMENT_MAP = {"Bullish": "positive", "Neutral": "neutral", "Bearish": "negative"}


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, cache_dir=CACHE_DIR, local_files_only=True
    )
    model.eval()
    return tokenizer, model


def predict(texts, tokenizer, model):
    results = []
    for text in texts:
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, padding=True, max_length=128
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        label_idx = torch.argmax(probs).item()
        raw_label = LABELS[label_idx]
        sentiment = SENTIMENT_MAP[raw_label]
        scores = {SENTIMENT_MAP[LABELS[i]]: float(probs[i]) for i in range(len(LABELS))}
        results.append(
            {
                "text": text[:100],
                "sentiment": sentiment,
                "confidence": float(probs[label_idx]),
                "scores": scores,
            }
        )
    return results


if __name__ == "__main__":
    texts = json.loads(sys.stdin.read())
    tokenizer, model = load_model()
    results = predict(texts, tokenizer, model)
    print(json.dumps(results))
