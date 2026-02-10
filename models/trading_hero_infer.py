import json
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

import platform

MODEL_NAME = "fuchenru/Trading-Hero-LLM"
if platform.system() == "Windows":
    CACHE_DIR = r"Q:\models\trading-hero-llm"
else:
    CACHE_DIR = "/home/deck/models/trading-hero-llm"
LABELS = {0: "neutral", 1: "positive", 2: "negative"}


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
            text, return_tensors="pt", truncation=True, padding=True, max_length=512
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)[0]
        label_idx = torch.argmax(probs).item()
        results.append(
            {
                "text": text[:100],
                "label": LABELS[label_idx],
                "confidence": float(probs[label_idx]),
                "scores": {LABELS[i]: float(probs[i]) for i in range(3)},
            }
        )
    return results


if __name__ == "__main__":
    texts = json.loads(sys.stdin.read())
    tokenizer, model = load_model()
    results = predict(texts, tokenizer, model)
    print(json.dumps(results))
