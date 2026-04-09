"""Unit tests for portfolio.bert_sentiment — in-process BERT cache.

Strategy: stub out torch + transformers so the tests don't need the real
models loaded. This keeps them fast and xdist-safe. A separate integration
test (not in this file) can exercise the real model loads if needed.
"""

from __future__ import annotations

import sys
import threading
import types

import pytest


@pytest.fixture
def fake_torch_and_transformers(monkeypatch):
    """Install stub torch and transformers modules that return predictable
    fake outputs so the BERT cache code can be tested without real models.

    Yields a dict with handles to the fakes so tests can override behaviour.
    """
    # --- Fake torch ---
    fake_torch = types.ModuleType("torch")
    fake_torch._cuda_available = True  # test can flip this

    class _FakeNoGradCtx:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    fake_torch.no_grad = lambda: _FakeNoGradCtx()

    class _FakeCuda:
        @staticmethod
        def is_available():
            return fake_torch._cuda_available

    fake_torch.cuda = _FakeCuda()

    class _FakeTensor:
        """Minimal fake of torch.Tensor: holds a Python list of floats.
        Supports indexing, .item(), softmax, argmax — enough for our code.
        """

        def __init__(self, data):
            self._data = list(data)

        def __getitem__(self, idx):
            if isinstance(idx, int):
                return _FakeScalar(self._data[idx])
            # Slice / multi-index — return another tensor
            return _FakeTensor(self._data[idx])

        def __len__(self):
            return len(self._data)

        def to(self, device):
            # no-op, just return self
            return self

    class _FakeScalar:
        def __init__(self, value):
            self._v = float(value)

        def item(self):
            return self._v

    def _fake_softmax(logits, dim):
        # logits is a 2D list [[a, b, c]]; return a tensor-like for [[softmaxed]]
        data = logits._data[0] if hasattr(logits, "_data") else logits[0]
        exp = [pow(2.71828, v) for v in data]
        total = sum(exp) or 1.0
        normalized = [v / total for v in exp]
        # Return a wrapper that indexes to a _FakeTensor of the row
        class _Wrapper:
            def __init__(self, rows):
                self._rows = rows

            def __getitem__(self, idx):
                return _FakeTensor(self._rows[idx])

        return _Wrapper([normalized])

    fake_torch.softmax = _fake_softmax

    def _fake_argmax(tensor):
        data = tensor._data if hasattr(tensor, "_data") else list(tensor)
        best_idx = max(range(len(data)), key=lambda i: data[i])
        return _FakeScalar(best_idx)

    fake_torch.argmax = _fake_argmax

    # --- Fake transformers ---
    fake_transformers = types.ModuleType("transformers")

    class _FakeLogits:
        def __init__(self, values):
            self._data = [values]

    class _FakeOutput:
        def __init__(self, logits_values):
            self.logits = _FakeLogits(logits_values)

    class _FakeModel:
        load_count = 0
        forward_count = 0

        def __init__(self, logits_values):
            self._logits_values = logits_values
            self._device = "cpu"

        def train(self, flag):
            # flag=False is PyTorch's inference-mode toggle
            return self

        def to(self, device):
            self._device = device
            return self

        def __call__(self, **inputs):
            _FakeModel.forward_count += 1
            return _FakeOutput(self._logits_values)

    class _FakeTokenizer:
        def __init__(self):
            self.calls = 0

        def __call__(self, text, **kwargs):
            self.calls += 1
            # Return a dict of fake tensors; keys don't matter for the fake
            return {"input_ids": _FakeTensor([1, 2, 3])}

    class _FakeAutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            return _FakeTokenizer()

    class _FakeAutoModel:
        # Overrideable per test to control which label wins
        default_logits = [0.1, 0.2, 0.9]  # last class wins by default

        @staticmethod
        def from_pretrained(*args, **kwargs):
            _FakeModel.load_count += 1
            return _FakeModel(_FakeAutoModel.default_logits)

    fake_transformers.AutoTokenizer = _FakeAutoTokenizer
    fake_transformers.AutoModelForSequenceClassification = _FakeAutoModel

    # Install into sys.modules BEFORE importing the module under test
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)

    # Reset the module-level cache for every test
    from portfolio import bert_sentiment
    bert_sentiment._reset_for_tests()
    # Also reset the fake counters
    _FakeModel.load_count = 0
    _FakeModel.forward_count = 0

    yield {
        "torch": fake_torch,
        "transformers": fake_transformers,
        "FakeModel": _FakeModel,
        "FakeAutoModel": _FakeAutoModel,
    }

    # Teardown — clear cache again so the next test starts fresh
    bert_sentiment._reset_for_tests()


# --- Tests -----------------------------------------------------------------


def test_available_models(fake_torch_and_transformers):
    from portfolio import bert_sentiment
    names = bert_sentiment.available_models()
    assert "CryptoBERT" in names
    assert "Trading-Hero-LLM" in names
    assert "FinBERT" in names


def test_predict_empty_returns_empty(fake_torch_and_transformers):
    from portfolio import bert_sentiment
    assert bert_sentiment.predict("CryptoBERT", []) == []


def test_predict_shape_matches_subprocess(fake_torch_and_transformers):
    from portfolio import bert_sentiment

    result = bert_sentiment.predict("CryptoBERT", ["Bitcoin rallies on ETF inflows"])
    assert isinstance(result, list)
    assert len(result) == 1
    entry = result[0]
    assert set(entry.keys()) == {"text", "sentiment", "confidence", "scores"}
    assert isinstance(entry["text"], str)
    assert entry["sentiment"] in {"positive", "negative", "neutral"}
    assert 0.0 <= entry["confidence"] <= 1.0
    assert set(entry["scores"].keys()) == {"positive", "negative", "neutral"}


def test_lazy_load_happens_once(fake_torch_and_transformers):
    from portfolio import bert_sentiment

    FakeModel = fake_torch_and_transformers["FakeModel"]
    assert not bert_sentiment.is_loaded("CryptoBERT")
    assert FakeModel.load_count == 0

    # First call triggers load
    bert_sentiment.predict("CryptoBERT", ["test"])
    assert bert_sentiment.is_loaded("CryptoBERT")
    assert FakeModel.load_count == 1

    # Second call reuses the cache
    bert_sentiment.predict("CryptoBERT", ["another test"])
    assert FakeModel.load_count == 1  # still 1, no reload


def test_different_models_load_independently(fake_torch_and_transformers):
    from portfolio import bert_sentiment

    FakeModel = fake_torch_and_transformers["FakeModel"]
    bert_sentiment.predict("CryptoBERT", ["crypto"])
    bert_sentiment.predict("FinBERT", ["stock"])
    assert FakeModel.load_count == 2
    assert bert_sentiment.is_loaded("CryptoBERT")
    assert bert_sentiment.is_loaded("FinBERT")
    assert not bert_sentiment.is_loaded("Trading-Hero-LLM")


def test_default_stays_on_cpu_even_with_cuda(fake_torch_and_transformers, monkeypatch):
    """2026-04-09 hotfix: default behaviour keeps BERT on CPU to avoid VRAM
    contention with llama-server. GPU path requires BERT_SENTIMENT_USE_GPU=1.
    """
    from portfolio import bert_sentiment
    fake_torch_and_transformers["torch"]._cuda_available = True
    monkeypatch.delenv("BERT_SENTIMENT_USE_GPU", raising=False)

    bert_sentiment.predict("CryptoBERT", ["test"])
    entry = bert_sentiment._models["CryptoBERT"]
    device = entry[2]
    assert device == "cpu"


def test_env_var_opt_in_moves_to_gpu(fake_torch_and_transformers, monkeypatch):
    """Setting BERT_SENTIMENT_USE_GPU=1 should re-enable GPU path when CUDA
    is available. Used for manual opt-in when VRAM pressure is known to be
    safe (e.g. if Chronos or llama-server are retired).
    """
    from portfolio import bert_sentiment
    fake_torch_and_transformers["torch"]._cuda_available = True
    monkeypatch.setenv("BERT_SENTIMENT_USE_GPU", "1")

    bert_sentiment.predict("CryptoBERT", ["test"])
    entry = bert_sentiment._models["CryptoBERT"]
    device = entry[2]
    assert device == "cuda"


def test_cuda_unavailable_falls_back_to_cpu(fake_torch_and_transformers, monkeypatch):
    """Even with BERT_SENTIMENT_USE_GPU=1, if CUDA is not available we must
    still return "cpu" and not crash.
    """
    from portfolio import bert_sentiment
    fake_torch_and_transformers["torch"]._cuda_available = False
    monkeypatch.setenv("BERT_SENTIMENT_USE_GPU", "1")

    bert_sentiment.predict("CryptoBERT", ["test"])
    entry = bert_sentiment._models["CryptoBERT"]
    device = entry[2]
    assert device == "cpu"


def test_cryptobert_label_mapping(fake_torch_and_transformers):
    """CryptoBERT native label 2 is 'Bullish' which must map to 'positive'."""
    from portfolio import bert_sentiment

    # Force the fake to return logits where index 2 wins (Bullish -> positive)
    fake_torch_and_transformers["FakeAutoModel"].default_logits = [0.1, 0.1, 0.9]

    result = bert_sentiment.predict("CryptoBERT", ["Big crypto rally"])
    assert result[0]["sentiment"] == "positive"

    # Now index 0 (Bearish -> negative)
    bert_sentiment._reset_for_tests()
    fake_torch_and_transformers["FakeAutoModel"].default_logits = [0.9, 0.1, 0.1]
    result = bert_sentiment.predict("CryptoBERT", ["Crypto crash"])
    assert result[0]["sentiment"] == "negative"


def test_trading_hero_label_mapping(fake_torch_and_transformers):
    """Trading-Hero-LLM: {0: neutral, 1: positive, 2: negative}."""
    from portfolio import bert_sentiment

    fake_torch_and_transformers["FakeAutoModel"].default_logits = [0.1, 0.9, 0.1]
    result = bert_sentiment.predict("Trading-Hero-LLM", ["Apple beats earnings"])
    assert result[0]["sentiment"] == "positive"

    bert_sentiment._reset_for_tests()
    fake_torch_and_transformers["FakeAutoModel"].default_logits = [0.1, 0.1, 0.9]
    result = bert_sentiment.predict("Trading-Hero-LLM", ["Stock plunges"])
    assert result[0]["sentiment"] == "negative"


def test_finbert_label_mapping(fake_torch_and_transformers):
    """FinBERT: {0: positive, 1: negative, 2: neutral}."""
    from portfolio import bert_sentiment

    fake_torch_and_transformers["FakeAutoModel"].default_logits = [0.9, 0.1, 0.1]
    result = bert_sentiment.predict("FinBERT", ["Good earnings"])
    assert result[0]["sentiment"] == "positive"

    bert_sentiment._reset_for_tests()
    fake_torch_and_transformers["FakeAutoModel"].default_logits = [0.1, 0.9, 0.1]
    result = bert_sentiment.predict("FinBERT", ["Guidance cut"])
    assert result[0]["sentiment"] == "negative"


def test_unknown_model_raises(fake_torch_and_transformers):
    from portfolio import bert_sentiment
    with pytest.raises(KeyError, match="Unknown BERT model"):
        bert_sentiment.predict("NotARealModel", ["test"])


def test_concurrent_predict_same_model(fake_torch_and_transformers):
    """Smoke test: two threads calling predict() on the same model should not
    corrupt each other. They serialize on the per-model lock but both succeed.
    """
    from portfolio import bert_sentiment

    results = {"a": None, "b": None, "error": None}

    def worker(key, texts):
        try:
            results[key] = bert_sentiment.predict("CryptoBERT", texts)
        except Exception as e:
            results["error"] = e

    t1 = threading.Thread(target=worker, args=("a", ["thread 1 text"]))
    t2 = threading.Thread(target=worker, args=("b", ["thread 2 text"]))
    t1.start()
    t2.start()
    t1.join(timeout=10)
    t2.join(timeout=10)

    assert results["error"] is None
    assert results["a"] is not None
    assert results["b"] is not None
    assert len(results["a"]) == 1
    assert len(results["b"]) == 1


def test_per_headline_failure_emits_neutral_placeholder(fake_torch_and_transformers):
    """If a single headline raises during forward pass, we should still return
    a full result list with a zero-confidence neutral entry for the failed one.
    """
    from portfolio import bert_sentiment

    # Trigger a load first
    bert_sentiment.predict("CryptoBERT", ["warmup"])

    # Now monkeypatch the cached model's __call__ to raise
    _tok, model, _dev, _lock = bert_sentiment._models["CryptoBERT"]

    call_count = {"n": 0}
    original_call = model.__call__

    def flaky_call(**kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("simulated per-headline failure")
        return original_call(**kwargs)

    # Monkeypatch the instance
    model.__class__.__call__ = lambda self, **kw: flaky_call(**kw)

    try:
        result = bert_sentiment.predict("CryptoBERT", ["bad headline", "good headline"])
        assert len(result) == 2
        # First one should be the neutral placeholder from the error path
        assert result[0]["sentiment"] == "neutral"
        assert result[0]["confidence"] == 0.0
        # Second one should be a real result
        assert result[1]["sentiment"] in {"positive", "negative", "neutral"}
    finally:
        # Restore so other tests don't break
        model.__class__.__call__ = lambda self, **kw: original_call(**kw)
