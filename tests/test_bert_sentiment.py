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
    # Sentinel for the meta-tensor recovery path (2026-05-04). The real
    # bert_sentiment._load_model passes torch_dtype=torch.float32 when
    # retrying after detecting meta tensors. Tests don't introspect the
    # value — only that the kwarg is forwarded — so a marker object is
    # enough.
    fake_torch.float32 = "float32"

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
        # logits._data is a list of rows (list of floats). Return a batch
        # wrapper that indexes rows back to _FakeTensor instances so the
        # batched predict path can do probs[i] per input.
        rows = logits._data if hasattr(logits, "_data") else logits

        class _BatchWrapper:
            def __init__(self, rows):
                # Softmax each row independently
                self._rows = []
                for row in rows:
                    exp = [pow(2.71828, v) for v in row]
                    total = sum(exp) or 1.0
                    self._rows.append([v / total for v in exp])

            def __getitem__(self, idx):
                return _FakeTensor(self._rows[idx])

            def __len__(self):
                return len(self._rows)

        return _BatchWrapper(rows)

    fake_torch.softmax = _fake_softmax

    def _fake_argmax(tensor):
        data = tensor._data if hasattr(tensor, "_data") else list(tensor)
        best_idx = max(range(len(data)), key=lambda i: data[i])
        return _FakeScalar(best_idx)

    fake_torch.argmax = _fake_argmax

    # --- Fake transformers ---
    fake_transformers = types.ModuleType("transformers")

    class _FakeLogits:
        def __init__(self, rows):
            # rows is a list of per-text logit vectors
            self._data = rows

    class _FakeOutput:
        def __init__(self, rows):
            self.logits = _FakeLogits(rows)

    class _FakeModel:
        load_count = 0
        forward_count = 0
        # Optional hook tests can use to trigger a per-text loop from the
        # batched path: if _batched_raises is True, __call__ raises when
        # given a batch of 2+ texts.
        _batched_raises = False

        def __init__(self, logits_values):
            # logits_values is a single per-text logit vector [a, b, c];
            # the model repeats it for every text in the input batch so
            # every text in a given test run gets the same prediction.
            self._logits_values = logits_values
            self._device = "cpu"

        def train(self, flag):
            # flag=False is PyTorch's inference-mode toggle
            return self

        def parameters(self):
            # Used by the 2026-05-04 meta-tensor defensive check in
            # _load_model. Default fixture fakes have no meta tensors;
            # the new TestMetaTensorRecovery tests below override
            # from_pretrained to return models with controllable meta
            # state via _make_fake_model_with_params.
            return iter([])

        def to(self, device):
            self._device = device
            return self

        def __call__(self, **inputs):
            _FakeModel.forward_count += 1
            # Figure out the batch size from the first input tensor
            first = next(iter(inputs.values()))
            batch_size = len(first) if hasattr(first, "__len__") else 1
            if _FakeModel._batched_raises and batch_size > 1:
                raise RuntimeError("simulated batched failure")
            rows = [list(self._logits_values) for _ in range(batch_size)]
            return _FakeOutput(rows)

    class _FakeTokenizer:
        def __init__(self):
            self.calls = 0

        def __call__(self, text_or_texts, **kwargs):
            self.calls += 1
            # Accept either a single string or a list of strings. Return a
            # dict of fake tensors whose outer length matches the batch.
            if isinstance(text_or_texts, str):
                batch = [text_or_texts]
            else:
                batch = list(text_or_texts)
            # We only need the outer length for the model's batch-size
            # detection, so a list of placeholder tokens is enough.
            return {"input_ids": _FakeTensor([i for i in range(len(batch))])}

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


def test_batched_failure_falls_back_to_per_text(fake_torch_and_transformers):
    """If the batched forward pass fails, predict() should fall back to a
    per-text loop and still return one result per input headline.
    """
    from portfolio import bert_sentiment
    FakeModel = fake_torch_and_transformers["FakeModel"]

    # Trigger a load first (single text, so batched-raises doesn't fire yet)
    bert_sentiment.predict("CryptoBERT", ["warmup"])

    # Now force batched calls to raise; per-text (batch_size=1) still works.
    FakeModel._batched_raises = True
    try:
        result = bert_sentiment.predict("CryptoBERT", ["headline one", "headline two"])
        assert len(result) == 2
        for entry in result:
            assert entry["sentiment"] in {"positive", "negative", "neutral"}
            assert 0.0 <= entry["confidence"] <= 1.0
    finally:
        FakeModel._batched_raises = False


def test_per_text_fallback_handles_single_failure(fake_torch_and_transformers):
    """In the per-text fallback path, if one headline raises during forward
    pass, we still return one result per input with a neutral placeholder
    for the failed one.
    """
    from portfolio import bert_sentiment
    FakeModel = fake_torch_and_transformers["FakeModel"]

    # Warm the cache
    bert_sentiment.predict("CryptoBERT", ["warmup"])

    # Force the batched path to fail so we enter _predict_per_text
    FakeModel._batched_raises = True

    # Patch the cached model's __call__ so the first TWO calls raise (call 1
    # is the batched attempt triggered by _batched_raises, call 2 is the
    # first per-text call — we want that one to also fail so the neutral
    # placeholder branch runs). Calls 3+ succeed.
    _tok, model, _dev, _lock = bert_sentiment._models["CryptoBERT"]
    call_count = {"n": 0}
    original_call = type(model).__call__

    def flaky_call(self, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise RuntimeError(f"simulated failure on call {call_count['n']}")
        return original_call(self, **kwargs)

    type(model).__call__ = flaky_call

    try:
        result = bert_sentiment.predict("CryptoBERT", ["bad headline", "good headline"])
        assert len(result) == 2
        # First one should be the neutral placeholder from the per-text error path
        assert result[0]["sentiment"] == "neutral"
        assert result[0]["confidence"] == 0.0
        # Second one should be a real result (call 3+ succeeds)
        assert result[1]["sentiment"] in {"positive", "negative", "neutral"}
        assert result[1]["confidence"] > 0.0
    finally:
        type(model).__call__ = original_call
        FakeModel._batched_raises = False


# ---------------------------------------------------------------------------
# Meta-tensor recovery (fix/bert-meta-tensor, 2026-05-04)
# ---------------------------------------------------------------------------
#
# Race between Chronos's CUDA load and concurrent BERT loads (commit
# 789cc91c, 2026-05-03 21:08) can leave some FinBERT weights on the
# `meta` device when accelerate's lazy init interleaves with CUDA init
# on another thread. _load_model now detects this at load time and
# retries once with explicit eager-init kwargs. Tests below stub
# from_pretrained to control the meta-tensor state per call so the
# defensive path is exercised without needing the real race.


class _FakeParam:
    """Minimal stand-in for a torch.nn.Parameter used by the meta-tensor
    detection code (`p.is_meta` attribute access only).
    """

    def __init__(self, is_meta: bool):
        self.is_meta = is_meta


def _make_fake_model_with_params(meta_count: int, total: int = 5):
    """Build a fake model whose `.parameters()` yields `total` _FakeParam
    instances, the first `meta_count` of which have is_meta=True. Returns
    a _FakeModel-like object with .parameters() and .train() defined.
    """
    class _FakeModelWithParams:
        load_count_local = 0

        def __init__(self):
            self._params = [
                _FakeParam(is_meta=(i < meta_count))
                for i in range(total)
            ]
            self._device = "cpu"

        def parameters(self):
            return iter(self._params)

        def train(self, _flag):
            return self

        def to(self, device):
            self._device = device
            return self

        def __call__(self, **inputs):
            # Minimal forward — not used by these tests.
            raise RuntimeError("meta-tensor tests don't exercise forward pass")

    return _FakeModelWithParams()


class TestMetaTensorRecovery:
    """Defensive load-time detection + retry for meta-tensor corruption.
    Race-induced bug in 2026-05-03; predict-time symptom is a per-text
    "Tensor on device meta is not on the expected device cpu!" warning
    on every headline. The fix detects at load time and retries with
    eager-init kwargs.
    """

    def test_clean_load_does_not_retry(self, fake_torch_and_transformers):
        """Happy path: from_pretrained returns a model with no meta
        params; the defensive check is a no-op and from_pretrained is
        called exactly once.
        """
        from portfolio import bert_sentiment
        FakeAutoModel = fake_torch_and_transformers["FakeAutoModel"]

        call_log: list[dict] = []

        def fake_from_pretrained(*args, **kwargs):
            call_log.append({"args": args, "kwargs": dict(kwargs)})
            return _make_fake_model_with_params(meta_count=0)

        FakeAutoModel.from_pretrained = staticmethod(fake_from_pretrained)
        bert_sentiment._reset_for_tests()

        bert_sentiment.predict("CryptoBERT", ["warmup"])

        assert len(call_log) == 1, f"expected 1 from_pretrained call, got {len(call_log)}"
        # Eager-init kwargs should NOT appear in the (single, clean) call.
        assert "torch_dtype" not in call_log[0]["kwargs"]
        assert "low_cpu_mem_usage" not in call_log[0]["kwargs"]

    def test_load_with_meta_tensors_retries_with_eager_init(
        self, fake_torch_and_transformers, caplog,
    ):
        """First from_pretrained returns a model with one meta param;
        the retry call with eager-init kwargs returns a clean model.
        Assert: two from_pretrained calls, second one carries
        torch_dtype + low_cpu_mem_usage=False, and a warning was logged.
        """
        from portfolio import bert_sentiment
        FakeAutoModel = fake_torch_and_transformers["FakeAutoModel"]

        call_log: list[dict] = []

        def fake_from_pretrained(*args, **kwargs):
            call_idx = len(call_log)
            call_log.append({"args": args, "kwargs": dict(kwargs)})
            if call_idx == 0:
                # First call: simulate the race — return a model with a
                # meta param (1 of 5).
                return _make_fake_model_with_params(meta_count=1)
            # Retry call: clean.
            return _make_fake_model_with_params(meta_count=0)

        FakeAutoModel.from_pretrained = staticmethod(fake_from_pretrained)
        bert_sentiment._reset_for_tests()

        with caplog.at_level("WARNING", logger="portfolio.bert_sentiment"):
            bert_sentiment.predict("CryptoBERT", ["warmup"])

        assert len(call_log) == 2, (
            f"expected 1 initial + 1 retry, got {len(call_log)}: {call_log}"
        )
        # Retry call must include eager-init kwargs
        retry_kwargs = call_log[1]["kwargs"]
        assert retry_kwargs.get("low_cpu_mem_usage") is False, retry_kwargs
        assert "torch_dtype" in retry_kwargs, retry_kwargs
        # Warning was logged with model name + meta-tensor mention
        warnings = [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]
        assert any(
            "CryptoBERT" in m and "meta tensors" in m for m in warnings
        ), warnings

    def test_load_with_persistent_meta_tensors_raises(
        self, fake_torch_and_transformers,
    ):
        """Both calls return models with meta tensors → RuntimeError.
        The caller's _get_model() doesn't catch this, so the model is
        NOT cached in _models; subsequent predict calls retry from
        scratch instead of compounding a corrupted load.
        """
        from portfolio import bert_sentiment
        FakeAutoModel = fake_torch_and_transformers["FakeAutoModel"]

        def fake_from_pretrained(*args, **kwargs):
            return _make_fake_model_with_params(meta_count=2)

        FakeAutoModel.from_pretrained = staticmethod(fake_from_pretrained)
        bert_sentiment._reset_for_tests()

        with pytest.raises(RuntimeError) as excinfo:
            bert_sentiment.predict("CryptoBERT", ["warmup"])
        # Error message should be informative for triage
        assert "still has meta tensors after retry" in str(excinfo.value)
        assert "CryptoBERT" in str(excinfo.value)
        # Bad model must NOT have been cached
        assert "CryptoBERT" not in bert_sentiment._models

    def test_finbert_retry_uses_snapshot_path(
        self, fake_torch_and_transformers, monkeypatch, tmp_path,
    ):
        """When FinBERT loads from a resolved snapshot, the retry must
        pass the same snapshot path (not the hub name) as the positional
        load_path so the second from_pretrained still finds the local
        weights.
        """
        from portfolio import bert_sentiment
        FakeAutoModel = fake_torch_and_transformers["FakeAutoModel"]

        # Fake the snapshot resolver so we don't depend on real files
        fake_snap = str(tmp_path / "fake-finbert-snapshot")
        monkeypatch.setattr(
            bert_sentiment, "_resolve_finbert_snapshot",
            lambda *_a, **_k: fake_snap,
        )

        call_log: list = []

        def fake_from_pretrained(*args, **kwargs):
            call_log.append({"args": args, "kwargs": dict(kwargs)})
            # Simulate: first call leaves meta tensors, retry is clean.
            if len(call_log) == 1:
                return _make_fake_model_with_params(meta_count=1)
            return _make_fake_model_with_params(meta_count=0)

        FakeAutoModel.from_pretrained = staticmethod(fake_from_pretrained)
        bert_sentiment._reset_for_tests()

        bert_sentiment.predict("FinBERT", ["warmup"])

        assert len(call_log) == 2
        # Both calls must use the snapshot path as positional arg, not the
        # hub name (the hub path doesn't have local weights and would 404).
        assert call_log[0]["args"][0] == fake_snap
        assert call_log[1]["args"][0] == fake_snap
        # Retry must have the eager kwargs
        assert call_log[1]["kwargs"].get("low_cpu_mem_usage") is False


def test_accelerate_version_returns_string_or_none():
    """Smoke test for the diagnostic helper used in the meta-tensor
    RuntimeError message. Returns either a version string or None
    depending on whether accelerate is installed in the test venv.
    """
    from portfolio.bert_sentiment import _accelerate_version
    result = _accelerate_version()
    assert result is None or isinstance(result, str)
