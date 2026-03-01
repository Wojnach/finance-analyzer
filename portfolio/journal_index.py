"""Smart journal retrieval using BM25 relevance ranking.

Replaces chronological "last N entries" with keyword-relevance-ranked retrieval
so Layer 2 sees the most contextually relevant prior analyses, not just the
most recent.
"""

import json
import logging
import math
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger("portfolio.journal_index")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"


# ---------------------------------------------------------------------------
# Minimal BM25 implementation (no external dependencies)
# ---------------------------------------------------------------------------

class BM25:
    """Okapi BM25 ranking function for document retrieval.

    BM25 scores documents by term frequency with diminishing returns
    (saturation) and inverse document frequency. No external deps needed.
    """

    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b
        self.doc_count = 0
        self.avg_doc_len = 0
        self.doc_lens = []
        self.term_doc_freq = Counter()  # term -> number of docs containing it
        self.doc_term_freqs = []  # list of Counter per document

    def fit(self, documents):
        """Index a list of token lists.

        Args:
            documents: list of list[str] (tokenized documents).
        """
        self.doc_count = len(documents)
        self.doc_lens = [len(d) for d in documents]
        self.avg_doc_len = sum(self.doc_lens) / self.doc_count if self.doc_count else 1
        self.term_doc_freq = Counter()
        self.doc_term_freqs = []

        for doc in documents:
            tf = Counter(doc)
            self.doc_term_freqs.append(tf)
            for term in set(doc):
                self.term_doc_freq[term] += 1

    def _idf(self, term):
        """Compute inverse document frequency for a term."""
        df = self.term_doc_freq.get(term, 0)
        if df == 0:
            return 0
        return math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1)

    def score(self, query_tokens):
        """Score all documents against a query.

        Args:
            query_tokens: list[str] of query terms.

        Returns:
            list[float] of scores (one per document, same order as fit()).
        """
        scores = []
        for i in range(self.doc_count):
            s = 0
            tf_doc = self.doc_term_freqs[i]
            doc_len = self.doc_lens[i]
            for term in query_tokens:
                idf = self._idf(term)
                tf = tf_doc.get(term, 0)
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_doc_len)
                s += idf * numerator / denominator if denominator > 0 else 0
            scores.append(s)
        return scores

    def top_k(self, query_tokens, k=8):
        """Return top-k document indices by BM25 score.

        Args:
            query_tokens: list[str] of query terms.
            k: number of results to return.

        Returns:
            list of (index, score) tuples, sorted by score descending.
        """
        scores = self.score(query_tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(i, s) for i, s in indexed[:k] if s > 0]


# ---------------------------------------------------------------------------
# Journal Index
# ---------------------------------------------------------------------------

# Price level buckets for matching "similar price environment"
_PRICE_BUCKETS = {
    "BTC-USD": [20000, 30000, 40000, 50000, 60000, 70000, 80000, 100000],
    "ETH-USD": [1000, 1500, 2000, 2500, 3000, 4000, 5000],
    "XAU-USD": [1800, 1900, 2000, 2100, 2200],
    "XAG-USD": [25, 30, 35, 50, 75, 100, 120],
}


def _price_bucket(ticker, price):
    """Convert a price to a searchable bucket token."""
    buckets = _PRICE_BUCKETS.get(ticker)
    if not buckets or price is None:
        return None
    for b in buckets:
        if price < b:
            return f"{ticker}_below_{b}"
    return f"{ticker}_above_{buckets[-1]}"


def _tokenize_entry(entry):
    """Extract searchable tokens from a journal entry.

    Tokens include: tickers mentioned, regime, outlook keywords, thesis words,
    watchlist items, price level buckets, decision actions.
    """
    tokens = []

    # Regime
    regime = entry.get("regime", "")
    if regime:
        tokens.append(f"regime_{regime}")

    # Trigger
    trigger = entry.get("trigger", "")
    if trigger:
        tokens.append(f"trigger_{trigger}")

    # Decisions
    decisions = entry.get("decisions", {})
    for strat in ("patient", "bold"):
        d = decisions.get(strat, {})
        action = d.get("action", "HOLD")
        if action != "HOLD":
            tokens.append(f"{strat}_{action.lower()}")
        reasoning = d.get("reasoning", "")
        if reasoning:
            tokens.extend(_clean_words(reasoning))

    # Tickers and their outlooks
    tickers = entry.get("tickers", {})
    for ticker, info in tickers.items():
        tokens.append(ticker.lower())
        outlook = info.get("outlook", "neutral")
        if outlook != "neutral":
            tokens.append(f"{ticker.lower()}_{outlook}")
        thesis = info.get("thesis", "")
        if thesis:
            tokens.extend(_clean_words(thesis))
        conviction = info.get("conviction", 0)
        if conviction >= 0.7:
            tokens.append(f"{ticker.lower()}_high_conviction")

        # Debate fields (bull/bear/synthesis)
        debate = info.get("debate")
        if debate and isinstance(debate, dict):
            for field in ("bull", "bear", "synthesis"):
                text = debate.get(field, "")
                if text:
                    tokens.extend(_clean_words(text))

    # Price buckets
    prices = entry.get("prices", {})
    for ticker, price in prices.items():
        bucket = _price_bucket(ticker, price)
        if bucket:
            tokens.append(bucket.lower())

    # Watchlist
    for item in entry.get("watchlist", []):
        tokens.extend(_clean_words(item))

    # Reflection
    reflection = entry.get("reflection", "")
    if reflection:
        tokens.extend(_clean_words(reflection))

    return tokens


def _clean_words(text):
    """Split text into lowercase word tokens, filtering noise."""
    if not text:
        return []
    words = re.findall(r"[a-zA-Z0-9_-]+", text.lower())
    # Filter very short words and common stop words
    stop = {"the", "a", "an", "is", "was", "are", "be", "to", "of", "and",
            "in", "for", "on", "at", "by", "or", "no", "not", "but", "with"}
    return [w for w in words if len(w) > 1 and w not in stop]


def _compute_importance(entry, now=None):
    """Compute importance score for a journal entry.

    Factors:
    - Time decay: more recent entries score higher
    - Trade action: entries with actual trades are more important
    - Conviction: high-conviction entries matter more
    - Reflection: entries with reflections carry lessons

    Returns:
        float: importance score (0.0 to 1.0)
    """
    if now is None:
        now = datetime.now(timezone.utc)

    score = 0.5  # base

    # Time decay: entries from last 2h get full score, exponential decay after
    try:
        ts = datetime.fromisoformat(entry.get("ts", ""))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        age_hours = (now - ts).total_seconds() / 3600
        # Half-life of 4 hours
        decay = 0.5 ** (age_hours / 4)
        score += 0.3 * decay
    except (ValueError, TypeError):
        pass

    # Trade action boost
    decisions = entry.get("decisions", {})
    for strat in ("patient", "bold"):
        action = decisions.get(strat, {}).get("action", "HOLD")
        if action != "HOLD":
            score += 0.1

    # High conviction boost
    tickers = entry.get("tickers", {})
    max_conviction = max(
        (info.get("conviction", 0) for info in tickers.values()),
        default=0,
    )
    if max_conviction >= 0.7:
        score += 0.1

    # Reflection boost (contains lessons)
    if entry.get("reflection"):
        score += 0.05

    return min(score, 1.0)


class JournalIndex:
    """BM25-indexed journal for relevance-ranked retrieval."""

    def __init__(self):
        self.entries = []
        self.bm25 = BM25()
        self.importances = []

    def build(self, entries):
        """Index a list of journal entries.

        Args:
            entries: list of journal entry dicts.
        """
        self.entries = entries
        documents = [_tokenize_entry(e) for e in entries]
        self.bm25.fit(documents)
        now = datetime.now(timezone.utc)
        self.importances = [_compute_importance(e, now) for e in entries]

    def query(self, market_state, k=8):
        """Retrieve the most relevant journal entries for current market state.

        Args:
            market_state: dict with keys like:
                - held_tickers: list[str]
                - regime: str
                - prices: dict[str, float]
                - signals: dict (ticker -> signal data)
            k: number of entries to return.

        Returns:
            list of journal entry dicts, ranked by relevance.
        """
        if not self.entries:
            return []

        query_tokens = _build_query_tokens(market_state)
        if not query_tokens:
            # Fallback: return most recent
            return self.entries[-k:]

        results = self.bm25.top_k(query_tokens, k=k * 2)  # Get more, then filter

        # Re-rank by BM25 score * importance
        ranked = []
        for idx, bm25_score in results:
            importance = self.importances[idx] if idx < len(self.importances) else 0.5
            combined = bm25_score * importance
            ranked.append((idx, combined))

        ranked.sort(key=lambda x: x[1], reverse=True)

        # Return top-k entries
        return [self.entries[idx] for idx, _ in ranked[:k]]


def _build_query_tokens(market_state):
    """Convert current market state into query tokens for BM25."""
    tokens = []

    regime = market_state.get("regime", "")
    if regime:
        tokens.append(f"regime_{regime}")

    for ticker in market_state.get("held_tickers", []):
        tokens.append(ticker.lower())

    prices = market_state.get("prices", {})
    for ticker, price in prices.items():
        bucket = _price_bucket(ticker, price)
        if bucket:
            tokens.append(bucket.lower())

    # Add tickers with non-HOLD signals
    signals = market_state.get("signals", {})
    for ticker, sig in signals.items():
        action = sig.get("action", "HOLD") if isinstance(sig, dict) else "HOLD"
        if action != "HOLD":
            tokens.append(ticker.lower())
            tokens.append(f"{ticker.lower()}_{action.lower()}")

    return tokens


# ---------------------------------------------------------------------------
# Top-level retrieval function
# ---------------------------------------------------------------------------

def retrieve_relevant_entries(signals, held_tickers, regime, prices, k=8):
    """Retrieve the most relevant journal entries for the current market state.

    This is the main entry point called by journal.py.

    Args:
        signals: dict of ticker -> signal data.
        held_tickers: list of currently held ticker symbols.
        regime: str (current market regime).
        prices: dict of ticker -> current USD price.
        k: number of entries to return.

    Returns:
        list of journal entry dicts, ranked by relevance.
        Falls back to chronological (most recent) on any error.
    """
    if not JOURNAL_FILE.exists():
        return []

    # Load all entries
    entries = []
    try:
        with open(JOURNAL_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except (IOError, OSError):
        return []

    if not entries:
        return []

    # Build index and query
    index = JournalIndex()
    index.build(entries)

    market_state = {
        "held_tickers": held_tickers or [],
        "regime": regime or "",
        "prices": prices or {},
        "signals": signals or {},
    }

    return index.query(market_state, k=k)
