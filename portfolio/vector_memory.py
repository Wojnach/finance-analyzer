"""ChromaDB-backed semantic memory for journal entries.

Uses ChromaDB's built-in all-MiniLM-L6-v2 embeddings (via onnxruntime,
no sentence-transformers needed). Lazy-init singleton. Embeds journal
entries on read, queries by market state similarity.

Config:
    "vector_memory": {
        "enabled": false,
        "collection": "trade_journal",
        "top_k": 5
    }

Requires: pip install chromadb
Defaults to disabled — graceful fallback if chromadb is not installed.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger("portfolio.vector_memory")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CHROMADB_DIR = DATA_DIR / "chromadb"
JOURNAL_FILE = DATA_DIR / "layer2_journal.jsonl"

# Singleton
_collection = None
_client = None


def _get_collection(collection_name="trade_journal"):
    """Lazy-init ChromaDB client and return the collection.

    Raises ImportError if chromadb is not installed.
    """
    global _client, _collection

    if _collection is not None:
        return _collection

    import chromadb

    CHROMADB_DIR.mkdir(parents=True, exist_ok=True)
    _client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
    _collection = _client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    logger.info("ChromaDB collection '%s' ready (%d entries)",
                collection_name, _collection.count())
    return _collection


def entry_to_text(entry):
    """Convert a journal entry to a searchable text string.

    Captures regime, decisions, ticker outlooks, theses, debate fields,
    watchlist, and reflection — everything Layer 2 would want to match on.
    """
    parts = []

    regime = entry.get("regime", "")
    if regime:
        parts.append(f"regime: {regime}")

    trigger = entry.get("trigger", "")
    if trigger:
        parts.append(f"trigger: {trigger}")

    decisions = entry.get("decisions", {})
    for strat in ("patient", "bold"):
        d = decisions.get(strat, {})
        action = d.get("action", "HOLD")
        reasoning = d.get("reasoning", "")
        parts.append(f"{strat}: {action} — {reasoning}")

    tickers = entry.get("tickers", {})
    for ticker, info in tickers.items():
        outlook = info.get("outlook", "neutral")
        thesis = info.get("thesis", "")
        conviction = info.get("conviction", 0)
        line = f"{ticker}: {outlook}"
        if conviction:
            line += f" ({conviction:.0%})"
        if thesis:
            line += f" — {thesis}"
        parts.append(line)

        # Debate fields
        debate = info.get("debate")
        if debate and isinstance(debate, dict):
            for field in ("bull", "bear", "synthesis"):
                text = debate.get(field, "")
                if text:
                    parts.append(f"  {field}: {text}")

    reflection = entry.get("reflection", "")
    if reflection:
        parts.append(f"reflection: {reflection}")

    watchlist = entry.get("watchlist", [])
    if watchlist:
        parts.append("watchlist: " + "; ".join(watchlist))

    return "\n".join(parts)


def _entry_id(entry):
    """Generate a stable ID for a journal entry based on its timestamp."""
    ts = entry.get("ts", "")
    return hashlib.md5(ts.encode()).hexdigest()


def embed_entries(entries, collection_name="trade_journal"):
    """Embed journal entries that aren't yet in ChromaDB.

    Args:
        entries: list of journal entry dicts.
        collection_name: ChromaDB collection name.

    Returns:
        int: number of newly embedded entries.
    """
    collection = _get_collection(collection_name)

    # Get existing IDs
    existing = set()
    if collection.count() > 0:
        result = collection.get()
        existing = set(result["ids"])

    new_docs = []
    new_ids = []
    new_metas = []

    for entry in entries:
        eid = _entry_id(entry)
        if eid in existing:
            continue
        text = entry_to_text(entry)
        if not text.strip():
            continue
        new_docs.append(text)
        new_ids.append(eid)
        new_metas.append({
            "ts": entry.get("ts", ""),
            "regime": entry.get("regime", ""),
        })

    if new_docs:
        collection.add(documents=new_docs, ids=new_ids, metadatas=new_metas)
        logger.info("Embedded %d new journal entries", len(new_docs))

    return len(new_docs)


def query_similar(query_text, top_k=5, collection_name="trade_journal"):
    """Query ChromaDB for journal entries similar to query_text.

    Args:
        query_text: text describing current market state.
        top_k: number of results.
        collection_name: ChromaDB collection name.

    Returns:
        list of dicts with keys: text, ts, regime, distance.
    """
    collection = _get_collection(collection_name)
    if collection.count() == 0:
        return []

    results = collection.query(
        query_texts=[query_text],
        n_results=min(top_k, collection.count()),
    )

    entries = []
    for i, doc in enumerate(results["documents"][0]):
        meta = results["metadatas"][0][i] if results["metadatas"] else {}
        dist = results["distances"][0][i] if results["distances"] else 0
        entries.append({
            "text": doc,
            "ts": meta.get("ts", ""),
            "regime": meta.get("regime", ""),
            "distance": dist,
        })

    return entries


def build_query_text(market_state):
    """Convert current market state into a query text for semantic search.

    Args:
        market_state: dict with signals, held_tickers, regime, prices.

    Returns:
        str: query text.
    """
    parts = []

    regime = market_state.get("regime", "")
    if regime:
        parts.append(f"regime: {regime}")

    held = market_state.get("held_tickers", [])
    if held:
        parts.append(f"holding: {', '.join(held)}")

    signals = market_state.get("signals", {})
    for ticker, sig in signals.items():
        if not isinstance(sig, dict):
            continue
        action = sig.get("action", "HOLD")
        if action != "HOLD":
            conf = sig.get("confidence", 0)
            parts.append(f"{ticker}: {action} ({conf:.0%})")

    return "\n".join(parts) if parts else ""


def get_semantic_context(market_state, bm25_timestamps=None,
                         top_k=5, collection_name="trade_journal"):
    """Full semantic retrieval pipeline: embed new entries, query, de-dup.

    Args:
        market_state: dict with signals, held_tickers, regime, prices.
        bm25_timestamps: set of timestamp strings already returned by BM25
            (for de-duplication).
        top_k: number of semantic results to return.
        collection_name: ChromaDB collection name.

    Returns:
        list of dicts with text, ts, regime, distance.
        Returns empty list on any error.
    """
    try:
        # Embed any new entries
        entries = _load_journal_entries()
        if entries:
            embed_entries(entries, collection_name)

        # Build query
        query = build_query_text(market_state)
        if not query:
            return []

        # Query
        results = query_similar(query, top_k=top_k * 2, collection_name=collection_name)

        # De-duplicate against BM25
        bm25_ts = set(bm25_timestamps or [])
        deduped = [r for r in results if r.get("ts", "") not in bm25_ts]

        return deduped[:top_k]
    except ImportError:
        logger.debug("chromadb not installed, skipping vector memory")
        return []
    except Exception as e:
        logger.warning("vector memory error: %s", e)
        return []


def _load_journal_entries():
    """Load all journal entries from JSONL."""
    if not JOURNAL_FILE.exists():
        return []
    entries = []
    with open(JOURNAL_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def reset():
    """Reset the singleton (for testing)."""
    global _client, _collection
    _client = None
    _collection = None
