"""Prophecy/Belief system — persistent macro convictions for Layer 2.

Manages a set of beliefs (macro convictions) that persist across invocations.
Each belief has a thesis, conviction level, direction, target price, timeframe,
supporting/opposing evidence, and checkpoints with dates/conditions that get
auto-evaluated against live prices.

Layer 2 reads these beliefs every invocation to maintain strategic context
and compare technical signals against fundamental convictions.
"""

import json
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from copy import deepcopy

logger = logging.getLogger("portfolio.prophecy")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
PROPHECY_FILE = DATA_DIR / "prophecy.json"

# Belief schema
BELIEF_TEMPLATE = {
    "id": "",                    # unique identifier (e.g., "silver_bull_2026")
    "ticker": "",                # primary ticker (e.g., "XAG-USD")
    "thesis": "",                # text description of the conviction
    "direction": "neutral",      # "bullish", "bearish", "neutral"
    "conviction": 0.5,           # 0.0-1.0 conviction level
    "target_price": None,        # target price (USD)
    "target_timeframe": "",      # e.g., "2026-Q4", "6 months"
    "entry_price": None,         # price when belief was created
    "created_at": "",            # ISO-8601
    "updated_at": "",            # ISO-8601
    "status": "active",          # "active", "paused", "expired", "confirmed", "invalidated"
    "supporting_evidence": [],   # list of strings
    "opposing_evidence": [],     # list of strings
    "checkpoints": [],           # list of checkpoint dicts
    "tags": [],                  # e.g., ["metals", "macro", "geopolitical"]
    "notes": "",                 # free-form notes
}

CHECKPOINT_TEMPLATE = {
    "id": "",                    # unique checkpoint identifier
    "condition": "",             # human-readable condition (e.g., "XAG breaks $35")
    "target_value": None,        # numeric target (price level)
    "comparison": "above",       # "above", "below", "between"
    "deadline": None,            # ISO-8601 deadline (optional)
    "status": "pending",         # "pending", "triggered", "expired", "missed"
    "triggered_at": None,        # when condition was met
    "created_at": "",            # ISO-8601
}


def load_beliefs():
    """Load all beliefs from prophecy.json.

    Returns:
        dict: {"beliefs": [...], "metadata": {...}}
    """
    if not PROPHECY_FILE.exists():
        return {"beliefs": [], "metadata": {"version": 1, "last_review": None}}

    try:
        data = json.loads(PROPHECY_FILE.read_text(encoding="utf-8"))
        if isinstance(data, list):
            # Legacy format — wrap in dict
            return {"beliefs": data, "metadata": {"version": 1, "last_review": None}}
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to load prophecy.json: %s", e)
        return {"beliefs": [], "metadata": {"version": 1, "last_review": None}}


def save_beliefs(data):
    """Save beliefs to prophecy.json."""
    data["metadata"]["last_review"] = datetime.now(timezone.utc).isoformat()
    with open(PROPHECY_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def add_belief(belief_dict):
    """Add a new belief. Fills in defaults from template.

    Args:
        belief_dict: Dict with belief fields. Must include "id" and "ticker".

    Returns:
        The added belief dict.
    """
    data = load_beliefs()

    # Check for duplicate ID
    existing_ids = {b["id"] for b in data["beliefs"]}
    if belief_dict.get("id") in existing_ids:
        raise ValueError(f"Belief with id '{belief_dict['id']}' already exists")

    # Merge with template
    belief = {**BELIEF_TEMPLATE, **belief_dict}
    now = datetime.now(timezone.utc).isoformat()
    if not belief["created_at"]:
        belief["created_at"] = now
    if not belief["updated_at"]:
        belief["updated_at"] = now

    data["beliefs"].append(belief)
    save_beliefs(data)
    return belief


def update_belief(belief_id, updates):
    """Update an existing belief.

    Args:
        belief_id: The belief's unique ID.
        updates: Dict of fields to update.

    Returns:
        The updated belief dict, or None if not found.
    """
    data = load_beliefs()

    for i, belief in enumerate(data["beliefs"]):
        if belief["id"] == belief_id:
            belief.update(updates)
            belief["updated_at"] = datetime.now(timezone.utc).isoformat()
            data["beliefs"][i] = belief
            save_beliefs(data)
            return belief

    return None


def remove_belief(belief_id):
    """Remove a belief by ID.

    Returns:
        True if removed, False if not found.
    """
    data = load_beliefs()
    original_len = len(data["beliefs"])
    data["beliefs"] = [b for b in data["beliefs"] if b["id"] != belief_id]

    if len(data["beliefs"]) < original_len:
        save_beliefs(data)
        return True
    return False


def get_belief(belief_id):
    """Get a single belief by ID.

    Returns:
        Belief dict or None.
    """
    data = load_beliefs()
    for belief in data["beliefs"]:
        if belief["id"] == belief_id:
            return belief
    return None


def get_active_beliefs(ticker=None):
    """Get all active beliefs, optionally filtered by ticker.

    Returns:
        List of active belief dicts.
    """
    data = load_beliefs()
    beliefs = [b for b in data["beliefs"] if b.get("status") == "active"]
    if ticker:
        beliefs = [b for b in beliefs if b.get("ticker") == ticker]
    return beliefs


def add_checkpoint(belief_id, checkpoint_dict):
    """Add a checkpoint to an existing belief.

    Args:
        belief_id: The belief's unique ID.
        checkpoint_dict: Dict with checkpoint fields.

    Returns:
        The added checkpoint dict, or None if belief not found.
    """
    data = load_beliefs()

    for i, belief in enumerate(data["beliefs"]):
        if belief["id"] == belief_id:
            cp = {**CHECKPOINT_TEMPLATE, **checkpoint_dict}
            if not cp["created_at"]:
                cp["created_at"] = datetime.now(timezone.utc).isoformat()
            if not cp["id"]:
                cp["id"] = f"cp_{len(belief.get('checkpoints', []))}"

            if "checkpoints" not in belief:
                belief["checkpoints"] = []
            belief["checkpoints"].append(cp)
            belief["updated_at"] = datetime.now(timezone.utc).isoformat()
            data["beliefs"][i] = belief
            save_beliefs(data)
            return cp

    return None


def evaluate_checkpoints(prices_usd):
    """Evaluate all pending checkpoints against current prices.

    Args:
        prices_usd: Dict {ticker: price_usd} of current prices.

    Returns:
        List of newly triggered checkpoint dicts (with belief_id added).
    """
    data = load_beliefs()
    triggered = []
    modified = False
    now = datetime.now(timezone.utc)

    for i, belief in enumerate(data["beliefs"]):
        if belief.get("status") != "active":
            continue

        ticker = belief.get("ticker", "")
        current_price = prices_usd.get(ticker)
        if current_price is None:
            continue

        for j, cp in enumerate(belief.get("checkpoints", [])):
            if cp.get("status") != "pending":
                continue

            # Check deadline expiry
            deadline = cp.get("deadline")
            if deadline:
                try:
                    deadline_dt = datetime.fromisoformat(deadline)
                    if now > deadline_dt:
                        cp["status"] = "expired"
                        data["beliefs"][i]["checkpoints"][j] = cp
                        modified = True
                        continue
                except (ValueError, TypeError):
                    pass

            # Check condition
            target = cp.get("target_value")
            comparison = cp.get("comparison", "above")

            if target is None:
                continue

            met = False
            if comparison == "above" and current_price >= target:
                met = True
            elif comparison == "below" and current_price <= target:
                met = True
            elif comparison == "between":
                # target_value should be [low, high]
                if isinstance(target, (list, tuple)) and len(target) == 2:
                    if target[0] <= current_price <= target[1]:
                        met = True

            if met:
                cp["status"] = "triggered"
                cp["triggered_at"] = now.isoformat()
                data["beliefs"][i]["checkpoints"][j] = cp
                modified = True
                triggered.append({**cp, "belief_id": belief["id"], "ticker": ticker, "price": current_price})

    if modified:
        save_beliefs(data)

    return triggered


def get_context_for_layer2(prices_usd=None):
    """Build compact belief context for Layer 2 consumption.

    Returns a dict suitable for inclusion in agent_summary_compact.json.
    Only includes active beliefs with relevant context.

    Args:
        prices_usd: Current prices for progress calculation.

    Returns:
        dict: {
            "beliefs": [
                {
                    "id": "silver_bull_2026",
                    "ticker": "XAG-USD",
                    "direction": "bullish",
                    "conviction": 0.8,
                    "thesis": "Silver to $120...",
                    "target_price": 120.0,
                    "progress_pct": 15.2,  # % of way from entry to target
                    "checkpoints_summary": "2/5 triggered",
                    "tags": ["metals", "macro"],
                }
            ],
            "total_active": 3,
        }
    """
    active = get_active_beliefs()

    if not active:
        return {"beliefs": [], "total_active": 0}

    compact_beliefs = []
    for belief in active:
        entry = {
            "id": belief["id"],
            "ticker": belief.get("ticker", ""),
            "direction": belief.get("direction", "neutral"),
            "conviction": belief.get("conviction", 0.5),
            "thesis": belief.get("thesis", "")[:200],  # truncate for compactness
            "target_price": belief.get("target_price"),
            "tags": belief.get("tags", []),
        }

        # Compute progress toward target
        if prices_usd and belief.get("ticker") in prices_usd and belief.get("target_price") and belief.get("entry_price"):
            current = prices_usd[belief["ticker"]]
            entry_price = belief["entry_price"]
            target = belief["target_price"]

            if target != entry_price:
                progress = (current - entry_price) / (target - entry_price) * 100
                entry["progress_pct"] = round(progress, 1)
                entry["current_price"] = round(current, 2)

        # Checkpoint summary
        checkpoints = belief.get("checkpoints", [])
        if checkpoints:
            triggered = sum(1 for cp in checkpoints if cp.get("status") == "triggered")
            total = len(checkpoints)
            entry["checkpoints_summary"] = f"{triggered}/{total} triggered"

        compact_beliefs.append(entry)

    return {
        "beliefs": compact_beliefs,
        "total_active": len(compact_beliefs),
    }


def print_prophecy_review():
    """Print a human-readable review of all beliefs."""
    data = load_beliefs()
    beliefs = data.get("beliefs", [])

    if not beliefs:
        print("No beliefs configured. Seed data/prophecy.json with macro convictions.")
        return

    print("=== Prophecy / Belief Review ===\n")

    active = [b for b in beliefs if b.get("status") == "active"]
    inactive = [b for b in beliefs if b.get("status") != "active"]

    for belief in active:
        direction_symbol = "^" if belief.get("direction") == "bullish" else "v" if belief.get("direction") == "bearish" else ">"
        conv = belief.get("conviction", 0)
        conv_bar = "#" * int(conv * 10) + "." * (10 - int(conv * 10))

        print(f"  {direction_symbol} [{belief['id']}] {belief.get('ticker', '?')}")
        print(f"    Thesis: {belief.get('thesis', 'N/A')}")
        print(f"    Conviction: [{conv_bar}] {conv:.0%}")
        if belief.get("target_price"):
            print(f"    Target: ${belief['target_price']} ({belief.get('target_timeframe', 'N/A')})")
        if belief.get("entry_price"):
            print(f"    Entry: ${belief['entry_price']}")

        # Checkpoints
        cps = belief.get("checkpoints", [])
        if cps:
            triggered_count = sum(1 for cp in cps if cp.get("status") == "triggered")
            print(f"    Checkpoints: {triggered_count}/{len(cps)}")
            for cp in cps:
                status_icon = "[x]" if cp.get("status") == "triggered" else "[ ]" if cp.get("status") == "pending" else "[!]"
                print(f"      {status_icon} {cp.get('condition', '?')} [{cp.get('status', '?')}]")

        # Evidence
        supporting = belief.get("supporting_evidence", [])
        opposing = belief.get("opposing_evidence", [])
        if supporting:
            print(f"    Supporting ({len(supporting)}):")
            for ev in supporting[:3]:
                print(f"      + {ev}")
        if opposing:
            print(f"    Opposing ({len(opposing)}):")
            for ev in opposing[:3]:
                print(f"      - {ev}")
        print()

    if inactive:
        print(f"  ({len(inactive)} inactive beliefs not shown)")
