"""Per-instrument predictive playbooks.

The user requirement: a *unique approach per instrument* — BTC is predicted
differently from ETH, silver differently from gold, a warrant differently from
its underlying. Each ``Playbook`` declares:

- ``signal_emphasis`` — which stored signals/equations matter most (prep.py
  bundles these from agent_summary; the prompt tells Claude to weight them),
- ``equations``       — explicit closed-form relationships to apply,
- ``web_questions``   — what /deep-research should investigate,
- ``forum_sources``   — where to read "what people think" (crowd sentiment),
- ``special_factors`` — instrument-specific drivers,
- ``price_model``     — how the target price is derived,
- ``required_inputs`` — feed keys prep tries to populate; coverage sufficiency
                        is graded ``found / len(required_inputs)``.

These keys are referenced by prep.py (bundling + coverage grading) and rendered
into the Claude prompt verbatim per instrument.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Playbook:
    instrument: str
    strategy_id: str
    asset_class: str  # crypto | metals | equity | oil | warrant
    price_model: str
    underlying: str | None = None  # warrants point at their underlying
    signal_emphasis: list[str] = field(default_factory=list)
    equations: list[str] = field(default_factory=list)
    web_questions: list[str] = field(default_factory=list)
    forum_sources: list[str] = field(default_factory=list)
    special_factors: list[str] = field(default_factory=list)
    required_inputs: list[str] = field(default_factory=list)


PLAYBOOKS: dict[str, Playbook] = {
    "BTC-USD": Playbook(
        instrument="BTC-USD",
        strategy_id="btc_onchain_flow_cycle",
        asset_class="crypto",
        price_model="On-chain valuation bands (MVRV/NUPL) anchor fair value; ETF net "
        "flow + funding/basis set short-horizon drift; F&G contrarian at extremes.",
        signal_emphasis=["on_chain_btc", "fear_greed", "drift_regime_gate", "rsi",
                         "mean_reversion", "statistical_jump_regime", "bocpd_regime_switch"],
        equations=["MVRV-Z = (mktcap - realised_cap) / std(mktcap)",
                   "NUPL = (mktcap - realised_cap) / mktcap",
                   "perp basis annualised = (perp - spot)/spot * (365/days)"],
        web_questions=["Spot BTC ETF net flows last 5 sessions and trend",
                       "BTC miner capitulation / hash ribbons status",
                       "Macro: DXY, real yields, risk-on/off this week",
                       "Any imminent regulatory/exchange solvency catalysts"],
        forum_sources=["r/Bitcoin", "crypto-Twitter / X", "TradingView BTC ideas"],
        special_factors=["halving-cycle position", "ETF flow regime", "stablecoin supply ratio"],
        required_inputs=["live_price", "on_chain_btc", "fear_greed", "regime",
                         "accuracy", "macro_beliefs", "recent_research"],
    ),
    "ETH-USD": Playbook(
        instrument="ETH-USD",
        strategy_id="eth_ratio_staking_beta",
        asset_class="crypto",
        price_model="Predict BTC first, then ETH/BTC ratio momentum + staking/L2 "
        "activity overlay; ETH = BTC_target * ratio_forecast.",
        signal_emphasis=["drift_regime_gate", "rsi", "mean_reversion", "crypto_macro",
                         "ml_classifier", "credit_spread_risk"],
        equations=["ETH = BTC * (ETH/BTC ratio)", "ratio momentum = EMA(ratio,7)/EMA(ratio,30)-1"],
        web_questions=["ETH/BTC ratio trend + drivers this week",
                       "ETH spot ETF flows + staking inflow/withdrawal queue",
                       "L2 / gas activity + DeFi TVL trend",
                       "Any Ethereum protocol upgrade catalysts"],
        forum_sources=["r/ethereum", "r/ethfinance", "crypto-Twitter"],
        special_factors=["BTC beta (~0.8-1.1)", "staking yield", "ETH/BTC regime"],
        required_inputs=["live_price", "crypto_macro", "regime", "rsi",
                         "accuracy", "macro_beliefs", "recent_research"],
    ),
    "XAU-USD": Playbook(
        instrument="XAU-USD",
        strategy_id="gold_realyield_dxy_cb",
        asset_class="metals",
        price_model="Real-yield + DXY regression anchors fair value; central-bank "
        "demand + geopolitical premium adjust; seasonality tilt.",
        signal_emphasis=["metals_cross_asset", "realized_skewness", "williams_vix_fix",
                         "rsi", "mean_reversion", "drift_regime_gate"],
        equations=["dGold ~ -beta * d(real_10y) - gamma * d(DXY)",
                   "Gold/Silver ratio mean-reversion band"],
        web_questions=["10y TIPS real yield + DXY direction this week",
                       "Central-bank gold buying + ETF holdings trend",
                       "Geopolitical risk premium drivers",
                       "Gold seasonality for current month"],
        forum_sources=["Kitco forums", "r/Gold", "TradingView gold ideas"],
        special_factors=["real yields (primary)", "DXY inverse", "CB buying", "geopolitics"],
        required_inputs=["live_price", "metals_cross_asset", "regime", "rsi",
                         "accuracy", "macro_beliefs", "recent_research"],
    ),
    "XAG-USD": Playbook(
        instrument="XAG-USD",
        strategy_id="silver_gsr_industrial_cot",
        asset_class="metals",
        price_model="Gold-Silver Ratio mean-reversion is primary: silver_target = "
        "gold_target / GSR_forecast; industrial/solar demand + COT positioning overlay. "
        "Higher beta + vol than gold.",
        signal_emphasis=["metals_cross_asset", "cot_positioning", "williams_vix_fix",
                         "rsi", "mean_reversion", "amihud_illiquidity_regime"],
        equations=["silver = gold / GSR", "GSR mean-reversion toward 60-80 fair band",
                   "silver realised vol ~ 1.5-2x gold"],
        web_questions=["Gold-Silver Ratio level + mean-reversion setup",
                       "Industrial/solar/EV silver demand trend + mine supply deficit",
                       "CFTC COT silver positioning (commercials vs managed money)",
                       "Silver ETP inflows"],
        forum_sources=["r/Silverbugs", "WallStreetSilver", "TradingView silver ideas"],
        special_factors=["GSR (primary)", "solar/industrial demand", "COT", "supply deficit"],
        required_inputs=["live_price", "metals_cross_asset", "cot_positioning", "regime",
                         "accuracy", "macro_beliefs", "recent_research"],
    ),
    "MSTR": Playbook(
        instrument="MSTR",
        strategy_id="mstr_mnav_btc_beta",
        asset_class="equity",
        price_model="Leveraged BTC proxy: MSTR_target = f(BTC_target) * mNAV_premium; "
        "BTC beta ~1.5-2x; convertible-debt/dilution overlay.",
        signal_emphasis=["on_chain_btc", "drift_regime_gate", "rsi", "mean_reversion"],
        equations=["mNAV = mktcap / (btc_held * BTC_price)",
                   "dMSTR ~ beta_btc * dBTC, beta in [1.5, 2.2]"],
        web_questions=["MSTR mNAV premium/discount to BTC NAV right now",
                       "Recent MSTR equity issuance / convertible debt / BTC purchases or sales",
                       "MSTR options skew + short interest",
                       "BTC outlook (feeds MSTR via beta)"],
        forum_sources=["r/MSTR", "StockTwits MSTR", "FinTwit"],
        special_factors=["mNAV premium", "BTC beta", "dilution/issuance", "US market hours"],
        required_inputs=["live_price", "on_chain_btc", "regime", "rsi",
                         "accuracy", "recent_research"],
    ),
    "CL=F": Playbook(
        instrument="CL=F",
        strategy_id="wti_opec_inventory_term",
        asset_class="oil",
        price_model="Inventory surprise + OPEC+ supply path set level; term-structure "
        "(contango/backwardation) sets carry/drift; USD + demand overlay.",
        signal_emphasis=["rsi", "mean_reversion", "drift_regime_gate"],
        equations=["backwardation = front - deferred > 0 (bullish carry)",
                   "dWTI ~ -k * d(EIA_inventory_surprise)"],
        web_questions=["EIA crude inventory last print + surprise vs consensus",
                       "OPEC+ latest decision / quota compliance",
                       "WTI term structure: contango or backwardation",
                       "China + global demand signals, USD direction"],
        forum_sources=["oil FinTwit", "r/oil", "TradingView WTI ideas"],
        special_factors=["OPEC+ supply", "EIA inventories", "term structure", "USD"],
        required_inputs=["live_price", "regime", "rsi", "recent_research"],
    ),
    "BZ=F": Playbook(
        instrument="BZ=F",
        strategy_id="brent_global_supply_spread",
        asset_class="oil",
        price_model="Global supply/geopolitics set level; WTI-Brent spread + freight "
        "overlay; predict WTI, then Brent = WTI + spread_forecast.",
        signal_emphasis=["rsi", "mean_reversion", "drift_regime_gate"],
        equations=["Brent = WTI + (Brent-WTI spread)", "spread widens on non-US supply risk"],
        web_questions=["Brent-WTI spread level + drivers",
                       "Geopolitical supply risk (Hormuz, Russia, Middle East)",
                       "OPEC+ supply path, Asian demand",
                       "Global floating storage / freight rates"],
        forum_sources=["oil FinTwit", "r/oil"],
        special_factors=["geopolitics (primary)", "WTI-Brent spread", "Asian demand"],
        required_inputs=["live_price", "regime", "rsi", "recent_research"],
    ),
    # --- warrants: predict the UNDERLYING then apply leverage/parity ------
    "XBT-TRACKER": Playbook(
        instrument="XBT-TRACKER",
        strategy_id="warrant_btc_tracker",
        asset_class="warrant",
        underlying="BTC-USD",
        price_model="Predict BTC-USD first; map to warrant via parity/leverage "
        "(P=(S-K)*FX/r, Omega=S/(S-K)); flag barrier-proximity risk. Needs live "
        "parity/barrier/strike from Avanza — absent => coverage downgrade.",
        signal_emphasis=["on_chain_btc", "fear_greed", "drift_regime_gate"],
        equations=["P = (S - K) * FX / parity_ratio", "Omega (leverage) = S / (S - K)"],
        web_questions=["BTC outlook (drives the tracker)"],
        forum_sources=["r/Bitcoin", "Avanza forum"],
        special_factors=["leverage/parity", "barrier proximity", "issuer spread"],
        required_inputs=["live_price", "underlying_prediction", "warrant_parity",
                         "warrant_barrier", "warrant_strike"],
    ),
    "ETH-TRACKER": Playbook(
        instrument="ETH-TRACKER",
        strategy_id="warrant_eth_tracker",
        asset_class="warrant",
        underlying="ETH-USD",
        price_model="Predict ETH-USD first; map via parity/leverage; flag barrier risk.",
        signal_emphasis=["crypto_macro", "drift_regime_gate"],
        equations=["P = (S - K) * FX / parity_ratio", "Omega = S / (S - K)"],
        web_questions=["ETH outlook (drives the tracker)"],
        forum_sources=["r/ethereum", "Avanza forum"],
        special_factors=["leverage/parity", "barrier proximity", "issuer spread"],
        required_inputs=["live_price", "underlying_prediction", "warrant_parity",
                         "warrant_barrier", "warrant_strike"],
    ),
    "MINI-SILVER": Playbook(
        instrument="MINI-SILVER",
        strategy_id="warrant_silver_mini_5x",
        asset_class="warrant",
        underlying="XAG-USD",
        price_model="Predict XAG-USD first; apply ~5x mini leverage via parity/barrier "
        "(P=(S-K)*FX/r); barrier proximity dominates risk at 5x.",
        signal_emphasis=["metals_cross_asset", "cot_positioning", "williams_vix_fix"],
        equations=["P = (S - K) * FX / parity_ratio", "Omega = S / (S - K) ~ 5x"],
        web_questions=["Silver outlook (drives the mini)"],
        forum_sources=["r/Silverbugs", "Avanza forum"],
        special_factors=["~5x leverage", "barrier proximity (primary risk)", "issuer spread"],
        required_inputs=["live_price", "underlying_prediction", "warrant_parity",
                         "warrant_barrier", "warrant_strike"],
    ),
    # --- Tier-2 Swedish equities: no signal feed -> expect coverage gaps ---
    "SAAB-B": Playbook(
        instrument="SAAB-B",
        strategy_id="saab_defense_orders",
        asset_class="equity",
        price_model="Defense-spend + order-book momentum; OMX beta; earnings/guidance "
        "catalysts. No internal signal feed -> rely on research + price action.",
        signal_emphasis=[],
        equations=["dSAAB ~ beta_omx * dOMX + idiosyncratic order news"],
        web_questions=["Recent SAAB order announcements + defense-budget trend (EU/NATO)",
                       "SAAB analyst revisions + next earnings date",
                       "OMX Stockholm direction this week"],
        forum_sources=["Placera", "Avanza forum", "Flashback finans"],
        special_factors=["defense spending", "order intake", "OMX beta", "SEK/EUR"],
        required_inputs=["live_price", "recent_research"],
    ),
    "SEB-C": Playbook(
        instrument="SEB-C",
        strategy_id="seb_rates_credit_nim",
        asset_class="equity",
        price_model="Bank: rate path + NIM + credit quality drive earnings; OMX beta; "
        "dividend calendar. No internal signal feed.",
        signal_emphasis=[],
        equations=["bank value ~ f(NIM, loan growth, credit losses, rate path)"],
        web_questions=["Riksbank/ECB rate path + Swedish credit conditions",
                       "SEB analyst revisions + dividend/earnings calendar",
                       "Nordic bank sector sentiment"],
        forum_sources=["Placera", "Avanza forum"],
        special_factors=["rate path / NIM", "credit quality", "dividend", "OMX beta"],
        required_inputs=["live_price", "recent_research"],
    ),
    "INVE-B": Playbook(
        instrument="INVE-B",
        strategy_id="investor_nav_discount",
        asset_class="equity",
        price_model="Holding co: NAV-discount mean-reversion + underlying holdings "
        "(ABB, AstraZeneca, Atlas Copco, EQT...) performance; OMX beta. No signal feed.",
        signal_emphasis=[],
        equations=["price = NAV * (1 - discount)", "discount mean-reverts to ~5-15% band"],
        web_questions=["Investor AB NAV-discount level vs historical band",
                       "Key holdings (ABB, AZN, Atlas Copco, EQT) recent moves",
                       "OMX direction + any Investor AB news"],
        forum_sources=["Placera", "Avanza forum"],
        special_factors=["NAV discount (primary)", "holdings performance", "OMX beta"],
        required_inputs=["live_price", "recent_research"],
    ),
}


def playbook_for(instrument: str) -> Playbook | None:
    return PLAYBOOKS.get(instrument)


def all_instruments() -> list[str]:
    return list(PLAYBOOKS.keys())
