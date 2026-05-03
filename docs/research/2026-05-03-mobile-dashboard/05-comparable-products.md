# 05 — Comparable Products: Mobile UX Patterns from Trading & Quant-Finance Apps

**Goal:** principles, not pixel-copies. We are redesigning the finance-analyzer dashboard
(currently desktop-first Flask + vanilla JS, ~3200 LOC) to be primarily phone-displayed.
This document distills how 8 established trading apps have already solved "dense quant
data on a 390 px screen", what they do badly, and what we should reuse.

The finance-analyzer's specific dense-data shapes that drive this research:
- A signal heatmap of 33 active signals × ~7 timeframes × 5 instruments
- Decision history tables with 8–10 columns
- Multi-line charts with annotations (price + signal markers + entries/exits)
- Real-time numeric tickers for BTC, ETH, XAU, XAG, MSTR with up/down momentum
- Per-strategy P&L (Patient + Bold + Warrants) and accuracy drill-downs

---

## 1. TradingView Mobile

**Home screen.** The Chart tab is the default landing surface. Above-the-fold: one
selected instrument with a candle chart at the user's last-used timeframe. A compact
legend shows close price + percentage change only (additional metrics like O/H/L
appear only on long-press "tracking mode"). Watchlist preview is one tap away.

**Navigation.** Bottom tab bar. The dominant tabs across reviews are: Chart,
Watchlist, News/Markets, Ideas, More. Watchlist is intentionally placed bottom-left
so the dominant thumb (right hand) reaches Chart and More most easily.

**Dense data.**
- Watchlist rows show symbol logo, last price, % change. Quick previews of charts
  and symbol info appear inline within the watchlist itself, removing the need to
  drill into a separate detail page just to see a sparkline.
- Chart hides the right widget bar entirely on mobile; the account manager and
  time-scale marks relocate or disappear; only one price scale renders at a time.
- Three-gesture vocabulary: single tap = drag chart / scale, long press =
  crosshair tracking mode that surfaces all OHLC + indicator values, double tap =
  enter line-edit mode for overlays.
- Multi-chart layouts on mobile are supported via a swift chart-switcher
  (functionally equivalent to swiping between charts), not simultaneous grids.
  ([source](https://www.tradingview.com/blog/en/multi-chart-view-on-a-mobile-screen-10924/))

**Alerts / notifications.** Bell icon opens a dedicated alerts panel listing all
active alerts with edit / delete inline. Triggered alerts deliver via push + optional
in-app pop-up, email, or sound. Push and the in-app inbox are deliberately
separate surfaces — push is ephemeral, the inbox is the audit trail.
([source](https://www.tradingview.com/support/solutions/43000595315-how-to-set-up-alerts/))

**Touch ergonomics.** Symbol switch, settings, and the alert-create FAB all
sit in the bottom corners. Pinch zoom on chart, slide-to-zoom on price axis,
double-tap-price to auto-fit.
([source](https://www.tradingview.com/charting-library-docs/latest/mobile_specifics/))

**Failure modes.**
- Trustpilot rating ~1.9/5 (≈50% one-star). Top complaints: chart freezes when
  switching timeframes; mobile drops to a different ticker on chart load;
  paid-tier features inaccessible on mobile; AI-only support loops.
  ([source](https://www.trustpilot.com/review/tradingview.com))
- "Sleek but restrictive for heavy charting and indicator management" is a
  consistent power-user complaint. ([source](https://justuseapp.com/en/app/1205990992/tradingview-track-all-markets/reviews))
- Indicator management UX is buried — adding a single indicator on mobile takes
  more taps than it should.

---

## 2. Coinbase / Coinbase Advanced Mobile

**Home screen.** A "Dashboard" page that shows top assets and a buy CTA, designed
to fit on a single non-scrollable screen with only essentials visible. Coinbase
explicitly engineers the Simple view to hide the order book entirely from
non-pros. ([source](https://help.coinbase.com/en/coinbase/trading-and-funding/advanced-trade/dashboard-overview))

**Navigation.** Coinbase runs two products on one app: Simple Trade (default) and
Advanced Trade (toggled). Simple = card-stack of holdings + Buy/Sell. Advanced
unlocks charts, order book, and Level-2 data behind a mode toggle, not a
separate tab. This dual-mode pattern is worth noting — same data, two
information densities, user picks.

**Dense data.**
- Order book on mobile uses a vertical split: asks in red on top, bids in green
  on bottom, price ladder reading top-to-bottom. (Compare desktop, which shows
  asks-and-bids side-by-side.) Vertical stack works because phone screens are
  taller than wide.
- Charts use the Advanced view; tools include depth view, candlestick variants,
  and overlays.
- Trade history scrolls inline below the order book.

**Alerts / notifications.** Push for price alerts and order fills; in-app inbox
under a bell icon. Alerts are coarse-grained vs TradingView (no indicator-based
alerts).

**Touch ergonomics.** Simple Trade is heavily thumb-zone optimized. Advanced
trades the cleanliness for density — buttons get smaller, screens scrollier.

**Failure modes.**
- "Coinbase has added many features and these additions have affected
  usability". Bloat from product expansion is the dominant criticism.
  ([source](https://jpux.medium.com/case-study-coinbase-ux-redesign-9fa4038f5d52))
- The Simple/Advanced toggle is itself a tax: switching modes reloads the
  surface; orientation in Advanced is harder than expected.
- Pro mobile is widely seen as a clipped version of Pro web — analytics that
  should be a chart on mobile are absent.

---

## 3. Robinhood Mobile

**Home screen.** Single hero number: portfolio value. Below it, the canonical
day-chart of total portfolio value. Below that, a card stack of holdings, each
card showing one asset with a tappable expand-for-details affordance. Above
the fold = "how am I doing today". ([source](https://worldbusinessoutlook.com/how-the-robinhood-ui-balances-simplicity-and-strategy-on-mobile/))

**Navigation.** Bottom bar with 4–5 destinations: Home, Browse/Search, Notifications,
Account. Buy/Sell is contextual on the asset page — not a global FAB.

**Dense data.** Robinhood's design philosophy is the *opposite* of dense:
"users don't have to deal with messy dashboards, they expand into areas they
care about". Cards default to a single number with optional expansion; charts
default to one timeframe; details surface progressively. Color is functional
(green/red/grey/light-green/blue) and consistent across the app.

**Alerts / notifications.** A dedicated Notifications tab in the bottom bar
which serves as the in-app inbox for fills, alerts, news, and social events. Push
goes to OS-level. Notifications inbox is also the social/announcements surface
— this conflation is part of the gamification critique.

**Touch ergonomics.** Excellent thumb reach. One-tap watchlist add, swipe-friendly
asset cards, vertical scroll feed model.

**Failure modes (the cautionary tale).**
- Confetti on every trade was an explicit dopamine loop and was removed in 2021
  after public backlash. ([source](https://www.cnbc.com/2021/03/31/robinhood-gets-rid-of-confetti-feature-amid-scrutiny-over-gamification.html))
- Massachusetts $7.5M settlement for "gamified investing approach" that
  encouraged risky behaviour in inexperienced users.
  ([source](https://tearsheet.co/marketing/the-double-edged-sword-of-good-ux-how-robinhoods-gamification-of-investing-backfired-during-the-market-downturn/))
- Top-trending stocks lists "much like top-ten scorers in a video game" surface
  the wrong thing — popularity, not edge.
  ([source](https://finmasters.com/gamification-of-investing/))
- Information thinness: the app deliberately omits the analytics a serious
  trader needs. Users check the app ~10×/day, conflating engagement with
  understanding. ([source](https://www.economicsonline.co.uk/all/behavioral-economics-and-the-gamification-of-finance-how-apps-like-robinhood-influence-trading-behavior.html/))

---

## 4. Webull Mobile

**Home screen.** A "Markets" hub displaying a heatmap, yield curves, net inflows,
and quick scanners. Reviewers describe Webull as "a mobile workstation, not just
an app". ([source](https://www.stockbrokers.com/review/webull))

**Navigation.** Bottom bar with five tabs: Markets, Watchlist, Feeds (social),
Menu, plus quote-search. The signature flourish is that drilling into a
ticker exposes 12 distinct sub-tabs of data on the detail page (Overview,
Chart, News, Financials, Options, Analysis, Press, ESG, Profile, etc.).

**Dense data.**
- 56 technical indicators on charts; "magnifying glass" pop-up while drawing
  trendlines so the touch finger never obscures the placement point — clever
  mobile-specific UX detail.
- Volume bars on the quote page visualize bid vs ask pressure (a density
  technique normally reserved for desktop Level-2 panels).
- "Replay Mode" lets users scrub price action as a time-lapse — uses the
  full screen because phones are bad at multi-pane analysis.
- News & Daily uses AI to compress filings/news into short summaries — small
  screens reward summarization over complete-text views.
- Mini heatmap on Markets hub uses tile size + color to show market cap +
  change at a glance.

**Alerts / notifications.** Alert center under Menu with granular per-ticker /
per-condition rules. Push, email, and in-app surfaces all active.

**Touch ergonomics.** Dense by design — Webull explicitly trades touch
spaciousness for desktop-grade density. This works because their target user
self-selects.

**Failure modes.**
- "Sheer volume of notifications can overwhelm users without customization".
  Defaults are noisy. ([source](https://www.stockbrokers.com/review/webull))
- "Extremely complicated" for beginners — 12-tab drill-down is a mountain
  if you don't know what you're looking for.
- Multi-leg options construction is "cumbersome compared to specialized
  platforms" — a sign that very interactive flows still struggle on mobile.
- Social Feeds tab adds noise to an already dense interface.

---

## 5. Bloomberg Mobile (Professional + Business apps)

**Home screen.** Bloomberg ships two apps. **Bloomberg Professional** (paid, for
Terminal subscribers) lands on a dashboard of news + watchlists + alerts, very
content-heavy. **Bloomberg: Business News** (free) lands on news cards + a
horizontal market-summary strip at the top.

**Navigation.** Both use a tab bar. Professional emphasizes News, Markets,
Portfolio, Alerts. Business emphasizes News, Markets, Watchlist, More. Cross-app
the design language is consistent: dark theme, monospace numerics,
information-first.

**Dense data.**
- Numerics are monospaced and column-aligned. Tables that on desktop use
  Bloomberg's signature high-contrast palette (black bg + amber/orange/yellow
  text) appear unchanged on mobile — the typographic system survives the
  shrink.
- Mini sparkline + last + % change is the canonical row format for any market
  list.
- News and data live side-by-side, never one-or-the-other — context is the
  product.

**Alerts / notifications.** Custom alert rules from the Terminal sync to the app;
push is event-by-event; in-app alert inbox is the audit log. Alerts integrate
with news (alert "X above 100" can also trigger contextual headlines).

**Touch ergonomics.** Functional rather than ergonomic — the audience is "people
who already use the Terminal", so touch density tilts toward density.

**Failure modes.**
- Streaming media (videos, audio) frequently fails — restart-required per app
  reviews. ([source](https://apps.apple.com/us/app/bloomberg-professional/id407761767))
- Some asset classes are read-only on mobile (commodities, bonds, FX, futures
  often don't expand into charts when tapped). ([source](https://play.google.com/store/apps/details?id=com.bloomberg.android.anywhere))
- Auth: opening any external trading app frequently logs you out of the
  Terminal app session.
- No bond yield curve chart on mobile — major omission for fixed-income users.

---

## 6. IBKR Mobile (and IBKR GlobalTrader)

**Home screen.** IBKR sells two mobile apps explicitly:
- **IBKR Mobile** lands on a Watchlist or last-used screen, near-1:1 power-port
  of TWS desktop.
- **IBKR GlobalTrader** lands on a portfolio/account screen with simpler card
  metrics.
([source](https://thepoorswiss.com/ibkr-global-trader/))

**Navigation.** IBKR Mobile uses a bottom tab bar covering Watchlists, Orders &
Trades, Portfolio, More. IBKR GlobalTrader: Home, Trade, Watchlist, Portfolio.
Both use bottom-thumb navigation but the *information per tab* differs by an
order of magnitude.

**Dense data.**
- Watchlist rows show symbol, last, change, change-percent, volume, bid/ask in
  one tight row — they keep the abbreviation discipline of a Bloomberg-style
  table on a phone.
- Order ticket is multi-step (route, type, TIF, conditional) — 4–5 progressive
  screens rather than one form.
- Charts default to compact, with a full-screen toggle that hides the tab bar.
- 150+ markets, 90+ exchanges — search is a primary surface, not buried.

**Alerts / notifications.** Conditional alerts ("price above X with volume
above Y") configurable on mobile. Push delivery + in-app alert center.

**Touch ergonomics.** Power-user-flavored. Hit areas are small, taps are dense,
but every action is reachable in 3 taps.

**Failure modes.**
- "Functional rather than polished — lacks the social features and clean UX"
  of consumer apps. ([source](https://www.newtrading.io/mobile-trading/))
- Steep learning curve — they ship two apps because one is genuinely
  inaccessible to newcomers.
- TWS Mobile's dense screens look identical on phone and tablet — phone users
  pay the price for tablet-grade density.

**Architectural lesson:** offering two apps with different density tiers is a
viable pattern when a single app can't serve both novice and pro. For our
single-user case we are the pro, so we don't need a Simple variant — but a
"summary mode" toggle is a possible compromise.

---

## 7. Avanza App (Swedish broker — user already trades here)

**Home screen.** Account list / portfolio summary card with total value, daily
change, holdings list. A horizontal scroll chips strip at the top filters
asset class (Aktier, Fonder, Sparkonto, etc.).

**Navigation.** Bottom bar covering Översikt (overview), Aktier (stocks), Fonder
(funds), Listor (lists), Mer (more). Swedish UX convention follows iOS Material
hybrid.

**Dense data.**
- Holdings rows are tight: name, ISIN-fragment or ticker, today's % change,
  total value. Sparklines occasionally appear on the asset detail page but are
  not in the holdings list.
- Order book on a stock detail page is a vertical ladder, 5 levels deep by
  default with a "see more" tap.
- Charts use a horizontal time-period strip (1D, 1V (vecka=week), 1M, 3M,
  YTD, 1Å, 3Å, 5Å, Max) — typical broker pattern.

**Alerts / notifications.** A Notifications tab/inbox + push. "Bevakningar"
(watches) live separately from notifications.

**Touch ergonomics.** Per Trustpilot reviews, navigation requires "many more
clicks than expected" and users land on the "wrong screen first" frequently.
Trade order flow is a multi-screen wizard.
([source](https://www.trustpilot.com/review/avanza.se))

**Failure modes.**
- "Very messy and cluttered, can never find what I'm looking for easily" is
  the dominant Trustpilot complaint.
- App "completely gives up when lots of trading is going on" — perceived
  reliability issues at peak hours.
- Order book often stops updating, requiring manual refresh — a tell that
  websocket reconnect logic is fragile.
- Buy/sell orders cancel on market close without intuitive feedback.

**Lesson for us.** Our user already lives in the Avanza app for execution —
the finance-analyzer dashboard should NOT compete on order entry. It should
complement: signal layer, accuracy drill-down, decision audit. Don't try to
reproduce Avanza's order tickets — link out where appropriate.

---

## 8. Tastytrade Mobile (options-focused)

**Home screen.** Dashboard tab with a market snapshot, market news, and
watchlists. Tastytrade unified the iOS, Android, web, and desktop dashboards on
the same conceptual layout — consistency across surfaces is a deliberate brand
choice. ([source](https://support.tastytrade.com/support/s/solutions/articles/43000435224))

**Navigation.** Customizable bottom navigation bar. Users explicitly drag-drop
to rearrange tabs via a "More" tab — the UX gives the user permission to
reshape primary nav. This is rare and powerful for a power-user app.

**Dense data.**
- Compact Table Layout toggle in settings — adds extra columns at the cost of
  per-cell padding. Density-as-a-preference is well-suited to a split user
  base.
- Greeks (delta, gamma, theta, vega) and bid/ask spreads are first-class on the
  options ticket.
- POP (Probability of Profit) and trader comments per position — tastytrade
  surfaces both quant + qualitative judgment on the same row.

**Alerts / notifications.** Reliable for fills + position alerts; less granular
than IBKR.

**Touch ergonomics.** Practical, not flashy. Hit targets adequate. Reviewers
note the curve analysis tool is "nearly unusable on mobile" because of screen
constraints — interactive payoff diagrams hit the screen-size wall.
([source](https://www.stockbrokers.com/review/tastytrade))

**Failure modes.**
- "Mobile feels like an afterthought compared to desktop". Anything truly
  multi-axis (curves, complex spreads) is "buried in clunky menus".
- Curve analysis was specifically called out as the failure case.
- Heavy reliance on parity between platforms means mobile inherits desktop's
  defaults rather than getting mobile-native shortcuts.

---

## 9. (Optional) MetaTrader 4 / 5 Mobile

**Home screen.** Quotes tab as default — a stacked list of currency pairs /
instruments with bid, ask, spread, and last update time.

**Navigation.** Bottom bar with Quotes, Charts, Trade, History, News, More.

**Dense data.**
- Quotes list is the textbook example of "vertical card list of tickers, each
  row = one instrument, all numerics tight-aligned".
- 30 built-in indicators + 24 analytical objects on charts.
- Three-finger swipe between asset classes (forex / metals / indices) — a
  power-user gesture you discover, not see.
- Pinch zoom transitions seamlessly from 1m to 1M timeframes — gesture
  velocity drives the timeframe.
([source](https://www.metaquotes.net/en/metatrader5/mobile-trading))

**Alerts / notifications.** Push; alert log in History tab. Alerts live with
trade history rather than as a standalone inbox.

**Touch ergonomics.** Old-school dense. Reviewers say "UX feels a bit outdated"
but "MT4 has a straightforward interface" once learned.

**Failure modes.** Aged visual language; assumes forex muscle memory; chart
toolbar is small; news tab is anaemic.

---

## Patterns we should adopt

1. **Bottom tab bar with 4–5 destinations, watchlist on the left.** TradingView,
   Robinhood, Webull, IBKR, MetaTrader, Tastytrade, Avanza all converge on this.
   Right-thumb users default to chart/details on the right; the discoverable
   "less common" tab (settings/more) on the right edge. For us:
   `Overview | Signals | Decisions | Health | More`.
2. **Default ticker row = symbol, price, % change, sparkline.** This is the
   universal compact format. Bloomberg + Webull + IBKR + TradingView all
   commit to this because it's the highest density a phone can scan in <0.5s.
3. **Long-press as the "more data" gesture on charts.** TradingView's
   crosshair-on-long-press is the cleanest pattern: keep the default view
   sparse, surface OHLC + indicator values only when the user explicitly asks.
   For us: long-press on a heatmap cell shows the full per-signal vote with
   ticker × timeframe × accuracy.
4. **Bottom sheet for asset / signal details.** Material's expanding bottom
   sheet — collapsed pill at the bottom that drags up to half-screen, then
   full-screen. Avoids losing context when drilling in.
   ([source](https://www.nngroup.com/articles/bottom-sheet/)) Suits our
   "tap a heatmap cell, get the rationale without leaving the heatmap" need.
5. **Density toggle, not density default.** Tastytrade's "Compact Table
   Layout" preference is the cleanest answer to the "but I want pro density"
   complaint. Default to spacious, opt-in to dense.
6. **Sparklines inline + monospaced numerics + decimal alignment.** Bloomberg's
   typographic discipline scales to mobile because monospaced digits + narrow
   line height let two columns of numerics be visually compared even at 12 px.
   ([source](https://stephaniewalter.design/blog/what-minimum-font-size-for-a-high-density-data-web-app-do-you-suggest/))
7. **Push notifications and an in-app inbox are separate surfaces.** TradingView,
   Bloomberg, Robinhood all have both. Push is ephemeral / interrupting; the
   inbox is the audit trail. We already have Telegram for push; the dashboard
   needs an alerts/decisions inbox tab, not push.
8. **WebSocket-pushed updates, not pull-to-refresh.** Pull-to-refresh on a
   real-time trading surface is an anti-pattern — the user should never feel
   they need to refresh the BTC price. Stale-state indicators (greying numbers
   when ws disconnects) are the right replacement.
9. **Color is functional language, not decoration.** Green/red for direction,
   grey for stale/closed, amber/yellow for caution, light variants for
   neutrality — Robinhood + Bloomberg both lock these in across the app and
   it carries semantic weight.
10. **Progressive disclosure on rich detail pages.** Webull's "12 tabs"
    underneath a ticker works because each tab is *one* surface, not a wall.
    Our per-ticker page should split: Overview / Signals / Charts / Decisions /
    Accuracy / Notes — never all on one screen.

---

## Patterns we should avoid

1. **Confetti / celebration animations on actions.** The Robinhood case-study
   on its own makes this an industry red flag. Don't celebrate trades or
   signal hits with motion — surface them as data.
   ([source](https://www.cnbc.com/2021/03/31/robinhood-gets-rid-of-confetti-feature-amid-scrutiny-over-gamification.html))
2. **Top-trending / leaderboard surfaces.** Robinhood's "most-traded" list
   surfaces popularity, not edge. We have signals — that *is* the trending
   list, weighted by accuracy. Don't add a duplicate "what's hot" panel.
3. **Bloat-as-product.** Coinbase keeps adding features and keeps degrading
   usability. We have the opposite problem (many existing endpoints) — when
   in doubt, hide. Don't pin a metric to the home screen "because we have it".
4. **Two-axis scrolling without sticky elements.** UXmatters and the design
   bootcamp article both flag dual-axis scroll as a major mobile anti-pattern.
   If the heatmap scrolls horizontally, the leftmost column (ticker name)
   MUST be sticky; if it scrolls vertically the header MUST be sticky.
   ([source](https://www.uxmatters.com/mt/archives/2020/07/designing-mobile-tables.php))
5. **Hamburger menu as primary nav.** Material and Apple HIG both recommend
   3–5 bottom tabs over a hamburger because hamburger destinations get
   ~10× less use. Avanza-style "Mer" (More) catch-all is fine for tertiary;
   primary destinations must be visible.
6. **Aspirational whitespace at the cost of density.** The "white space
   killed an enterprise app" piece argues power users punish sparse layouts.
   Don't pad signal rows so they only show 3 instruments per screen — show 8.
   ([source](https://uxdesign.cc/how-white-space-killed-an-enterprise-app-and-why-data-density-matters-b3afad6a5f2a))
7. **Notification fatigue / mixed-purpose inbox.** Webull's inbox conflates
   social noise + market alerts + system messages. Either separate them or
   ruthlessly prune. For the finance-analyzer: trades, alerts, system
   warnings are three different streams.
8. **Mobile-as-clipped-desktop.** Tastytrade's curve analysis on mobile is the
   case study. Some surfaces just don't fit a phone — link out to a
   "view full analysis on desktop" with a copy-link affordance, don't try
   to cram a 2D parameter sweep into 390 px.
9. **Pull-to-refresh as primary update mechanism.** Real-time trading data
   should push. Pull-to-refresh implies "data was wrong before".
10. **Multi-step order tickets reproducing Avanza.** We don't execute trades —
    we generate signals + decisions for the user to act on (in Avanza). Don't
    build a fake order ticket. Show the recommended size + entry + stop and
    a "copy to Avanza" button.

---

## Mobile patterns specifically for dense data

### A. The 33-signal heatmap on a 390 px screen

**The cleanest answer (synthesised from Webull, Bloomberg, and TradingView):**
**a transposed, sticky-leftmost, color-cell-only grid with progressive disclosure.**

Concretely:

1. **Pivot the matrix.** Rows = signals (33), columns = timeframes (~7).
   Display *one ticker at a time*. To switch ticker, swipe horizontally on the
   header, or use a sticky chip-bar at the top. (BTC | ETH | XAU | XAG | MSTR.)
   This is the same pattern MetaTrader uses for "swipe between asset classes".
2. **Leftmost column is sticky** with the signal name truncated to ~10–12 chars
   (`RSI(14)`, `MACD`, `EMA(9,21)`, `OnChain`, etc.). This lets the user scroll
   vertically through 33 rows without losing label.
3. **Cells are color-only.** No numbers in the cell — just a background:
   strong-green / weak-green / neutral-grey / weak-red / strong-red, encoding
   {STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL}. Each cell ~38×24 px gives
   ~7 columns × 14 visible rows = a screenful you can read in one glance.
4. **Header row is sticky** with the timeframe (`5m | 15m | 1h | 4h | 1d | 3d | 1w`).
5. **Long-press a cell** opens a bottom sheet with: signal name (full),
   timeframe, current vote, confidence %, recent accuracy %, recent
   sample size, and a 3-line plain-language rationale. Bottom sheet is
   dismissable with a swipe-down. (TradingView's long-press tracking pattern.)
6. **Tap a row label** filters the heatmap to that one signal across all
   tickers — i.e. a "deep-dive on this signal" view.
7. **Above the grid:** a single "consensus chip" per ticker — STRONG_BUY /
   BUY / HOLD / SELL / STRONG_SELL with the confidence number.
8. **Disabled signals** (force-HOLD via DISABLED_SIGNALS) are striped grey,
   not coloured — keeps "not voting" visually distinct from "voting HOLD".
9. **Sparklines: no.** A 38 px wide cell is too small for a meaningful trend.
   The rationale text in the bottom sheet replaces it.

**Why this works:** the eye reads color faster than numbers. 14 visible rows ×
7 cols = 98 cells of information per screen — beats anything text-based. The
sticky-leftmost-column pattern is well-supported on mobile and avoids the
two-axis-scroll anti-pattern (vertical scroll only).
([source](https://www.uxmatters.com/mt/archives/2020/07/designing-mobile-tables.php),
[source](https://www.datawrapper.de/blog/sticky-table-columns))

**Alternative considered + rejected:** showing all 5 tickers as columns. That
caps you at 33 rows × 5 tickers × no-room-for-timeframe — loses the timeframe
dimension. Pivoting to one-ticker-at-a-time is the unlock.

### B. Decision history table with 8–10 columns

You cannot show 8–10 columns on a 390 px screen as a true table. The
conventional answers from the design literature:
([source](https://medium.com/design-bootcamp/designing-user-friendly-data-tables-for-mobile-devices-c470c82403ad),
[source](https://www.uxmatters.com/mt/archives/2020/07/designing-mobile-tables.php))

1. **Card-row pattern (recommended).** Each decision = one card. Card header:
   timestamp + ticker + decision verdict (BUY/SELL/HOLD) + P&L if closed.
   Card body (collapsed): 1–2 priority columns (size + confidence). Tap to
   expand → all 10 columns visible as a vertical key-value list within the
   card. This avoids horizontal scroll entirely.
2. **Concatenated columns.** Combine `entry_price + entry_time` into one cell
   ("$108,432 · 14:23"), `confidence + accuracy_window` into one ("78% · 30d").
   Reduces 10 cols to 5–6.
3. **Filter strip on top.** Instead of showing all decisions, filter chips:
   `Today | Patient | Bold | Wins | Losses`. Power-user equivalent of column
   sort.
4. **Sort by tap on the header is supported** on the expanded view.

For our case, the decision history is read-mostly and chronological — the
card-list pattern with a filter strip is the better fit than a sticky-column
table.

### C. Multi-line charts with annotation

Lessons from TradingView + Webull:

1. **One chart fills 60–70% of the viewport when in focus.** The legend is
   terse; long-press surfaces full OHLC + indicator values.
2. **Annotations as inline icons at the X-axis position they refer to.**
   Tap an icon to open a bottom sheet with the annotation text. Don't try
   to render annotation text on the chart itself — it doesn't fit.
3. **Multi-series toggling** via chips below the chart — each chip is a
   series legend that doubles as a toggle.
4. **Time-period pill row at the bottom** (1H | 4H | 1D | 1W | 1M | YTD).
   Adopted universally.
5. **Pinch + slide-on-axis** for zoom/scroll. Double-tap-axis to auto-fit.
6. **Crosshair on long-press** — the value readout follows the crosshair to
   a fixed corner of the chart; never floats with the finger (which would be
   blocked by the touch).

### D. Real-time numeric tickers with up/down momentum

The Webull / Bloomberg / IBKR approach:

1. **Color-flash on update.** Each price update briefly flashes the cell
   green (up-tick) or red (down-tick) for ~250 ms, then reverts to base color.
   Encodes momentum without consuming additional space.
2. **Monospaced numerics.** Even at 12–14 px, monospaced digits stay
   column-aligned, which lets the eye compare rows.
3. **Decimal alignment.** $108,432.51 above $97.13 should align on the
   decimal point. CSS `font-variant-numeric: tabular-nums;` is the magic.
4. **Stale indicator on disconnect.** When the websocket disconnects, the
   number greys out + a small "•" pulses. On reconnect, restore color.
   This is the right replacement for pull-to-refresh.
   ([source](https://medium.com/@emily19980210/optimizing-real-time-market-data-feeds-a-python-websocket-approach-for-us-stocks-f141781752fb))
5. **Day change as percent + absolute, side-by-side.** "+1.34% / +1,432 SEK"
   gives both ratio and amount; reduces cognitive cost of computing.
6. **Tap-to-expand row** to reveal: 24h high/low, volume, last update time,
   exchange. (Robinhood-style card expansion.)

### E. Accuracy drill-down

This is finance-analyzer-specific (Bloomberg / IBKR equivalents are read-only
P&L charts). Suggested pattern:

1. Top: sticky chip-bar of horizon (1d | 3d | 5d | 10d).
2. Mid: a vertical bar list, one row per signal, sorted by accuracy desc.
   Row = signal name + bar (color-coded over/under 47%) + percentage + sample
   size. Bottom-zone-friendly.
3. Tap a row → bottom sheet with calibration plot + recent samples.
4. Filter chip "show only force-HOLD" / "show only active" toggles the
   greying. ([source](https://primer.style/product/ui-patterns/progressive-disclosure/))

---

## Cross-app convergence summary

| Pattern                         | TradingView | Coinbase | Robinhood | Webull | Bloomberg | IBKR | Avanza | Tastytrade |
|---------------------------------|:-----------:|:--------:|:---------:|:------:|:---------:|:----:|:------:|:----------:|
| Bottom tab bar (4–5 dest.)      | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Sparklines inline in lists      | ✓ |   |   | ✓ | ✓ | ✓ |   | ✓ |
| Long-press for crosshair / detail | ✓ |   |   | ✓ | ✓ | ✓ |   | ✓ |
| Bottom sheet drill-down         | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Density toggle / pref           |   |   |   |   |   | ✓ |   | ✓ |
| Push + in-app inbox separate    | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| WebSocket-pushed price          | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Pull-to-refresh primary         |   |   |   |   |   |   |   |   |
| Confetti / celebration          |   |   | (removed) |   |   |   |   |   |
| Trending / popularity feed      |   | ✓ | ✓ | ✓ |   |   |   |   |
| Signal heatmap on phone         |   |   |   | ✓ (mkt) |   |   |   |   |
| User-rearrangeable nav          |   |   |   |   |   |   |   | ✓ |

---

## Sources

- TradingView mobile docs: <https://www.tradingview.com/charting-library-docs/latest/mobile_specifics/>
- TradingView multi-chart on mobile: <https://www.tradingview.com/blog/en/multi-chart-view-on-a-mobile-screen-10924/>
- TradingView alerts setup: <https://www.tradingview.com/support/solutions/43000595315-how-to-set-up-alerts/>
- TradingView Trustpilot reviews: <https://www.trustpilot.com/review/tradingview.com>
- TradingView app store reviews: <https://justuseapp.com/en/app/1205990992/tradingview-track-all-markets/reviews>
- TradingView mobile guide: <https://www.financialtechwiz.com/post/tradingview-mobile-app/>
- Coinbase Advanced dashboard: <https://help.coinbase.com/en/coinbase/trading-and-funding/advanced-trade/dashboard-overview>
- Coinbase UX redesign case study: <https://jpux.medium.com/case-study-coinbase-ux-redesign-9fa4038f5d52>
- Coinbase activation funnel: <https://medium.com/the-plg-insider/cryptos-user-activation-crisis-a-product-case-study-on-coinbase-s-activation-funnel-e2a21b6eef48>
- Robinhood UI balance: <https://worldbusinessoutlook.com/how-the-robinhood-ui-balances-simplicity-and-strategy-on-mobile/>
- Robinhood gamification (Tearsheet): <https://tearsheet.co/marketing/the-double-edged-sword-of-good-ux-how-robinhoods-gamification-of-investing-backfired-during-the-market-downturn/>
- Robinhood confetti removal (CNBC): <https://www.cnbc.com/2021/03/31/robinhood-gets-rid-of-confetti-feature-amid-scrutiny-over-gamification.html>
- Robinhood gamification analysis (Finmasters): <https://finmasters.com/gamification-of-investing/>
- Robinhood behavioural economics: <https://www.economicsonline.co.uk/all/behavioral-economics-and-the-gamification-of-finance-how-apps-like-robinhood-influence-trading-behavior.html/>
- Webull review (StockBrokers.com): <https://www.stockbrokers.com/review/webull>
- Webull notifications complaints: <https://apps.apple.com/us/app/webull-investing-trading/id1179213067?see-all=reviews>
- Bloomberg how Terminal designers conceal complexity: <https://www.bloomberg.com/company/stories/how-bloomberg-terminal-ux-designers-conceal-complexity/> (gated, summary only)
- Bloomberg Terminal density discussion (HN): <https://news.ycombinator.com/item?id=19153875>
- Bloomberg Terminal makeover (UX Magazine): <https://uxmag.com/articles/the-impossible-bloomberg-makeover>
- Bloomberg Professional app reviews (App Store): <https://apps.apple.com/us/app/bloomberg-professional/id407761767>
- IBKR Mobile vs GlobalTrader: <https://thepoorswiss.com/ibkr-global-trader/>
- IBKR mobile platform overview: <https://www.interactivebrokers.com/en/trading/ibkr-mobile.php>
- 10 best mobile trading apps 2026: <https://www.newtrading.io/mobile-trading/>
- Avanza Trustpilot: <https://www.trustpilot.com/review/avanza.se>
- Tastytrade mobile platform: <https://tastytrade.com/mobile-platform/>
- Tastytrade mobile overview: <https://support.tastytrade.com/support/s/solutions/articles/43000435224>
- Tastytrade review (StockBrokers.com): <https://www.stockbrokers.com/review/tastytrade>
- Tastytrade mobile app rewrite: <https://devexperts.com/app/uploads/case-studies/Tasty_trade.pdf>
- MetaTrader 5 mobile: <https://www.metaquotes.net/en/metatrader5/mobile-trading>
- Smashing Magazine — thumb zone: <https://www.smashingmagazine.com/2016/09/the-thumb-zone-designing-for-mobile-users/>
- Smashing Magazine — bottom navigation pattern: <https://www.smashingmagazine.com/2019/08/bottom-navigation-pattern-mobile-web-pages/>
- UXmatters — designing mobile tables: <https://www.uxmatters.com/mt/archives/2020/07/designing-mobile-tables.php>
- Design Bootcamp — mobile data tables: <https://medium.com/design-bootcamp/designing-user-friendly-data-tables-for-mobile-devices-c470c82403ad>
- Datawrapper — sticky table columns: <https://www.datawrapper.de/blog/sticky-table-columns>
- NN/G — bottom sheets: <https://www.nngroup.com/articles/bottom-sheet/>
- UXmatters — progressive disclosure: <https://www.uxmatters.com/mt/archives/2020/05/designing-for-progressive-disclosure.php>
- Stephanie Walter — minimum font size for high-density data apps: <https://stephaniewalter.design/blog/what-minimum-font-size-for-a-high-density-data-web-app-do-you-suggest/>
- UX Design — white space killed an enterprise app: <https://uxdesign.cc/how-white-space-killed-an-enterprise-app-and-why-data-density-matters-b3afad6a5f2a>
- Material — bottom sheets: <https://m2.material.io/components/sheets-bottom>
- Primer — progressive disclosure: <https://primer.style/product/ui-patterns/progressive-disclosure/>
- WebSocket vs polling for real-time market data: <https://medium.com/@emily19980210/optimizing-real-time-market-data-feeds-a-python-websocket-approach-for-us-stocks-f141781752fb>
- NN/G — mobile carousels: <https://www.nngroup.com/articles/mobile-carousels/>
