# Dashboard Terminal Reskin — Design Spec

**Date:** 2026-05-22
**Branch:** `dashboard-terminal-reskin`
**Scope:** Visual reskin of the Flask dashboard (`dashboard/static/`) to a
retro trading-terminal aesthetic. CSS-led, no API/behavior changes.

## Inspiration

User-supplied reference: "BONEREAPER // POLYMARKET BOT" terminal dashboard.
Key traits to reproduce:

- **Cream-paper panels** with **hard black borders** (square corners).
- **Monospace typography everywhere** — tabular numerics, uppercase labels.
- **Bracketed panel headers** — `[ WALLET ]`, `◆ STRATEGY` style.
- **Phosphor palette** — vivid green = profit/active, alert red = loss,
  amber = pending/caution, black = structure.
- **Boxed status chips** — `LIVE`, `SCANNING`, `ACTIVE` as bright-fill
  rectangular pills.
- **Dense panels** — tight padding, hard dividers, ledger feel.
- Dark header bar with light text.

## Out of scope

- The animated decision-tree **flow-viz panel** (separate future build).
- Any REST API / endpoint / data change.
- New views or routes.
- Light-theme polish beyond the cream-paper palette swap.

## Approach

The CSS is already fully token-driven (`tokens.css` → consumed by
`base/layout/components/responsive.css`; `chart-config.js` reads tokens via
`getChartColors()`). So the reskin splits cleanly:

- **Palette** lives in `tokens.css` — two themes, both terminal-styled.
- **Chrome** (borders, brackets, mono, chips, square corners) lives in
  `base.css` + `components.css` + `layout.css` — theme-agnostic, keyed off
  tokens.

Two palettes, shared chrome:

- **`light` = cream-paper** — matches the inspiration screenshot. Default
  recommendation for the user (toggle in header).
- **`dark` = dark-CRT** — phosphor-on-black night variant, same chrome.

No third theme. No default-logic change in `theme.js` (YAGNI) — the existing
toggle + `localStorage("pi-theme")` persistence stands.

## Palette tokens (`tokens.css`)

### `html.light` — cream paper
```
--bg:    #d9d3c0   --bg2:   #cfc9b4
--card:  #ebe6d6   --card2: #e1dcc9   --hover: #d4cdb6
--bdr:   #15140f   --bdr2:  #15140f          (hard black borders)
--tx:    #15140f   --txd:   #4a463a   --txm:  #6b6657
--grn:   #13a04a   --red:   #d22f2f   --yel:  #c08a14
--org:   #cc6a16   --blu:   #1f6fb0   --cyn:  #157f8c
```

### `:root` (dark) — dark CRT
```
--bg:    #080b08   --bg2:   #0b0f0b
--card:  #0e130e   --card2: #131a13   --hover: #1a241a
--bdr:   #234023   --bdr2:  #356035
--tx:    #cfe8cf   --txd:   #7da37d   --txm:  #557055
--grn:   #00ff88   --red:   #ff4444   --yel:  #ffc233
--org:   #ff8c2b   --blu:   #38bdf8   --cyn:  #2dd4bf
```

### Both themes
- `--ty-stack` → monospace stack (same value as `--ty-mono`). Mono becomes
  the document default font.
- Radii squared: `--rad-sm/md/lg` → `0`; `--rad-pill` → `2px` (chips are
  rectangular boxes, not capsules).
- Heatmap / flash / `--hm-*` rgba scales re-derived from the new accent hues.

## Chrome checklist

### `base.css`
- [ ] `body` uses mono stack, terminal bg, tabular-nums.
- [ ] Scrollbars: square thumb, `--bdr2` colored, no pill radius.
- [ ] `:focus-visible` ring uses `--grn` (was `--cyn`).

### `components.css`
- [ ] `.card` — square corners, hard `1px solid var(--bdr)` border,
      `--card` bg; hover lifts border to accent, not radius.
- [ ] `.card__title` — uppercase, letter-spaced, bracketed via
      `::before { content: "[ " }` + `::after { content: " ]" }`.
- [ ] `.section-title` — keep the leading bar but square it; uppercase mono.
- [ ] `.chip` / `.badge` — rectangular (2px radius), uppercase, hard
      border, mono. `.chip.active` + status variants get bright accent fill
      with contrasting text.
- [ ] `.badge--BUY/SELL/HOLD` — terminal accent fills.
- [ ] `.pulse-dot` — square `2px` not circle; keep ok/warn/fail/idle colors.
- [ ] `.accordion`, `.banner`, `.heatmap-wrap`, `.bottom-sheet` — square
      corners, hard borders.
- [ ] Status-pill helper for `LIVE` / `SCANNING` / `ACTIVE` — bright fill,
      uppercase, mono, 2px box.

### `layout.css`
- [ ] `.app-header` — dark bar (both themes use the dark band), mono brand,
      phosphor accent, square.
- [ ] `.bottom-nav` — hard top border, square active-item highlight box.
- [ ] Grid/section gaps tightened to the dense ledger feel.

### `charts/chart-config.js`
- [ ] Chart font family → mono. Grid lines → dotted/dim. Colors already
      flow from `getChartColors()` — verify they pick up the new tokens.

### `responsive.css`
- [ ] Re-check breakpoints still hold after chrome changes; mobile-first
      layout must not regress.

## Verification (loop completion gate)

The reskin is **done** when ALL of these hold:

1. Dashboard serves on `:5055` and the SPA boots with **zero console
   errors** (`/static/js/main.js` module graph intact).
2. Playwright screenshots — captured for routes **home, signals,
   decisions, more, health** in **both** `light` and `dark` themes —
   show: square hard-bordered panels, monospace text, bracketed card
   titles, boxed status chips, the cream-paper (light) / phosphor-black
   (dark) palettes.
3. Mobile viewport (≤430px) renders without horizontal overflow or
   clipped chrome.
4. Existing dashboard-related `pytest` (if any under `tests/`) still
   passes — the reskin touches no Python.
5. Every CSS file changed is committed to `dashboard-terminal-reskin`.

## Files touched

| File | Change |
|------|--------|
| `dashboard/static/css/tokens.css` | Two terminal palettes, mono default, squared radii |
| `dashboard/static/css/base.css` | Mono body, square scrollbars, focus ring |
| `dashboard/static/css/components.css` | Square hard-bordered chrome, bracketed titles, boxed chips |
| `dashboard/static/css/layout.css` | Dark header bar, terminal bottom-nav, dense grid |
| `dashboard/static/css/responsive.css` | Breakpoint re-check |
| `dashboard/static/js/charts/chart-config.js` | Mono chart font, dim grid |

No HTML markup changes required — bracketed headers and chips are achieved
via CSS pseudo-elements on existing classes.
