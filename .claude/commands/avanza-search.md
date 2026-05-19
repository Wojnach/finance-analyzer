Search for instruments on Avanza by name, ticker, or ISIN.

Usage: /avanza-search <query>

Run this to search:

```
.venv/Scripts/python.exe -c "
import sys, json
sys.path.insert(0, '.')
from portfolio.avanza_session import api_post
result = api_post('/_api/search/filtered-search', {'query': '$ARGUMENTS', 'limit': 20})
if not result:
    print('No results or session not active.')
else:
    hits = result.get('hits', [])
    total = result.get('totalNumberOfHits', 0)
    print(f'Found {total} results (showing {len(hits)}):')
    print()
    print(f'{\"OB_ID\":>10s}  {\"Type\":<14s}  {\"Price\":>10s}  {\"Spread\":>8s}  Name')
    print('-' * 80)
    for h in hits:
        name = h.get('title', '')
        ob_id = str(h.get('orderBookId', ''))
        itype = h.get('type', '?')
        p = h.get('price', {})
        price = p.get('last', '')
        spread = p.get('spread', '') or ''
        chg = p.get('todayChangePercent', '') or ''
        print(f'{ob_id:>10s}  {itype:<14s}  {price:>10s}  {spread:>8s}  {name}')
"
```

Show results in a clean table. The OB_ID (orderbook ID) is what's needed for trading via `avanza_orders.py`.
