"""Makes `python -m portfolio.swedbank` work, as CLAUDE.md documents and as
`cli.py`'s own `prog=` string already claimed.
"""

from portfolio.swedbank.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
