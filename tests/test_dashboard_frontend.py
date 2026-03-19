from pathlib import Path

INDEX_HTML = Path("dashboard/static/index.html").read_text(encoding="utf-8")


def test_frontend_propagates_query_token_to_api_requests():
    assert 'new URLSearchParams(window.location.search).get("token")' in INDEX_HTML
    assert "function _withApiToken(apiUrl)" in INDEX_HTML
    assert "var r = await fetch(liveUrl);" in INDEX_HTML


def test_frontend_applies_static_filters_for_messages_and_decisions():
    assert "function _applyStaticFilters(apiUrl, data)" in INDEX_HTML
    assert 'if (path === "/api/telegrams")' in INDEX_HTML
    assert 'if (path === "/api/decisions")' in INDEX_HTML


def test_frontend_does_not_eager_load_accuracy_on_boot():
    init_section = INDEX_HTML.split("/* ================================================================\n         Init", 1)[1]
    assert "refresh();" in init_section
    assert "startCd();" in init_section
    assert "loadAccuracy();" not in init_section
    assert "loadAccuracyHistory();" not in init_section


def test_frontend_supports_sorting_portfolio_accuracy_signals():
    assert 'var _accuracySort = "pct_desc";' in INDEX_HTML
    assert "function setAccuracySort(mode)" in INDEX_HTML
    assert "function _sortAccuracyEntries(entries, mode)" in INDEX_HTML
    assert 'Sort portfolio-loop signals for each horizon panel' in INDEX_HTML
    assert 'Accuracy desc' in INDEX_HTML
