"""Writer citation resolution (§7).

The writer sees evidence as ``[1] (source) ...`` and the model tends to cite by the
bracket index, which renders as meaningless "1, 2, 3" sources. `_resolve_citations`
maps those indices back to the real evidence sources so the brief is genuinely cited.
"""

from maestro.agents.writer import _resolve_citations
from maestro.state import Evidence


def _ev(source: str) -> Evidence:
    return Evidence(subtask_id="r1", source=source, content="x")


def test_numeric_citations_map_to_sources():
    evidence = [_ev("https://a.com"), _ev("https://b.com"), _ev("https://c.com")]
    assert _resolve_citations(["1", "3"], evidence) == ["https://a.com", "https://c.com"]


def test_bracketed_indices_and_real_sources_dedup():
    evidence = [_ev("src_solar"), _ev("src_wind")]
    # "[1]" resolves to src_solar; "src_wind" passes through; "1" is a dup -> dropped
    assert _resolve_citations(["[1]", "src_wind", "1"], evidence) == ["src_solar", "src_wind"]


def test_already_real_sources_unchanged():
    evidence = [_ev("src_a"), _ev("src_b")]
    assert _resolve_citations(["src_a", "src_b"], evidence) == ["src_a", "src_b"]


def test_unrecognized_labels_kept_as_last_resort():
    evidence = [_ev("src_a")]
    assert _resolve_citations(["src_a", "manual-note"], evidence) == ["src_a", "manual-note"]
