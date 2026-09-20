import pytest

from scripts.shorten_m3_window_index import shorter_windows


def test_short_windows_deduplicate_overlapping_parents_without_losing_targets():
    source = [(0, 19, 0), (0, 19, 1), (0, 27, 0), (0, 27, 1), (1, 3, 2)]
    actual = list(shorter_windows(source, 65))
    expected = {(episode, start + offset, target) for episode, start, target in source
                for offset in range(0, 57, 8)}
    assert actual == sorted(expected)
    assert len(actual) == 26
    assert {r[0] for r in actual} == {0, 1}
    for episode, start, target in actual:
        assert any(e == episode and t == target and s <= start and start + 9 <= s + 65
                   for e, s, t in source)


def test_short_windows_preserve_gaps_and_identity_context():
    source = [(0, 0, 0), (0, 200, 0)]
    assert list(shorter_windows(source, 9)) == source
    shorter = list(shorter_windows(source, 17))
    assert shorter == [(0, 0, 0), (0, 8, 0), (0, 200, 0), (0, 208, 0)]


def test_short_windows_reject_invalid_context_and_disordered_source():
    with pytest.raises(ValueError, match="contexts"):
        list(shorter_windows([(0, 0, 0)], 9, 65))
    with pytest.raises(ValueError, match="contexts"):
        list(shorter_windows([(0, 0, 0)], 64))
    with pytest.raises(ValueError, match="ordered"):
        list(shorter_windows([(1, 0, 0), (0, 0, 0)], 65))
