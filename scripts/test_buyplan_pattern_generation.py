#!/usr/bin/env python3
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from buyplan import generate_patterns


def _build_dummy_matches(n: int):
    out = []
    for i in range(n):
        # maxP が単調増加するように作る（low順が安定）
        p_home = 0.42 + i * 0.01
        p_draw = 0.33 - i * 0.004
        p_away = 1.0 - p_home - p_draw
        out.append(
            {
                "match_id": i + 1,
                "prob_home": p_home,
                "prob_draw": p_draw,
                "prob_away": p_away,
            }
        )
    return out


def _diff_count(base_picks, target_picks):
    return sum(1 for b, t in zip(base_picks, target_picks) if b["selected"] != t["selected"])


def test_pattern_count_and_change_count_for_13_matches():
    matches = _build_dummy_matches(13)
    patterns = generate_patterns(matches)
    assert len(patterns) == 10
    for i, p in enumerate(patterns, start=1):
        assert p["pattern_no"] == i
        assert len(p["picks"]) == 13

    base = patterns[0]["picks"]
    # N=13 は midPair が同一点になるため No.10 は 4変更
    expected = [0, 1, 2, 1, 2, 3, 2, 3, 4, 4]
    actual = [_diff_count(base, p["picks"]) for p in patterns]
    assert actual == expected, f"expected={expected} actual={actual}"


def test_pattern10_protects_top2_for_small_n():
    matches = _build_dummy_matches(4)
    patterns = generate_patterns(matches)
    base = patterns[0]["picks"]
    p10 = patterns[9]["picks"]

    # maxP 上位2試合（match_id=4,3）を変更しないこと
    protected_ids = {4, 3}
    for b, t in zip(base, p10):
        if b["match_id"] in protected_ids:
            assert b["selected"] == t["selected"], f"protected changed: {b['match_id']}"


if __name__ == "__main__":
    test_pattern_count_and_change_count_for_13_matches()
    test_pattern10_protects_top2_for_small_n()
    print("ok")
