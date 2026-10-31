"""Observation-only draw classifier for raw score-model output.

The classifier never changes probabilities or the raw H/D/A prediction. Its
output is diagnostic metadata for acceptance evaluation and later BuyPlan use.
Thresholds in v0 are explicit hypotheses, not fitted production rules.
"""
from __future__ import annotations

from dataclasses import dataclass

from common import number


FILTER_VERSION = "draw_filter_observation_v0"
FILTER_VERSION_V1 = "draw_filter_score_structure_v1"
FILTER_VERSION_V2 = "draw_filter_score_plus_legacy_structure_v2"
FALSE_DRAW_TYPES = {"j1_false_draw_watch", "j2_false_draw_watch"}
TRAP_TYPES = {"draw_trap", "j2_draw_trap"}
REJECT_BASIS = {"draw_compressed", "flat_draw_trap"}


@dataclass(frozen=True)
class DrawAssessment:
    draw_class: str
    confidence: str
    fallback_side: str
    reason: str


def _raw_rank(row) -> tuple[str, float, float]:
    probabilities = {
        "H": number(row.get("next_prob_home"), 0.0),
        "D": number(row.get("next_prob_draw"), 0.0),
        "A": number(row.get("next_prob_away"), 0.0),
    }
    ordered = sorted(probabilities, key=lambda symbol: (-probabilities[symbol], "HDA".index(symbol)))
    return ordered[0], probabilities[ordered[0]], probabilities[ordered[1]]


def classify_draw(row) -> DrawAssessment:
    raw_symbol, top_probability, second_probability = _raw_rank(row)
    p_home = number(row.get("next_prob_home"), 0.0)
    p_draw = number(row.get("next_prob_draw"), 0.0)
    p_away = number(row.get("next_prob_away"), 0.0)
    side = "H" if p_home >= p_away else "A"
    side_probability = max(p_home, p_away)
    draw_gap = p_draw - side_probability
    lambda_total = number(row.get("final_lambda_total"), number(row.get("base_lambda_total"), 99.0))
    tied_score_probability = (
        number(row.get("next_prob_0_0"), 0.0)
        + number(row.get("next_prob_1_1"), 0.0)
        + number(row.get("next_prob_2_2"), 0.0)
    )
    purchase_type = str(row.get("match_purchase_type") or "").strip()
    basis = str(row.get("lab_basis_hint") or "").strip()
    flip = number(row.get("lab_flip_weight"), 0.0)
    entropy = number(row.get("lab_scenario_entropy_score"), 0.0)

    evidence = (
        f"raw={raw_symbol}|pD={p_draw:.3f}|side={side}:{side_probability:.3f}|"
        f"gap={draw_gap:.3f}|lambda_total={lambda_total:.3f}|"
        f"p00+p11+p22={tied_score_probability:.3f}|type={purchase_type or '-'}|"
        f"basis={basis or '-'}|flip={flip:.3f}|entropy={entropy:.3f}"
    )

    if raw_symbol == "D":
        if purchase_type in FALSE_DRAW_TYPES | TRAP_TYPES or basis in REJECT_BASIS:
            return DrawAssessment("D-reject", "watch", side, f"legacy_false_draw_or_trap|{evidence}")
        if lambda_total <= 2.35 and draw_gap >= 0.025 and tied_score_probability >= 0.26:
            return DrawAssessment("D-core", "candidate", side, f"low_event_score_support|{evidence}")
        return DrawAssessment("D-uncertain", "candidate", side, f"raw_draw_without_core_support|{evidence}")

    close_to_top = top_probability - p_draw <= 0.10
    if p_draw >= 0.25 and close_to_top and lambda_total <= 2.75:
        return DrawAssessment("D-cover", "candidate", side, f"side_first_with_draw_cover|{evidence}")
    return DrawAssessment("not_applicable", "none", side, f"draw_not_actionable|{evidence}")


def classify_draw_v1(row) -> DrawAssessment:
    """Four-way score-structure classifier with no legacy class inputs.

    Thresholds are frozen design hypotheses for forward evaluation. They were
    not selected to produce a target draw count or fitted against actuals.
    """
    p_home = number(row.get("next_prob_home"), 0.0)
    p_draw = number(row.get("next_prob_draw"), 0.0)
    p_away = number(row.get("next_prob_away"), 0.0)
    side = "H" if p_home >= p_away else "A"
    side_probability = max(p_home, p_away)
    lambda_home = number(row.get("final_lambda_home"), number(row.get("base_lambda_home"), 0.0))
    lambda_away = number(row.get("final_lambda_away"), number(row.get("base_lambda_away"), 0.0))
    lambda_total = lambda_home + lambda_away
    lambda_balance = abs(lambda_home - lambda_away) / max(lambda_total, 1e-9)
    tied_score_probability = sum(
        number(row.get(column), 0.0)
        for column in ("next_prob_0_0", "next_prob_1_1", "next_prob_2_2")
    )
    gap_to_side = side_probability - p_draw
    evidence = (
        f"pD={p_draw:.3f}|side={side}:{side_probability:.3f}|gap={gap_to_side:.3f}|"
        f"lambdaH={lambda_home:.3f}|lambdaA={lambda_away:.3f}|"
        f"lambda_total={lambda_total:.3f}|lambda_balance={lambda_balance:.3f}|"
        f"p00+p11+p22={tied_score_probability:.3f}"
    )
    if (
        p_draw >= 0.28 and lambda_total <= 2.35 and lambda_balance <= 0.15
        and tied_score_probability >= 0.27
    ):
        return DrawAssessment("D-core", "candidate", side, f"balanced_low_event|{evidence}")
    if p_draw >= 0.25 and gap_to_side <= 0.10 and lambda_total <= 2.75:
        return DrawAssessment("D-cover", "candidate", side, f"side_first_close_draw|{evidence}")
    if p_draw < 0.22 or lambda_total >= 3.20 or lambda_balance >= 0.35:
        return DrawAssessment("D-reject", "none", side, f"weak_or_open_or_imbalanced|{evidence}")
    return DrawAssessment("D-uncertain", "watch", side, f"mixed_score_structure|{evidence}")


def classify_draw_v2(row) -> DrawAssessment:
    """Add narrow historical matchup-structure guards to score v1.

    Legacy labels never change probabilities and cannot create D-core. Trap
    structures can reject D; split-side can only promote uncertain to cover.
    """
    base = classify_draw_v1(row)
    basis = str(row.get("lab_basis_hint") or "").strip()
    p_draw = number(row.get("next_prob_draw"), 0.0)
    side_probability = max(
        number(row.get("next_prob_home"), 0.0),
        number(row.get("next_prob_away"), 0.0),
    )
    lambda_home = number(row.get("final_lambda_home"), number(row.get("base_lambda_home"), 0.0))
    lambda_away = number(row.get("final_lambda_away"), number(row.get("base_lambda_away"), 0.0))
    lambda_total = lambda_home + lambda_away
    structure = f"basis={basis or '-'}|score_v1={base.draw_class}"

    if basis in REJECT_BASIS:
        return DrawAssessment(
            "D-reject", "none", base.fallback_side,
            f"legacy_trap_structure_guard|{structure}|{base.reason}",
        )
    if (
        basis == "split_side" and base.draw_class == "D-uncertain"
        and p_draw >= 0.24 and side_probability - p_draw <= 0.15
        and lambda_total <= 2.85
    ):
        return DrawAssessment(
            "D-cover", "candidate", base.fallback_side,
            f"split_side_cover_only|{structure}|{base.reason}",
        )
    return DrawAssessment(
        base.draw_class, base.confidence, base.fallback_side,
        f"score_v1_kept|{structure}|{base.reason}",
    )
