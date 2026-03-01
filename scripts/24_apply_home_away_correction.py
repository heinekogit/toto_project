import os
import numpy as np
import pandas as pd


def _normalize_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.clip(probs.astype(float), 0.0, None)
    s = probs.sum()
    if s <= 0:
        return np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0], dtype=float)
    return probs / s


def apply_probability_multipliers(
    home_win_prob: float,
    draw_prob: float,
    away_win_prob: float,
    multipliers: tuple[float, float, float],
) -> tuple[float, float, float]:
    probs = _normalize_probs(np.array([home_win_prob, draw_prob, away_win_prob], dtype=float))
    mult = np.clip(np.array(multipliers, dtype=float), 0.0, None)
    corrected = _normalize_probs(probs * mult)
    return float(corrected[0]), float(corrected[1]), float(corrected[2])


def clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, float(value)))


def compute_delta_from_home_advantage_diff(
    home_advantage_diff: float,
    k: float = 0.03,
    max_abs_delta: float = 0.05,
) -> float:
    raw = float(home_advantage_diff) * float(k)
    limit = abs(float(max_abs_delta))
    return clamp(raw, -limit, limit)


def apply_home_advantage_diff_correction(
    home_win_prob: float,
    draw_prob: float,
    away_win_prob: float,
    home_advantage_diff: float,
    k: float = 0.03,
    max_abs_delta: float = 0.05,
    home_weight: float = 0.6,
    draw_weight: float = 0.3,
    away_weight: float = -0.9,
) -> tuple[float, float, float]:
    """
    差分ベース補正:
      delta = clamp(home_advantage_diff * k, -max_abs_delta, +max_abs_delta)
      pH += delta * home_weight
      pD += delta * draw_weight
      pA += delta * away_weight
      -> 最後に正規化
    """
    p_h, p_d, p_a = _normalize_probs(np.array([home_win_prob, draw_prob, away_win_prob], dtype=float))
    delta = compute_delta_from_home_advantage_diff(
        home_advantage_diff=home_advantage_diff,
        k=k,
        max_abs_delta=max_abs_delta,
    )
    p_h += delta * float(home_weight)
    p_d += delta * float(draw_weight)
    p_a += delta * float(away_weight)
    p_h, p_d, p_a = _normalize_probs(np.array([p_h, p_d, p_a], dtype=float))
    return float(p_h), float(p_d), float(p_a)


def _home_away_base_multipliers(target_side: str, delta: float) -> tuple[float, float, float]:
    side = str(target_side).strip().lower()
    d = max(float(delta), 0.0)
    if side not in {"home", "away"}:
        raise ValueError("target_side は 'home' または 'away' を指定してください。")
    # 仕様: home/away いずれも away勝率を下げて drawへ寄せる
    # (away視点では away勝率が「勝利確率」に相当)
    return 1.0, 1.0 + d, max(1.0 - d, 0.0)


def _argmax_index(probs: tuple[float, float, float]) -> int:
    return int(np.argmax(np.array(probs, dtype=float)))


def _scale_delta_to_keep_argmax(
    home_win_prob: float,
    draw_prob: float,
    away_win_prob: float,
    target_side: str,
    delta: float,
) -> float:
    base_probs = _normalize_probs(np.array([home_win_prob, draw_prob, away_win_prob], dtype=float))
    base_idx = int(np.argmax(base_probs))
    lo, hi = 0.0, max(float(delta), 0.0)

    # 二分探索で「argmaxが変わらない最大delta」を探す
    for _ in range(30):
        mid = (lo + hi) / 2.0
        multipliers = _home_away_base_multipliers(target_side=target_side, delta=mid)
        corrected = apply_probability_multipliers(
            home_win_prob=base_probs[0],
            draw_prob=base_probs[1],
            away_win_prob=base_probs[2],
            multipliers=multipliers,
        )
        if _argmax_index(corrected) == base_idx:
            lo = mid
        else:
            hi = mid
    return lo


def apply_home_away_correction(
    home_win_prob: float,
    draw_prob: float,
    away_win_prob: float,
    target_side: str,
    delta: float = 0.03,
    preserve_argmax: bool = True,
) -> tuple[float, float, float]:
    """
    [home_win_prob, draw_prob, away_win_prob] を補正する。

    補正ルール:
    - target_side='home' : 敗戦確率(away_win_prob)を減らし、その分drawへ寄せる
    - target_side='away' : 勝利確率(away_win_prob)を減らし、その分drawへ寄せる
    - preserve_argmax=True の場合、ホーム補正単独で最大確率クラスが入れ替わらない範囲にdeltaを自動縮小する
    """
    effective_delta = (
        _scale_delta_to_keep_argmax(
            home_win_prob=home_win_prob,
            draw_prob=draw_prob,
            away_win_prob=away_win_prob,
            target_side=target_side,
            delta=delta,
        )
        if preserve_argmax
        else max(float(delta), 0.0)
    )
    multipliers = _home_away_base_multipliers(target_side=target_side, delta=effective_delta)
    return apply_probability_multipliers(
        home_win_prob=home_win_prob,
        draw_prob=draw_prob,
        away_win_prob=away_win_prob,
        multipliers=multipliers,
    )


def apply_home_away_correction_to_dataframe(
    df: pd.DataFrame,
    side_col: str,
    home_col: str = "prob_home_win",
    draw_col: str = "prob_draw",
    away_col: str = "prob_away_win",
    delta: float = 0.03,
    out_home_col: str = "prob_home_win_adj",
    out_draw_col: str = "prob_draw_adj",
    out_away_col: str = "prob_away_win_adj",
    preserve_argmax: bool = True,
) -> pd.DataFrame:
    # 互換維持: 既存IFのまま残す（新規は差分補正関数を推奨）
    out = df.copy()
    required = [side_col, home_col, draw_col, away_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"必要列が不足しています: {missing}")

    corrected = out.apply(
        lambda r: apply_home_away_correction(
            home_win_prob=r[home_col],
            draw_prob=r[draw_col],
            away_win_prob=r[away_col],
            target_side=r[side_col],
            delta=delta,
            preserve_argmax=preserve_argmax,
        ),
        axis=1,
    )
    out[[out_home_col, out_draw_col, out_away_col]] = pd.DataFrame(corrected.tolist(), index=out.index)
    return out


def apply_home_advantage_diff_correction_to_dataframe(
    df: pd.DataFrame,
    diff_col: str = "home_advantage_diff",
    home_col: str = "prob_home_win",
    draw_col: str = "prob_draw",
    away_col: str = "prob_away_win",
    k: float = 0.03,
    max_abs_delta: float = 0.05,
    home_weight: float = 0.6,
    draw_weight: float = 0.3,
    away_weight: float = -0.9,
    out_home_col: str = "prob_home_win_adj",
    out_draw_col: str = "prob_draw_adj",
    out_away_col: str = "prob_away_win_adj",
) -> pd.DataFrame:
    out = df.copy()
    required = [diff_col, home_col, draw_col, away_col]
    missing = [c for c in required if c not in out.columns]
    if missing:
        raise ValueError(f"必要列が不足しています: {missing}")

    corrected = out.apply(
        lambda r: apply_home_advantage_diff_correction(
            home_win_prob=r[home_col],
            draw_prob=r[draw_col],
            away_win_prob=r[away_col],
            home_advantage_diff=r[diff_col],
            k=k,
            max_abs_delta=max_abs_delta,
            home_weight=home_weight,
            draw_weight=draw_weight,
            away_weight=away_weight,
        ),
        axis=1,
    )
    out[[out_home_col, out_draw_col, out_away_col]] = pd.DataFrame(corrected.tolist(), index=out.index)
    return out


def main() -> None:
    # 例: 過去検証CSVに target_side='home' を固定で適用
    input_csv = os.environ.get("INPUT_CSV", "backtest_j1_2025_rounds.csv")
    output_csv = os.environ.get("OUTPUT_CSV", "data/j1_2025_probs_home_away_corrected.csv")
    k = float(os.environ.get("K", "0.03"))
    max_abs_delta = float(os.environ.get("MAX_ABS_DELTA", "0.05"))
    diff_col = os.environ.get("DIFF_COL", "home_advantage_diff")

    df = pd.read_csv(input_csv)
    out_df = apply_home_advantage_diff_correction_to_dataframe(
        df=df,
        diff_col=diff_col,
        k=k,
        max_abs_delta=max_abs_delta,
    )
    out_df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"出力: {output_csv}")
    print(f"rows: {len(out_df)}")


if __name__ == "__main__":
    main()
