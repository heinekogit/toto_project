#!/usr/bin/env python3
import argparse
import os
import unicodedata

import pandas as pd


def _norm_team(v):
    if pd.isna(v):
        return ""
    s = unicodedata.normalize("NFKC", str(v)).strip()
    s = s.replace("　", " ")
    s = s.replace(" ", "")
    s = s.replace("・", "")
    return s.upper()


def _to_result(home_score, away_score):
    if pd.isna(home_score) or pd.isna(away_score):
        return None
    hs = float(home_score)
    aw = float(away_score)
    if hs > aw:
        return "H"
    if hs < aw:
        return "A"
    return "D"


def calc_hda_ratio(series_of_hda):
    s = pd.Series(series_of_hda).astype(str).str.upper().str.strip()
    valid = s[s.isin(["H", "D", "A"])]
    total = int(len(valid))
    h = int((valid == "H").sum())
    d = int((valid == "D").sum())
    a = int((valid == "A").sum())
    denom = total if total > 0 else 1
    return {
        "H": {"count": h, "pct": (h * 100.0 / denom) if total else 0.0},
        "D": {"count": d, "pct": (d * 100.0 / denom) if total else 0.0},
        "A": {"count": a, "pct": (a * 100.0 / denom) if total else 0.0},
        "total": total,
    }


def _fmt_ratio_line(prefix, ratio, with_total=False):
    base = (
        f"[{prefix}] H={ratio['H']['pct']:.1f}% ({ratio['H']['count']}) "
        f"D={ratio['D']['pct']:.1f}% ({ratio['D']['count']}) "
        f"A={ratio['A']['pct']:.1f}% ({ratio['A']['count']})"
    )
    if with_total:
        base += f" total={ratio['total']}"
    return base


def _read_predictions(path, league_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"predictions csv not found: {path}")
    df = pd.read_csv(path)
    need = {"home_team", "away_team"}
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing columns: {missing}")

    work = df.copy()
    work["league_src"] = league_name
    if "match_id" not in work.columns:
        work["match_id"] = pd.NA
    if "predicted_result" not in work.columns:
        work["predicted_result"] = pd.NA
    if "prob_home_win" not in work.columns and "prob_home" in work.columns:
        work["prob_home_win"] = work["prob_home"]
    if "prob_away_win" not in work.columns and "prob_away" in work.columns:
        work["prob_away_win"] = work["prob_away"]
    if "prob_home_win" not in work.columns and "p_home" in work.columns:
        work["prob_home_win"] = work["p_home"]
    if "prob_away_win" not in work.columns and "p_away" in work.columns:
        work["prob_away_win"] = work["p_away"]
    if "prob_draw" not in work.columns and "p_draw" in work.columns:
        work["prob_draw"] = work["p_draw"]
    if "prob_home_win" not in work.columns:
        work["prob_home_win"] = pd.NA
    if "prob_draw" not in work.columns:
        work["prob_draw"] = pd.NA
    if "prob_away_win" not in work.columns:
        work["prob_away_win"] = pd.NA
    if "decision_reason" not in work.columns:
        work["decision_reason"] = pd.NA
    if "draw_gap" not in work.columns:
        work["draw_gap"] = pd.NA
    if "home_score" not in work.columns:
        work["home_score"] = pd.NA
    if "away_score" not in work.columns:
        work["away_score"] = pd.NA
    work["_home_key"] = work["home_team"].map(_norm_team)
    work["_away_key"] = work["away_team"].map(_norm_team)
    work["_pair_key"] = work["_home_key"] + "||" + work["_away_key"]
    work["_pred_ok"] = work["predicted_result"].astype(str).str.upper().isin(["H", "D", "A"])
    work["_score_ok"] = pd.to_numeric(work["home_score"], errors="coerce").notna() & pd.to_numeric(
        work["away_score"], errors="coerce"
    ).notna()
    return work


def _combine_prediction_rows(df):
    pred_base = (
        df.sort_values(["_pred_ok", "_score_ok"], ascending=[False, False])
        .drop_duplicates(subset=["_pair_key"], keep="first")
        .copy()
    )
    score_base = (
        df[df["_score_ok"]]
        .sort_values(["_score_ok"], ascending=[False])
        .drop_duplicates(subset=["_pair_key"], keep="first")[["_pair_key", "home_score", "away_score"]]
    )
    out = pred_base.merge(score_base, on="_pair_key", how="left", suffixes=("", "__score"))
    for col in ["home_score", "away_score"]:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(pd.to_numeric(out[f"{col}__score"], errors="coerce"))
    out = out.drop(columns=["home_score__score", "away_score__score"], errors="ignore")
    return out


def _read_toto_order_csv(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"toto order csv not found: {path}")
    src = pd.read_csv(path, header=None, encoding="utf-8-sig")
    if src.shape[1] < 4:
        raise ValueError(f"toto order csv format error (need >=4 cols): {path}")
    out = pd.DataFrame(
        {
            "match_no": pd.to_numeric(src.iloc[:, 0], errors="coerce"),
            "home_team": src.iloc[:, 1].astype(str),
            "away_team": src.iloc[:, 3].astype(str),
        }
    ).dropna(subset=["match_no"])
    out["match_no"] = out["match_no"].astype(int)
    out["_home_key"] = out["home_team"].map(_norm_team)
    out["_away_key"] = out["away_team"].map(_norm_team)
    out["_pair_key"] = out["_home_key"] + "||" + out["_away_key"]
    out = out.sort_values("match_no").reset_index(drop=True)
    return out


def _build_merged(j1_path, j2_path, toto_order_path):
    j1 = _read_predictions(j1_path, "j1")
    j2 = _read_predictions(j2_path, "j2")
    pred_all = _combine_prediction_rows(pd.concat([j1, j2], ignore_index=True))
    toto = _read_toto_order_csv(toto_order_path)
    merged = toto.merge(
        pred_all[
            [
                "_pair_key",
                "match_id",
                "league_src",
                "home_team",
                "away_team",
                "prob_home_win",
                "prob_draw",
                "prob_away_win",
                "predicted_result",
                "decision_reason",
                "draw_gap",
                "home_score",
                "away_score",
            ]
        ],
        on="_pair_key",
        how="left",
        suffixes=("_toto", ""),
    )
    return merged


def _missing_cards_text(merged):
    pred_valid = merged["predicted_result"].astype(str).str.upper().isin(["H", "D", "A"])
    miss = merged.loc[~pred_valid, ["home_team_toto", "away_team_toto"]].copy()
    if miss.empty:
        return ""
    cards = [f"{r['home_team_toto']} vs {r['away_team_toto']}" for _, r in miss.iterrows()]
    return " ; ".join(cards)


def _summarize_single(merged, excluded_leagues):
    rows = int(len(merged))
    pred_valid = merged["predicted_result"].astype(str).str.upper().isin(["H", "D", "A"])
    matched = int(pred_valid.sum())
    missing = max(0, 13 - matched)

    print(
        "[ROUND_SUMMARY] "
        f"filter=toto_order_csv rows={rows} matched={matched}/13 missing={missing} "
        f"excluded_leagues={excluded_leagues}"
    )

    missing_cards = _missing_cards_text(merged)
    if missing_cards:
        print(f"[MISSING_CARDS] {missing_cards}")

    pred_ratio_matched = calc_hda_ratio(merged.loc[pred_valid, "predicted_result"])
    pred_ratio_toto13 = pred_ratio_matched if matched == 13 else None

    if matched == 13:
        print(_fmt_ratio_line("PRED_RATIO", pred_ratio_toto13, with_total=False))
    else:
        print(
            "[WARN] insufficient matched cards for toto13 ratio: "
            f"matched={matched}/13 (excluded_leagues={excluded_leagues})"
        )
        print("[PRED_RATIO] unavailable (insufficient matched cards)")
        print(_fmt_ratio_line("PRED_RATIO_MATCHED_ONLY", pred_ratio_matched, with_total=True))

    actual_series = pd.Series(
        [_to_result(h, a) for h, a in zip(merged["home_score"].tolist(), merged["away_score"].tolist())],
        dtype="object",
    )
    actual_valid = actual_series.notna() & pred_valid
    actual_ratio_matched = calc_hda_ratio(actual_series[actual_valid])

    if matched == 13 and int(actual_valid.sum()) == 13:
        actual_ratio_toto13 = calc_hda_ratio(actual_series)
        print(_fmt_ratio_line("ACTUAL_RATIO", actual_ratio_toto13, with_total=False))
    else:
        print("[ACTUAL_RATIO] unavailable (insufficient matched cards or scores not found)")
        if int(actual_valid.sum()) > 0:
            print(_fmt_ratio_line("ACTUAL_RATIO_MATCHED_ONLY", actual_ratio_matched, with_total=True))

    return {
        "rows": rows,
        "matched": matched,
        "missing": missing,
        "pred_ratio_toto13": pred_ratio_toto13,
        "pred_ratio_matched": pred_ratio_matched,
        "actual_ratio_toto13": calc_hda_ratio(actual_series) if (matched == 13 and int(actual_valid.sum()) == 13) else None,
        "actual_ratio_matched": actual_ratio_matched if int(actual_valid.sum()) > 0 else None,
    }


def _summary_row(kind, ratio, filter_label, rows, matched, missing, excluded_leagues):
    return {
        "kind": kind,
        "H_cnt": ratio["H"]["count"],
        "D_cnt": ratio["D"]["count"],
        "A_cnt": ratio["A"]["count"],
        "total": ratio["total"],
        "H_pct": ratio["H"]["pct"],
        "D_pct": ratio["D"]["pct"],
        "A_pct": ratio["A"]["pct"],
        "filter": filter_label,
        "rows": rows,
        "matched": matched,
        "missing": missing,
        "excluded_leagues": excluded_leagues,
    }


def _write_summary_csv(out_csv, summary, excluded_leagues):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    filter_label = "toto_order_csv"
    rows_out = []
    if summary["pred_ratio_toto13"] is not None:
        rows_out.append(
            _summary_row(
                "pred_toto13",
                summary["pred_ratio_toto13"],
                filter_label,
                summary["rows"],
                summary["matched"],
                summary["missing"],
                excluded_leagues,
            )
        )
    if summary["pred_ratio_matched"] is not None:
        rows_out.append(
            _summary_row(
                "pred_matched_only",
                summary["pred_ratio_matched"],
                filter_label,
                summary["rows"],
                summary["matched"],
                summary["missing"],
                excluded_leagues,
            )
        )
    if summary["actual_ratio_toto13"] is not None:
        rows_out.append(
            _summary_row(
                "actual_toto13",
                summary["actual_ratio_toto13"],
                filter_label,
                summary["rows"],
                summary["matched"],
                summary["missing"],
                excluded_leagues,
            )
        )
    if summary["actual_ratio_matched"] is not None:
        rows_out.append(
            _summary_row(
                "actual_matched_only",
                summary["actual_ratio_matched"],
                filter_label,
                summary["rows"],
                summary["matched"],
                summary["missing"],
                excluded_leagues,
            )
        )
    pd.DataFrame(rows_out).to_csv(out_csv, index=False, encoding="utf-8-sig")


def _run_hfa_compare(args, merged_a, merged_b):
    pred_a_valid = merged_a["predicted_result"].astype(str).str.upper().isin(["H", "D", "A"])
    pred_b_valid = merged_b["predicted_result"].astype(str).str.upper().isin(["H", "D", "A"])
    common_mask = pred_a_valid & pred_b_valid
    matched = int(common_mask.sum())
    print(f"[HFA_COMPARE] matched={matched}/13")

    # Probability-level verification for identical ON/OFF inputs.
    pa_h = pd.to_numeric(merged_a.get("prob_home_win"), errors="coerce")
    pa_d = pd.to_numeric(merged_a.get("prob_draw"), errors="coerce")
    pa_a = pd.to_numeric(merged_a.get("prob_away_win"), errors="coerce")
    pb_h = pd.to_numeric(merged_b.get("prob_home_win"), errors="coerce")
    pb_d = pd.to_numeric(merged_b.get("prob_draw"), errors="coerce")
    pb_a = pd.to_numeric(merged_b.get("prob_away_win"), errors="coerce")
    dh = (pa_h - pb_h).abs()
    dd = (pa_d - pb_d).abs()
    da = (pa_a - pb_a).abs()
    eps = 1e-12
    any_diff = ((dh > eps) | (dd > eps) | (da > eps)).fillna(False)
    num_rows_with_any_diff = int(any_diff.sum())
    max_abs_diff_prob_home = float(dh.max(skipna=True)) if dh.notna().any() else 0.0
    max_abs_diff_prob_draw = float(dd.max(skipna=True)) if dd.notna().any() else 0.0
    max_abs_diff_prob_away = float(da.max(skipna=True)) if da.notna().any() else 0.0
    print(
        "[HFA_COMPARE_CHECK] "
        f"max_abs_diff_prob_home={max_abs_diff_prob_home:.6f} "
        f"max_abs_diff_prob_draw={max_abs_diff_prob_draw:.6f} "
        f"max_abs_diff_prob_away={max_abs_diff_prob_away:.6f} "
        f"num_rows_with_any_diff={num_rows_with_any_diff}"
    )
    if num_rows_with_any_diff == 0:
        print("[WARN] HFA_ON and HFA_OFF prediction CSVs are identical (no probability differences detected)")

    ratio_a = calc_hda_ratio(merged_a.loc[common_mask, "predicted_result"])
    ratio_b = calc_hda_ratio(merged_b.loc[common_mask, "predicted_result"])
    print(_fmt_ratio_line(f"PRED_RATIO_MATCHED_ONLY:{args.label_a}", ratio_a, with_total=True))
    print(_fmt_ratio_line(f"PRED_RATIO_MATCHED_ONLY:{args.label_b}", ratio_b, with_total=True))

    diff_h = ratio_a["H"]["pct"] - ratio_b["H"]["pct"]
    diff_d = ratio_a["D"]["pct"] - ratio_b["D"]["pct"]
    diff_a = ratio_a["A"]["pct"] - ratio_b["A"]["pct"]
    print(f"[PRED_RATIO_DIFF_PP] H={diff_h:.1f}pp D={diff_d:.1f}pp A={diff_a:.1f}pp")

    # Label-change visualization on matched-only cards.
    label_df = _build_hfa_label_change_df(merged_a, merged_b)
    if label_df.empty:
        changed = 0
        unchanged = 0
    else:
        changed = int(label_df["changed"].fillna(False).astype(bool).sum())
        unchanged = int(len(label_df) - changed)
    print(f"[HFA_LABEL_CHANGE_SUMMARY] matched={len(label_df)} changed={changed} unchanged={unchanged}")
    if changed > 0:
        top_n = int(max(1, args.top))
        preview = label_df[label_df["changed"]].head(top_n)
        for _, r in preview.iterrows():
            print(
                "[HFA_LABEL_CHANGE] "
                f"match_no={r.get('match_no')} home={r.get('home_team')} away={r.get('away_team')} "
                f"OFF={r.get('pred_result_off')} -> ON={r.get('pred_result_on')} "
                f"diff(H/D/A)={float(r.get('diff_prob_home', 0.0)):+.3f}/"
                f"{float(r.get('diff_prob_draw', 0.0)):+.3f}/"
                f"{float(r.get('diff_prob_away', 0.0)):+.3f}"
            )

    out_csv = args.out_csv_hfa_compare
    if not out_csv:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        out_csv = os.path.join(
            base_dir,
            "data",
            "reports",
            "merge_qc",
            f"all_{args.season}",
            "round_summary_toto_hfa_compare.csv",
        )

    rows = [
        {
            "kind": args.label_a,
            "rows": 13,
            "matched": matched,
            "missing": max(0, 13 - matched),
            "excluded_leagues": args.excluded_leagues,
            "H_cnt": ratio_a["H"]["count"],
            "D_cnt": ratio_a["D"]["count"],
            "A_cnt": ratio_a["A"]["count"],
            "total": ratio_a["total"],
            "H_pct": ratio_a["H"]["pct"],
            "D_pct": ratio_a["D"]["pct"],
            "A_pct": ratio_a["A"]["pct"],
            "max_abs_diff_prob_home": max_abs_diff_prob_home,
            "max_abs_diff_prob_draw": max_abs_diff_prob_draw,
            "max_abs_diff_prob_away": max_abs_diff_prob_away,
            "num_rows_with_any_diff": num_rows_with_any_diff,
        },
        {
            "kind": args.label_b,
            "rows": 13,
            "matched": matched,
            "missing": max(0, 13 - matched),
            "excluded_leagues": args.excluded_leagues,
            "H_cnt": ratio_b["H"]["count"],
            "D_cnt": ratio_b["D"]["count"],
            "A_cnt": ratio_b["A"]["count"],
            "total": ratio_b["total"],
            "H_pct": ratio_b["H"]["pct"],
            "D_pct": ratio_b["D"]["pct"],
            "A_pct": ratio_b["A"]["pct"],
            "max_abs_diff_prob_home": max_abs_diff_prob_home,
            "max_abs_diff_prob_draw": max_abs_diff_prob_draw,
            "max_abs_diff_prob_away": max_abs_diff_prob_away,
            "num_rows_with_any_diff": num_rows_with_any_diff,
        },
        {
            "kind": "DIFF_PP",
            "rows": 13,
            "matched": matched,
            "missing": max(0, 13 - matched),
            "excluded_leagues": args.excluded_leagues,
            "H_cnt": pd.NA,
            "D_cnt": pd.NA,
            "A_cnt": pd.NA,
            "total": matched,
            "H_pct": diff_h,
            "D_pct": diff_d,
            "A_pct": diff_a,
            "max_abs_diff_prob_home": max_abs_diff_prob_home,
            "max_abs_diff_prob_draw": max_abs_diff_prob_draw,
            "max_abs_diff_prob_away": max_abs_diff_prob_away,
            "num_rows_with_any_diff": num_rows_with_any_diff,
        },
    ]
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[HFA_COMPARE_CSV] saved={out_csv}")

    label_out_csv = args.out_csv_label_changes
    if not label_out_csv:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        label_out_csv = os.path.join(
            base_dir,
            "data",
            "reports",
            "merge_qc",
            f"all_{args.season}",
            "toto_hfa_label_changes.csv",
        )
    os.makedirs(os.path.dirname(label_out_csv), exist_ok=True)
    label_export = label_df.copy()
    if args.only_changed:
        label_export = label_export[label_export["changed"]].copy()
    label_export.to_csv(label_out_csv, index=False, encoding="utf-8-sig")
    print(f"[HFA_LABEL_CHANGE_CSV] saved={label_out_csv} rows={len(label_export)}")


def _argmax_hda(prob_home, prob_draw, prob_away):
    vals = {"H": prob_home, "D": prob_draw, "A": prob_away}
    clean = {k: pd.to_numeric(v, errors="coerce") for k, v in vals.items()}
    if all(pd.isna(v) for v in clean.values()):
        return pd.NA
    return max(clean, key=lambda k: (-1e18 if pd.isna(clean[k]) else clean[k]))


def _calc_draw_gap(prob_home, prob_draw, prob_away):
    ph = pd.to_numeric(prob_home, errors="coerce")
    pdw = pd.to_numeric(prob_draw, errors="coerce")
    pa = pd.to_numeric(prob_away, errors="coerce")
    if pd.isna(ph) or pd.isna(pdw) or pd.isna(pa):
        return pd.NA
    return max(float(ph), float(pa)) - float(pdw)


def _build_hfa_label_change_df(merged_a, merged_b):
    left = merged_a.copy()
    right = merged_b.copy()
    left["pred_result_on"] = left["predicted_result"].astype(str).str.upper()
    right["pred_result_off"] = right["predicted_result"].astype(str).str.upper()
    left_valid = left["pred_result_on"].isin(["H", "D", "A"])
    right_valid = right["pred_result_off"].isin(["H", "D", "A"])
    left = left[left_valid].copy()
    right = right[right_valid].copy()
    if left.empty or right.empty:
        return pd.DataFrame(
            columns=[
                "match_no",
                "match_id",
                "home_team",
                "away_team",
                "pred_result_off",
                "pred_result_on",
                "changed",
                "prob_home_off",
                "prob_draw_off",
                "prob_away_off",
                "prob_home_on",
                "prob_draw_on",
                "prob_away_on",
                "diff_prob_home",
                "diff_prob_draw",
                "diff_prob_away",
                "argmax_off",
                "argmax_on",
                "decide_reason_off",
                "decide_reason_on",
                "draw_gap_off",
                "draw_gap_on",
            ]
        )

    left["match_id"] = left.get("match_id", pd.Series(index=left.index, dtype="object")).astype(str).str.strip()
    right["match_id"] = right.get("match_id", pd.Series(index=right.index, dtype="object")).astype(str).str.strip()
    has_match_id_left = left["match_id"].ne("") & left["match_id"].ne("nan")
    has_match_id_right = right["match_id"].ne("") & right["match_id"].ne("nan")

    join_cols = [
        "match_no",
        "_pair_key",
        "match_id",
        "home_team_toto",
        "away_team_toto",
        "pred_result_on",
        "pred_result_off",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "decision_reason",
        "draw_gap",
    ]
    left_sel = left[[c for c in join_cols if c in left.columns]].copy()
    right_sel = right[[c for c in join_cols if c in right.columns]].copy()

    both_with_id = left_sel[has_match_id_left].merge(
        right_sel[has_match_id_right],
        on="match_id",
        how="inner",
        suffixes=("_on", "_off"),
    )
    used_pair = set(both_with_id.get("_pair_key_on", pd.Series(dtype="object")).astype(str).tolist())
    rem_left = left_sel[~left_sel.get("_pair_key", pd.Series(dtype="object")).astype(str).isin(used_pair)]
    rem_right = right_sel[~right_sel.get("_pair_key", pd.Series(dtype="object")).astype(str).isin(used_pair)]
    by_pair = rem_left.merge(rem_right, on="_pair_key", how="inner", suffixes=("_on", "_off"))
    aligned = pd.concat([both_with_id, by_pair], ignore_index=True, sort=False)
    if aligned.empty:
        return pd.DataFrame()

    out = pd.DataFrame()
    out["match_no"] = pd.to_numeric(
        aligned.get("match_no_on", aligned.get("match_no_off", pd.NA)),
        errors="coerce",
    )
    out["match_id"] = aligned.get("match_id", pd.NA)
    out["home_team"] = aligned.get("home_team_toto_on", aligned.get("home_team_toto_off", pd.NA))
    out["away_team"] = aligned.get("away_team_toto_on", aligned.get("away_team_toto_off", pd.NA))
    out["pred_result_off"] = aligned.get("pred_result_off", pd.NA)
    out["pred_result_on"] = aligned.get("pred_result_on", pd.NA)
    out["changed"] = out["pred_result_on"] != out["pred_result_off"]

    out["prob_home_off"] = pd.to_numeric(aligned.get("prob_home_win_off"), errors="coerce")
    out["prob_draw_off"] = pd.to_numeric(aligned.get("prob_draw_off"), errors="coerce")
    out["prob_away_off"] = pd.to_numeric(aligned.get("prob_away_win_off"), errors="coerce")
    out["prob_home_on"] = pd.to_numeric(aligned.get("prob_home_win_on"), errors="coerce")
    out["prob_draw_on"] = pd.to_numeric(aligned.get("prob_draw_on"), errors="coerce")
    out["prob_away_on"] = pd.to_numeric(aligned.get("prob_away_win_on"), errors="coerce")

    out["diff_prob_home"] = out["prob_home_on"] - out["prob_home_off"]
    out["diff_prob_draw"] = out["prob_draw_on"] - out["prob_draw_off"]
    out["diff_prob_away"] = out["prob_away_on"] - out["prob_away_off"]
    out["argmax_off"] = out.apply(
        lambda r: _argmax_hda(r["prob_home_off"], r["prob_draw_off"], r["prob_away_off"]), axis=1
    )
    out["argmax_on"] = out.apply(
        lambda r: _argmax_hda(r["prob_home_on"], r["prob_draw_on"], r["prob_away_on"]), axis=1
    )
    out["decide_reason_off"] = aligned.get("decision_reason_off", pd.NA)
    out["decide_reason_on"] = aligned.get("decision_reason_on", pd.NA)
    out["draw_gap_off"] = pd.to_numeric(aligned.get("draw_gap_off"), errors="coerce")
    out["draw_gap_on"] = pd.to_numeric(aligned.get("draw_gap_on"), errors="coerce")
    out["draw_gap_off"] = out.apply(
        lambda r: r["draw_gap_off"]
        if pd.notna(r["draw_gap_off"])
        else _calc_draw_gap(r["prob_home_off"], r["prob_draw_off"], r["prob_away_off"]),
        axis=1,
    )
    out["draw_gap_on"] = out.apply(
        lambda r: r["draw_gap_on"]
        if pd.notna(r["draw_gap_on"])
        else _calc_draw_gap(r["prob_home_on"], r["prob_draw_on"], r["prob_away_on"]),
        axis=1,
    )

    out["__abs_diff_sum"] = (
        out["diff_prob_home"].abs().fillna(0.0)
        + out["diff_prob_draw"].abs().fillna(0.0)
        + out["diff_prob_away"].abs().fillna(0.0)
    )
    out = out.sort_values(
        by=["changed", "__abs_diff_sum", "match_no"],
        ascending=[False, False, True],
        na_position="last",
    ).drop(columns=["__abs_diff_sum"])
    return out


def parse_args():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    data_dir = os.path.join(base_dir, "data")
    manual_dir = os.path.join(data_dir, "manual")
    season = int(os.environ.get("SEASON_YEAR", "2026"))
    p = argparse.ArgumentParser(description="toto対象13試合の予測/実結果比率サマリ")
    p.add_argument("--season", type=int, default=season)
    p.add_argument("--j1", default=None, help="default: ./j1_{season}_predictions.csv")
    p.add_argument("--j2", default=None, help="default: ./j2_{season}_predictions.csv")
    p.add_argument("--toto-order", default=os.path.join(manual_dir, "toto並び順.csv"))
    p.add_argument("--out-csv", default=None, help="default: data/reports/merge_qc/all_{season}/round_summary_toto.csv")
    p.add_argument("--excluded-leagues", default=os.environ.get("EXCLUDED_LEAGUES", "j3"))
    p.add_argument("--strict-13", action="store_true", help="13件でなければ終了コード1")

    p.add_argument("--compare-hfa", action="store_true")
    p.add_argument("--pred-a-j1", default=None)
    p.add_argument("--pred-a-j2", default=None)
    p.add_argument("--pred-b-j1", default=None)
    p.add_argument("--pred-b-j2", default=None)
    p.add_argument("--label-a", default="A")
    p.add_argument("--label-b", default="B")
    p.add_argument("--out-csv-hfa-compare", default=None)
    p.add_argument("--out-csv-label-changes", default=None)
    p.add_argument("--only-changed", action="store_true")
    p.add_argument("--top", type=int, default=10)
    return p.parse_args()


def main():
    args = parse_args()
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    j1_path = args.j1 or os.path.join(base_dir, f"j1_{args.season}_predictions.csv")
    j2_path = args.j2 or os.path.join(base_dir, f"j2_{args.season}_predictions.csv")
    out_csv = args.out_csv or os.path.join(
        base_dir, "data", "reports", "merge_qc", f"all_{args.season}", "round_summary_toto.csv"
    )

    merged = _build_merged(j1_path, j2_path, args.toto_order)
    rows = int(len(merged))
    if rows != 13:
        print(f"[WARN] toto対象行数が13ではありません: rows={rows}")
        if args.strict_13:
            raise RuntimeError(f"strict-13: rows={rows}")

    summary = _summarize_single(merged, args.excluded_leagues)
    _write_summary_csv(out_csv, summary, args.excluded_leagues)
    print(f"[ROUND_SUMMARY_CSV] saved={out_csv}")

    if args.compare_hfa:
        a_j1 = args.pred_a_j1 or j1_path
        a_j2 = args.pred_a_j2 or j2_path
        b_j1 = args.pred_b_j1
        b_j2 = args.pred_b_j2
        if not (b_j1 and b_j2):
            raise ValueError("--compare-hfa requires --pred-b-j1 and --pred-b-j2")
        merged_a = _build_merged(a_j1, a_j2, args.toto_order)
        merged_b = _build_merged(b_j1, b_j2, args.toto_order)
        _run_hfa_compare(args, merged_a, merged_b)

if __name__ == "__main__":
    main()
