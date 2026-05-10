import os
import re
import unicodedata
import glob
import pandas as pd
from report_view_utils import write_html_table


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORT_DIR = os.path.abspath(os.path.dirname(__file__))
HTML_DIR = os.path.join(REPORT_DIR, "html")
CSV_DIR = os.path.join(REPORT_DIR, "csv")
TOTO_TARGET_HISTORY_CSV = os.path.join(ROOT_DIR, "data", "manual", "toto節リスト.csv")

PRED_DESC_MAP = {
    "節": "節",
    "section": "節",
    "date": "日付",
    "match_id": "試合ID",
    "datetime": "試合日時",
    "stadium": "スタジアム",
    "home_team": "ホーム",
    "away_team": "アウェイ",
    "predicted_result": "最終表示予想: 画面・buyplanで使う採用票（後段override反映後の最終値）",
    "predicted_result_main": "予想01: 本線票。勝敗確率のargmaxで決める基準予想",
    "predicted_result_type_a": "予想02旧互換列: 旧type_aの保持用",
    "predicted_result_type_b": "予想02: LAB展開票。hold/stall/flip を反映した別勝敗予想",
    "predicted_result_type_c": "予想03: 予想02準拠のDグラデーション票。予想02で落とした弱いD揺れを補う",
    "predicted_result_main_symbol": "予想01 toto記号（生確率argmax。predicted_result_main は後段補正後の最終本線）",
    "predicted_result_type_a_symbol": "予想02旧互換 toto記号",
    "predicted_result_type_b_symbol": "予想02 toto記号",
    "predicted_result_type_c_symbol": "予想03 toto記号",
    "match_type": "試合タイプ",
    "match_type_primary": "主試合タイプ",
    "match_type_flags": "試合タイプフラグ",
    "match_type_reason": "試合タイプ判定理由",
    "draw_risk_flag": "引分リスク",
    "draw_gap": "最大勝敗確率-引分確率",
    "type_adjust_note": "試合タイプ診断メモ",
    "type_adjust_note_a": "予想02旧互換補正メモ",
    "type_adjust_note_b": "予想02補正メモ",
    "type_adjust_note_c": "予想03補正メモ",
    "adjusted_prob_home_a": "予想02補正後H確率",
    "adjusted_prob_draw_a": "予想02補正後D確率",
    "adjusted_prob_away_a": "予想02補正後A確率",
    "adjusted_prob_home_b": "予想02補正後H確率",
    "adjusted_prob_draw_b": "予想02補正後D確率",
    "adjusted_prob_away_b": "予想02補正後A確率",
    "home_score": "ホーム得点（予測時は空）",
    "away_score": "アウェイ得点（予測時は空）",
    "prob_home_win": "ホーム勝ち確率",
    "prob_draw": "引き分け確率",
    "prob_away_win": "アウェイ勝ち確率",
    "flab_trial_flag": "Football LAB試験フラグ",
    "flab_trial_score": "Football LAB試験スコア",
    "flab_trial_reason": "Football LAB試験理由",
    "flab_trial_avg_abs_edge": "Football LAB平均絶対差",
    "flab_trial_home_edge_allowed_shot_conversion": "被シュート成功率差(ホーム優位換算)",
    "flab_trial_home_edge_shot_conversion": "シュート成功率差(ホーム優位換算)",
    "home_elo": "ホームElo",
    "away_elo": "アウェイElo",
}

BACKTEST_DESC_MAP = {
    "節": "節",
    "match_id": "試合ID",
    "datetime": "試合日時",
    "stadium": "スタジアム",
    "home_team": "ホーム",
    "away_team": "アウェイ",
    "home_score": "ホーム得点",
    "away_score": "アウェイ得点",
    "prob_home_win": "ホーム勝ち確率",
    "prob_draw": "引き分け確率",
    "prob_away_win": "アウェイ勝ち確率",
    "predicted_highest_prob_result": "予測結果",
    "actual_result": "実際の結果",
    "is_correct": "的中",
    "toto_target_status": "toto対象との対応",
    "match_no": "toto並び順",
}


def reorder_backtest_columns(df):
    out = _consolidate_frame(df)
    drop_cols = [
        "home_elo",
        "away_elo",
        "management_manager_name_home",
        "management_manager_name_away",
    ]
    out = _without_columns(out, drop_cols, copy=False)
    front = ["league", "節", "match_id", "datetime", "stadium", "home_team", "away_team"]
    focus = ["actual_result", "is_correct", "predicted_result"]
    ordered = [c for c in front if c in out.columns]
    ordered += [c for c in focus if c in out.columns and c not in ordered]
    ordered += [c for c in out.columns if c not in ordered]
    return out.loc[:, ordered].copy()


def reorder_prediction_columns(df):
    out = _consolidate_frame(df)
    out = _without_columns(
        out,
        [
            "predicted_result_type_a",
            "predicted_result_type_a_symbol",
            "type_adjust_note_a",
            "type_a_symbol",
            "type_a_reason",
            "type_a_draw_signal_strong",
            "type_a_draw_signal_weak",
        ],
        copy=False,
    )
    # Reporting view should show the effective candidate values, not raw sparse branch outputs.
    if "predicted_result_main" in out.columns and "predicted_result_type_b" in out.columns:
        b_raw = out["predicted_result_type_b"].fillna("").astype(str).str.upper()
        main_raw = out["predicted_result_main"].fillna("").astype(str).str.upper()
        b_final = b_raw.where(b_raw.ne(""), main_raw)
        out["predicted_result_type_b"] = b_final
        if "predicted_result_type_b_symbol" in out.columns:
            out["predicted_result_type_b_symbol"] = b_final.map({"H": "1", "D": "0", "A": "2"}).fillna("")
    if "predicted_result_main" in out.columns and "predicted_result_type_c" in out.columns:
        c_raw = out["predicted_result_type_c"].fillna("").astype(str).str.upper()
        b_base = out.get("predicted_result_type_b", pd.Series(index=out.index, dtype="object")).fillna("").astype(str).str.upper()
        main_raw = out["predicted_result_main"].fillna("").astype(str).str.upper()
        c_final = c_raw.where(c_raw.ne(""), b_base.where(b_base.ne(""), main_raw))
        out["predicted_result_type_c"] = c_final
        if "predicted_result_type_c_symbol" in out.columns:
            out["predicted_result_type_c_symbol"] = c_final.map({"H": "1", "D": "0", "A": "2"}).fillna("")
    front = ["league", "節", "match_id", "datetime", "stadium", "home_team", "away_team"]
    focus = [
        "predicted_result",
        "predicted_result_main",
        "predicted_result_type_b",
        "predicted_result_type_c",
        "predicted_result_main_symbol",
        "predicted_result_type_b_symbol",
        "predicted_result_type_c_symbol",
        "match_type",
        "match_type_primary",
        "match_type_flags",
        "draw_risk_flag",
        "draw_gap",
        "type_adjust_note_b",
        "type_adjust_note_c",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "adjusted_prob_home_a",
        "adjusted_prob_draw_a",
        "adjusted_prob_away_a",
        "adjusted_prob_home_b",
        "adjusted_prob_draw_b",
        "adjusted_prob_away_b",
        "flab_trial_flag",
        "flab_trial_score",
        "flab_trial_reason",
    ]
    ordered = [c for c in front if c in out.columns]
    ordered += [c for c in focus if c in out.columns and c not in ordered]
    ordered += [c for c in out.columns if c not in ordered]
    return out.loc[:, ordered].copy()


def ensure_league_column(df, league_value):
    out = df.copy()
    lv = str(league_value).upper()
    if "league" in out.columns:
        out["league"] = lv
        cols = ["league"] + [c for c in out.columns if c != "league"]
        return out[cols]
    out.insert(0, "league", lv)
    return out


def _norm_team_key(v):
    s = str(v) if not pd.isna(v) else ""
    s = unicodedata.normalize("NFKC", s)
    s = s.strip().replace("　", " ")
    s = re.sub(r"\s+", "", s)
    s = s.replace("・", "").replace(".", "").replace("･", "")
    s = s.upper()
    team_alias = {
        "FC東京": "FC東京",
        "FC今治": "今治",
        "SC相模原": "相模原",
        "RB大宮": "大宮",
        "RB大宮アルディージャ": "大宮",
        "横浜FC": "横浜FC",
        "横浜FM": "横浜FM",
        "川崎F": "川崎F",
        "東京V": "東京V",
        "C大阪": "C大阪",
        "G大阪": "G大阪",
    }
    return team_alias.get(s, s)


def _consolidate_frame(df, min_blocks=64):
    if df is None or df.empty:
        return df
    mgr = getattr(df, "_mgr", None)
    nblocks = getattr(mgr, "nblocks", None)
    if nblocks is not None and nblocks >= min_blocks:
        return df.copy()
    return df


def _without_columns(df, columns, copy=False):
    if df is None or df.empty:
        return df.copy() if copy and df is not None else df
    remove = set(columns)
    kept = [c for c in df.columns if c not in remove]
    out = df.loc[:, kept]
    return out.copy() if copy else out


def _iter_blocks_by_column(df, column):
    if df is None or df.empty or column not in df.columns:
        return
    values = df[column]
    if not isinstance(values, pd.Series):
        values = pd.Series(values, index=df.index)
    normalized = values.astype("string").str.strip()
    seen = set()
    for raw in pd.unique(normalized.dropna()):
        key = str(raw)
        if key in seen:
            continue
        seen.add(key)
        yield key, df.loc[normalized.eq(raw)]


def _with_backtest_key(df):
    out = df.copy()
    out["__dt"] = pd.to_datetime(out.get("datetime"), errors="coerce")
    out["__day"] = out["__dt"].dt.strftime("%Y-%m-%d")
    out["__home_k"] = out.get("home_team", pd.Series(index=out.index, dtype="object")).map(_norm_team_key)
    out["__away_k"] = out.get("away_team", pd.Series(index=out.index, dtype="object")).map(_norm_team_key)
    out["__pair_key"] = out["__home_k"] + "|" + out["__away_k"]
    out["__pair_day"] = out["__home_k"] + "|" + out["__away_k"] + "|" + out["__day"].fillna("")
    if "match_id" not in out.columns:
        out["match_id"] = pd.NA
    out["match_id"] = out["match_id"].astype(str).str.strip()
    out["__match_id_ok"] = out["match_id"].ne("") & out["match_id"].ne("nan")
    hs = out.get("home_score", pd.Series(index=out.index, dtype="object"))
    aw = out.get("away_score", pd.Series(index=out.index, dtype="object"))
    out["__score_ok"] = pd.to_numeric(hs, errors="coerce").notna() & pd.to_numeric(aw, errors="coerce").notna()
    out["__actual_ok"] = out.get("actual_result", pd.Series(index=out.index, dtype="object")).notna()
    return out


def _with_prediction_key(df):
    out = df.copy()
    out["__dt"] = pd.to_datetime(out.get("datetime"), errors="coerce")
    out["__home_k"] = out.get("home_team", pd.Series(index=out.index, dtype="object")).map(_norm_team_key)
    out["__away_k"] = out.get("away_team", pd.Series(index=out.index, dtype="object")).map(_norm_team_key)
    out["__pair_key"] = out["__home_k"] + "|" + out["__away_k"]
    if "match_id" not in out.columns:
        out["match_id"] = pd.NA
    out["match_id"] = out["match_id"].astype(str).str.strip()
    out["__match_id_ok"] = out["match_id"].ne("") & out["match_id"].ne("nan")
    core_cols = ["predicted_result", "prob_home_win", "prob_draw", "prob_away_win", "home_score", "away_score", "stadium"]
    out["__pred_quality"] = out[[c for c in core_cols if c in out.columns]].notna().sum(axis=1)
    return out


def _dedupe_prediction_rows(df):
    if df is None or df.empty:
        return df
    x = _consolidate_frame(_with_prediction_key(df))
    x = x.sort_values(
        ["__pred_quality", "__dt"],
        ascending=[False, False],
        na_position="last",
    )
    # 予測ビューでは「同一カード重複」を最優先で排除する。
    # 同節内で match_id が異なる重複（延期/補完データ混入）でも1行に統一する。
    out = x.drop_duplicates(subset=["__pair_key"], keep="first")
    drop_cols = [c for c in out.columns if c.startswith("__")]
    return out.drop(columns=drop_cols, errors="ignore")


def _merge_backtest_candidates(candidates):
    parts = []
    for src_rank, src_name, df in candidates:
        x = _with_backtest_key(df)
        x["__src_rank"] = int(src_rank)
        x["__src_name"] = src_name
        parts.append(x)
    merged = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if merged.empty:
        return merged
    merged = _consolidate_frame(merged)
    pred = merged.get("predicted_result", pd.Series(index=merged.index, dtype="object")).astype("string")
    pred = pred.mask(pred.str.strip().isin(["", "nan", "none", "nat", "<na>"]), pd.NA)
    for alt_col in ["predicted_highest_prob_result", "final_result", "argmax_result"]:
        if alt_col in merged.columns:
            alt = merged[alt_col].astype("string")
            alt = alt.mask(alt.str.strip().isin(["", "nan", "none", "nat", "<na>"]), pd.NA)
            pred = pred.fillna(alt)
    merged["predicted_result"] = pred
    merged["__pred_ok"] = pred.notna()
    base_cols = [c for c in merged.columns if not c.startswith("__")]
    merged["__filled_n"] = merged[base_cols].notna().sum(axis=1)
    merged = merged.sort_values(
        ["__pred_ok", "__filled_n", "__actual_ok", "__score_ok", "__src_rank"],
        ascending=[False, False, False, False, True],
        na_position="last",
    )
    # backtestビューは「同一カード（home/away）を1行」に寄せる。
    # match_idが時刻違い（xx00/xx03）で重複しても、情報量が高い行を優先採用する。
    out = merged.drop_duplicates(subset=["__pair_key"], keep="first")
    drop_cols = [c for c in out.columns if c.startswith("__")]
    return out.drop(columns=drop_cols, errors="ignore")


def _keep_finished_backtest_rows(df):
    if df is None or df.empty:
        return df
    out = df.copy()
    hs = pd.to_numeric(out.get("home_score"), errors="coerce")
    aw = pd.to_numeric(out.get("away_score"), errors="coerce")
    actual = out.get("actual_result", pd.Series(index=out.index, dtype="object"))
    finished = actual.notna() | (hs.notna() & aw.notna())
    return out[finished].copy()


def _enrich_backtest_with_results(df, league, year):
    if df.empty:
        return df
    result_path = os.path.join(ROOT_DIR, "data", f"{league}_{year}_latest_results.csv")
    if not os.path.exists(result_path):
        return df
    try:
        res = pd.read_csv(result_path)
    except Exception:
        return df
    if not {"home_team", "away_team"}.issubset(res.columns):
        return df
    left = _with_backtest_key(df)
    right = _with_backtest_key(res)
    right_cols = ["match_id", "__pair_day", "home_score", "away_score"]
    r = right[right_cols].drop_duplicates(subset=["match_id", "__pair_day"], keep="last")
    # match_id優先で補完
    by_id = left.merge(
        r[["match_id", "home_score", "away_score"]].dropna(subset=["match_id"]).drop_duplicates("match_id", keep="last"),
        on="match_id",
        how="left",
        suffixes=("", "__res_id"),
    )
    for c in ["home_score", "away_score"]:
        by_id[c] = pd.to_numeric(by_id[c], errors="coerce").fillna(pd.to_numeric(by_id[f"{c}__res_id"], errors="coerce"))
    by_id = by_id.drop(columns=["home_score__res_id", "away_score__res_id"], errors="ignore")
    # pair-dayで残り補完
    by_pd = by_id.merge(
        r[["__pair_day", "home_score", "away_score"]].drop_duplicates("__pair_day", keep="last"),
        on="__pair_day",
        how="left",
        suffixes=("", "__res_pd"),
    )
    for c in ["home_score", "away_score"]:
        by_pd[c] = pd.to_numeric(by_pd[c], errors="coerce").fillna(pd.to_numeric(by_pd[f"{c}__res_pd"], errors="coerce"))
    by_pd = by_pd.drop(columns=["home_score__res_pd", "away_score__res_pd"], errors="ignore")
    # actual/is_correct 再計算
    hs = pd.to_numeric(by_pd.get("home_score"), errors="coerce")
    aw = pd.to_numeric(by_pd.get("away_score"), errors="coerce")
    actual = pd.Series(pd.NA, index=by_pd.index, dtype="object")
    actual = actual.mask(hs > aw, "H").mask(hs < aw, "A").mask((hs == aw) & hs.notna() & aw.notna(), "D")
    by_pd["actual_result"] = actual
    if "predicted_result" in by_pd.columns:
        pred = by_pd["predicted_result"].astype(str).str.upper()
        by_pd["is_correct"] = (pred == by_pd["actual_result"]).where(by_pd["actual_result"].notna(), pd.NA)
    drop_cols = [c for c in by_pd.columns if c.startswith("__")]
    return by_pd.drop(columns=drop_cols, errors="ignore")


def _filter_backtest_by_match_id_league(df, league):
    if df is None or df.empty or "match_id" not in df.columns:
        return df
    out = df.copy()
    mid = out["match_id"].astype(str).str.strip()
    known = mid.notna() & mid.ne("") & mid.ne("nan")
    keep = ~known | mid.str.lower().str.startswith(f"{str(league).lower()}_")
    return out[keep].copy()


def _dedupe_merged_backtest_round(df):
    if df is None or df.empty:
        return df
    x = _consolidate_frame(_with_backtest_key(df))

    # 表示列 predicted_result を優先補完（空/NaNは代替列で埋める）
    pred = x.get("predicted_result", pd.Series(index=x.index, dtype="object")).astype("string")
    pred = pred.mask(pred.str.strip().isin(["", "nan", "none", "nat", "<na>"]), pd.NA)
    for alt_col in ["predicted_highest_prob_result", "final_result", "argmax_result"]:
        if alt_col in x.columns:
            alt = x[alt_col].astype("string")
            alt = alt.mask(alt.str.strip().isin(["", "nan", "none", "nat", "<na>"]), pd.NA)
            pred = pred.fillna(alt)
    x["predicted_result"] = pred

    pred_s = pred.str.strip().str.lower().fillna("")
    x["__pred_ok"] = ~pred_s.isin(["", "nan", "none", "nat", "<na>"])
    base_cols = [c for c in x.columns if not c.startswith("__")]
    x["__filled_n"] = x[base_cols].notna().sum(axis=1)
    x["__league_pref"] = x.get("league", pd.Series(index=x.index, dtype="object")).astype(str).map(
        lambda v: {"J1": 0, "J2": 1, "J3": 2}.get(v.upper(), 9)
    )
    x = x.sort_values(
        ["__pred_ok", "__actual_ok", "__score_ok", "__filled_n", "__league_pref"],
        ascending=[False, False, False, False, True],
        na_position="last",
    )
    # 2026特別大会のように、同一カードが J1/J2 両方へ複製されるケースがある。
    # 結合ビューでは同一日・同一カードを1行に統一し、情報量の高い行を優先採用する。
    out = x.drop_duplicates(subset=["__pair_day"], keep="first")
    drop_cols = [c for c in out.columns if c.startswith("__")]
    return out.drop(columns=drop_cols, errors="ignore")


def backtest_quality_score(df):
    if df is None or df.empty:
        return (-1, -1, -1)
    score = pd.to_numeric(df.get("home_score"), errors="coerce").notna() & pd.to_numeric(
        df.get("away_score"), errors="coerce"
    ).notna()
    score_cnt = int(score.sum())
    actual_cnt = int(df.get("actual_result", pd.Series(index=df.index, dtype="object")).notna().sum())
    row_cnt = int(len(df))
    return (actual_cnt, score_cnt, row_cnt)


def sanitize_round(value):
    text = str(value).strip()
    return re.sub(r"[\\s/\\\\]+", "_", text)


def extract_round_number(label):
    m = re.search(r"第(\d+)節", str(label).strip())
    return int(m.group(1)) if m else None


def round_label_from_section(section_value):
    text = str(section_value).strip()
    m = re.search(r"第(\d+)節", text)
    if m:
        return f"第{m.group(1)}節"
    m = re.search(r"第(\d+)戦", text)
    if m:
        return f"第{m.group(1)}戦"
    if text.isdigit():
        return f"第{text}節"
    return text


def parse_league_year_from_predictions_filename(fname):
    m = re.match(r"^(j[123])_(\d{4})_predictions\.csv$", fname)
    if m:
        return m.group(1), m.group(2)
    return "na", "na"


def parse_league_year_from_backtest_filename(fname):
    base = os.path.basename(fname)
    m = re.match(r"^backtest_(j[123])_(\d{4})(?:_rounds)?\.csv$", base)
    if m:
        return m.group(1), m.group(2)
    return "na", "na"


def parse_league_year_from_match_id(match_id):
    text = str(match_id).strip()
    m = re.match(r"^(j[123])_(\d{4})_", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).lower(), m.group(2)
    return "na", "na"


def load_toto_round_map():
    out = {}
    if not os.path.exists(TOTO_TARGET_HISTORY_CSV):
        return out
    try:
        df = pd.read_csv(TOTO_TARGET_HISTORY_CSV)
    except Exception:
        return out
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()
    j1_round_col = "J1_round" if "J1_round" in df.columns else ("j1_round" if "j1_round" in df.columns else None)
    needed = {"season", "toto_round"}
    if j1_round_col is None or not needed.issubset(df.columns):
        return out
    work = df.copy()
    work["season"] = pd.to_numeric(work["season"], errors="coerce")
    work[j1_round_col] = pd.to_numeric(work[j1_round_col], errors="coerce")
    work["toto_round"] = pd.to_numeric(work["toto_round"], errors="coerce")
    work = work.dropna(subset=["season", j1_round_col, "toto_round"])
    work = work.drop_duplicates(subset=["season", j1_round_col], keep="last")
    for _, row in work.iterrows():
        out[(str(int(row["season"])), int(row[j1_round_col]))] = str(int(row["toto_round"]))
    return out


def load_toto_target_history():
    if not os.path.exists(TOTO_TARGET_HISTORY_CSV):
        return pd.DataFrame()
    try:
        df = pd.read_csv(TOTO_TARGET_HISTORY_CSV)
    except Exception:
        return pd.DataFrame()
    # 手編集CSVの末尾空列がそのまま backtest HTML に混入しないように除外する。
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed:")].copy()
    needed = {"season", "toto_round", "match_no", "home_team", "away_team"}
    if not needed.issubset(df.columns):
        return pd.DataFrame()
    out = df.copy()
    for col in ["season", "toto_round", "match_no"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=["season", "toto_round", "match_no", "home_team", "away_team"]).copy()
    out["season"] = out["season"].astype(int).astype(str)
    out["toto_round"] = out["toto_round"].astype(int).astype(str)
    out["match_no"] = out["match_no"].astype(int)
    out["home_team"] = out["home_team"].astype(str).str.strip()
    out["away_team"] = out["away_team"].astype(str).str.strip()
    out["_pair_key"] = out["home_team"].map(_norm_team_key) + "|" + out["away_team"].map(_norm_team_key)
    return out


def build_toto_target_backtest_view(merged_df, year, toto_round, target_history):
    if merged_df is None or merged_df.empty or target_history is None or target_history.empty:
        return pd.DataFrame()
    targets = target_history[
        (target_history["season"] == str(year)) & (target_history["toto_round"] == str(toto_round))
    ].copy()
    if targets.empty:
        return pd.DataFrame()
    targets = targets.sort_values("match_no").reset_index(drop=True)

    source = merged_df.copy()
    source["_pair_key"] = source["home_team"].map(_norm_team_key) + "|" + source["away_team"].map(_norm_team_key)
    source = source.drop_duplicates(subset=["_pair_key"], keep="first")

    merged = targets.merge(
        source,
        on="_pair_key",
        how="left",
        suffixes=("_target", ""),
    )
    if "home_team_target" in merged.columns:
        merged["home_team"] = merged["home_team_target"]
    elif "home_team_x" in merged.columns and "home_team_y" in merged.columns:
        merged["home_team"] = merged["home_team_x"]
    if "away_team_target" in merged.columns:
        merged["away_team"] = merged["away_team_target"]
    elif "away_team_x" in merged.columns and "away_team_y" in merged.columns:
        merged["away_team"] = merged["away_team_x"]
    if "season_target" in merged.columns:
        merged["season"] = merged["season_target"]
    elif "season_x" in merged.columns and "season_y" in merged.columns:
        merged["season"] = merged["season_x"]
    merged["toto_round"] = str(toto_round)
    merged["toto_target_status"] = merged["match_id"].notna().map(lambda v: "MATCHED" if v else "TARGET_ONLY")
    if "league" in merged.columns:
        merged["league"] = merged["league"].fillna("TARGET_ONLY")
    else:
        merged["league"] = "TARGET_ONLY"
    if "datetime" not in merged.columns:
        merged["datetime"] = pd.NA
    if "stadium" not in merged.columns:
        merged["stadium"] = pd.NA
    keep_front = ["match_no", "league", "toto_round", "toto_target_status", "home_team", "away_team"]
    remain = [
        c
        for c in merged.columns
        if c not in keep_front
        and c != "_pair_key"
        and c not in {
            "home_team_target",
            "away_team_target",
            "season_target",
            "home_team_x",
            "home_team_y",
            "away_team_x",
            "away_team_y",
            "season_x",
            "season_y",
        }
    ]
    out = merged[keep_front + remain].copy()
    return out.sort_values("match_no").reset_index(drop=True)


def _latest_snapshot_path(base_dir, prefix):
    pat = os.path.join(base_dir, f"{prefix}_asof_*.csv")
    cands = sorted(glob.glob(pat))
    return cands[-1] if cands else None


def _team_key_series(s):
    return s.astype("string").map(_norm_team_key)


def _fill_if_missing(df, col, values):
    if col not in df.columns:
        return
    m = df[col].isna()
    if m.any():
        df.loc[m, col] = values.loc[m]


def _merge_prediction_supplement(out_df, supp_df):
    if out_df is None or out_df.empty or supp_df is None or supp_df.empty:
        return out_df
    if "match_id" not in out_df.columns or "match_id" not in supp_df.columns:
        return out_df

    base = _consolidate_frame(out_df)
    supp = _consolidate_frame(supp_df)
    base["match_id"] = base["match_id"].astype(str).str.strip()
    supp["match_id"] = supp["match_id"].astype(str).str.strip()
    supp = supp.drop_duplicates(subset=["match_id"], keep="last")

    merge_cols = [c for c in supp.columns if c != "match_id"]
    merged = base.merge(supp[["match_id"] + merge_cols], on="match_id", how="left", suffixes=("", "__supp"))
    merged = _consolidate_frame(merged)
    for col in merge_cols:
        supp_col = f"{col}__supp"
        if col in base.columns and supp_col in merged.columns:
            merged[col] = merged[col].where(merged[col].notna(), merged[supp_col])
        elif col in merged.columns:
            merged[col] = merged[col]
        else:
            merged[col] = merged[supp_col]
    drop_cols = [c for c in merged.columns if c.endswith("__supp")]
    return merged.drop(columns=drop_cols, errors="ignore")


def _to_num(s):
    return pd.to_numeric(s, errors="coerce")


def _enrich_backtest_from_reference_sources(df, league, year):
    if df is None or df.empty:
        return df
    out = _consolidate_frame(_with_backtest_key(df))

    # results: stadium/datetime/score 補完キー
    result_path = os.path.join(ROOT_DIR, "data", f"{league}_{year}_latest_results.csv")
    res = None
    if os.path.exists(result_path):
        try:
            res = pd.read_csv(result_path)
        except Exception:
            res = None
    if res is not None and not res.empty and {"home_team", "away_team"}.issubset(res.columns):
        r = _with_backtest_key(res)
        # match_id 直結
        rid = r.dropna(subset=["match_id"]).drop_duplicates("match_id", keep="last")
        out = out.merge(
            rid[["match_id", "datetime", "stadium", "home_score", "away_score", "__pair_day"]].rename(
                columns={
                    "datetime": "datetime__res",
                    "stadium": "stadium__res",
                    "home_score": "home_score__res",
                    "away_score": "away_score__res",
                    "__pair_day": "__pair_day__res",
                }
            ),
            on="match_id",
            how="left",
        )
        # pair_day 補完
        rpd = r.drop_duplicates("__pair_day", keep="last")
        out = out.merge(
            rpd[["__pair_day", "datetime", "stadium", "home_score", "away_score", "match_id"]].rename(
                columns={
                    "datetime": "datetime__res_pd",
                    "stadium": "stadium__res_pd",
                    "home_score": "home_score__res_pd",
                    "away_score": "away_score__res_pd",
                    "match_id": "match_id__res_pd",
                }
            ),
            on="__pair_day",
            how="left",
        )
        # map用 match_id（weather/fatigue fallback）
        out["__match_id_from_results"] = out["match_id__res_pd"]
        _fill_if_missing(out, "datetime", out.get("datetime__res").fillna(out.get("datetime__res_pd")))
        _fill_if_missing(out, "stadium", out.get("stadium__res").fillna(out.get("stadium__res_pd")))
        if "home_score" in out.columns:
            hs = _to_num(out["home_score"])
            hs = hs.fillna(_to_num(out.get("home_score__res"))).fillna(_to_num(out.get("home_score__res_pd")))
            out["home_score"] = hs
        if "away_score" in out.columns:
            aw = _to_num(out["away_score"])
            aw = aw.fillna(_to_num(out.get("away_score__res"))).fillna(_to_num(out.get("away_score__res_pd")))
            out["away_score"] = aw

    alt_league = "j2" if str(league).lower() == "j1" else ("j1" if str(league).lower() == "j2" else None)

    # fatigue (J1/J2 横断参照)
    fat_parts = []
    fatigue_paths = [os.path.join(ROOT_DIR, "data", f"team_fatigue_scores_{league}_{year}.csv")]
    if alt_league:
        fatigue_paths.append(os.path.join(ROOT_DIR, "data", f"team_fatigue_scores_{alt_league}_{year}.csv"))
    for fatigue_path in fatigue_paths:
        if not os.path.exists(fatigue_path):
            continue
        try:
            fat_parts.append(pd.read_csv(fatigue_path))
        except Exception:
            continue
    fat = pd.concat(fat_parts, ignore_index=True) if fat_parts else None
    if fat is not None and not fat.empty:
        need_cols = {"match_id", "home_fatigue_score", "away_fatigue_score"}
        if need_cols.issubset(fat.columns):
            fid = fat[["match_id", "home_fatigue_score", "away_fatigue_score"]].copy()
            fid["match_id"] = fid["match_id"].astype(str).str.strip()
            fid = fid[fid["match_id"].ne("") & fid["match_id"].ne("nan")]
            fid = fid.drop_duplicates("match_id", keep="last")
            fill_home = pd.Series(pd.NA, index=out.index, dtype="object")
            fill_away = pd.Series(pd.NA, index=out.index, dtype="object")
            if not fid.empty:
                out = out.merge(
                    fid.rename(
                        columns={
                            "home_fatigue_score": "home_fatigue_score__fat",
                            "away_fatigue_score": "away_fatigue_score__fat",
                        }
                    ),
                    on="match_id",
                    how="left",
                )
                fill_home = out.get("home_fatigue_score__fat")
                fill_away = out.get("away_fatigue_score__fat")
                if "__match_id_from_results" in out.columns:
                    out = out.merge(
                        fid.rename(
                            columns={
                                "match_id": "__match_id_from_results",
                                "home_fatigue_score": "home_fatigue_score__fat_res",
                                "away_fatigue_score": "away_fatigue_score__fat_res",
                            }
                        ),
                        on="__match_id_from_results",
                        how="left",
                    )
                    fill_home = _to_num(fill_home).fillna(_to_num(out.get("home_fatigue_score__fat_res")))
                    fill_away = _to_num(fill_away).fillna(_to_num(out.get("away_fatigue_score__fat_res")))
            if "home_fatigue_score" in out.columns:
                _fill_if_missing(out, "home_fatigue_score", fill_home)
            if "away_fatigue_score" in out.columns:
                _fill_if_missing(out, "away_fatigue_score", fill_away)

    # weather snapshot (J1/J2 横断参照)
    wdir = os.path.join(ROOT_DIR, "data", "weather_snapshots")
    wparts = []
    for lg in [league, alt_league]:
        if not lg:
            continue
        wpath = _latest_snapshot_path(wdir, f"weather_features_{lg}_{year}")
        if wpath and os.path.exists(wpath):
            try:
                wparts.append(pd.read_csv(wpath))
            except Exception:
                pass
    w = pd.concat(wparts, ignore_index=True) if wparts else None
    if w is not None and not w.empty:
        if w is not None and not w.empty and "match_id" in w.columns:
            wid = w.drop_duplicates("match_id", keep="last").copy()
            wid = wid.rename(
                columns={
                    "kickoff_jst": "datetime_weather__w",
                    "stadium_name": "stadium_weather__w",
                    "temp_kickoff": "temperature__w",
                    "wind_kickoff": "wind_speed__w",
                }
            )
            out = out.merge(
                wid[
                    [
                        "match_id",
                        "datetime_weather__w",
                        "stadium_weather__w",
                        "temperature__w",
                        "wind_speed__w",
                        "is_rain",
                        "is_heavy_rain",
                        "is_strong_wind",
                        "weather_fetch_ok",
                    ]
                ].rename(
                    columns={
                        "is_rain": "is_rain__w",
                        "is_heavy_rain": "is_heavy_rain__w",
                        "is_strong_wind": "is_strong_wind__w",
                        "weather_fetch_ok": "weather_fetch_ok__w",
                    }
                ),
                on="match_id",
                how="left",
            )
            if "__match_id_from_results" in out.columns:
                out = out.merge(
                    wid[
                        [
                            "match_id",
                            "datetime_weather__w",
                            "stadium_weather__w",
                            "temperature__w",
                            "wind_speed__w",
                            "is_rain",
                            "is_heavy_rain",
                            "is_strong_wind",
                            "weather_fetch_ok",
                        ]
                    ].rename(
                        columns={
                            "match_id": "__match_id_from_results",
                            "datetime_weather__w": "datetime_weather__w_res",
                            "stadium_weather__w": "stadium_weather__w_res",
                            "temperature__w": "temperature__w_res",
                            "wind_speed__w": "wind_speed__w_res",
                            "is_rain": "is_rain__w_res",
                            "is_heavy_rain": "is_heavy_rain__w_res",
                            "is_strong_wind": "is_strong_wind__w_res",
                            "weather_fetch_ok": "weather_fetch_ok__w_res",
                        }
                    ),
                    on="__match_id_from_results",
                    how="left",
                )
            _fill_if_missing(
                out,
                "datetime_weather",
                out.get("datetime_weather__w").fillna(out.get("datetime_weather__w_res")),
            )
            _fill_if_missing(
                out,
                "stadium_weather",
                out.get("stadium_weather__w").fillna(out.get("stadium_weather__w_res")),
            )
            _fill_if_missing(out, "temperature", _to_num(out.get("temperature__w")).fillna(_to_num(out.get("temperature__w_res"))))
            _fill_if_missing(out, "wind_speed", _to_num(out.get("wind_speed__w")).fillna(_to_num(out.get("wind_speed__w_res"))))
            for c in ["is_rain", "is_heavy_rain", "is_strong_wind"]:
                _fill_if_missing(out, c, out.get(f"{c}__w").fillna(out.get(f"{c}__w_res")))
            if "weather_missing" in out.columns:
                wf = out.get("weather_fetch_ok__w").fillna(out.get("weather_fetch_ok__w_res"))
                miss = wf.map(lambda v: pd.NA if pd.isna(v) else (not bool(v)))
                _fill_if_missing(out, "weather_missing", miss)
            _fill_if_missing(out, "stadium", out.get("stadium_weather__w").fillna(out.get("stadium_weather__w_res")))

    # stats (latest snapshot, J1/J2 横断参照)
    sdir = os.path.join(ROOT_DIR, "data", "stats_snapshots")
    stats_parts = []
    stats_src_names = []
    for lg in [league, alt_league]:
        if not lg:
            continue
        spath = _latest_snapshot_path(sdir, f"team_master_stats_{lg}_{year}")
        if spath and os.path.exists(spath):
            try:
                stats_parts.append(pd.read_csv(spath))
                stats_src_names.append(os.path.basename(spath))
            except Exception:
                pass
    st = pd.concat(stats_parts, ignore_index=True) if stats_parts else None
    if st is not None and not st.empty:
        if st is not None and not st.empty and "team_name" in st.columns:
            st = st.copy()
            st["__team_k"] = _team_key_series(st["team_name"])
            out["__home_k"] = _team_key_series(out.get("home_team", pd.Series(index=out.index, dtype="object")))
            out["__away_k"] = _team_key_series(out.get("away_team", pd.Series(index=out.index, dtype="object")))
            home_map = st.drop_duplicates("__team_k", keep="last").set_index("__team_k")
            away_map = home_map
            for mcol in [c for c in st.columns if c not in {"team_name", "__team_k"}]:
                hcol = f"stats_{mcol}_home"
                acol = f"stats_{mcol}_away"
                if hcol in out.columns:
                    _fill_if_missing(out, hcol, out["__home_k"].map(home_map[mcol]))
                if acol in out.columns:
                    _fill_if_missing(out, acol, out["__away_k"].map(away_map[mcol]))
            if "stats_asof" in out.columns:
                # 現行運用に合わせ、リーグ主系列の最新asofを設定
                spath_main = _latest_snapshot_path(sdir, f"team_master_stats_{league}_{year}")
                asof_txt = os.path.basename(spath_main).split("_asof_")[-1].replace(".csv", "") if spath_main else ""
                try:
                    asof_txt = f"{asof_txt[:4]}-{asof_txt[4:6]}-{asof_txt[6:8]}"
                except Exception:
                    pass
                _fill_if_missing(out, "stats_asof", pd.Series(asof_txt, index=out.index))
            if "stats_source_csv" in out.columns:
                src_txt = ",".join(stats_src_names) if stats_src_names else ""
                _fill_if_missing(out, "stats_source_csv", pd.Series(src_txt, index=out.index))

    # motivation / rankmot (J1/J2 横断参照)
    mot_parts = []
    for lg in [league, alt_league]:
        if not lg:
            continue
        mpath = os.path.join(ROOT_DIR, "data", f"{lg}_{year}_motivation.csv")
        if not os.path.exists(mpath):
            continue
        try:
            mot_parts.append(pd.read_csv(mpath))
        except Exception:
            continue
    mt = pd.concat(mot_parts, ignore_index=True) if mot_parts else None
    if mt is not None and not mt.empty:
        if mt is not None and not mt.empty and "team_name" in mt.columns:
            mt = mt.copy()
            mt["__team_k"] = _team_key_series(mt["team_name"])
            mref = mt.drop_duplicates("__team_k", keep="last").set_index("__team_k")
            out["__home_k"] = _team_key_series(out.get("home_team", pd.Series(index=out.index, dtype="object")))
            out["__away_k"] = _team_key_series(out.get("away_team", pd.Series(index=out.index, dtype="object")))
            for bcol in [c for c in mt.columns if c not in {"team_name", "__team_k"}]:
                hcol = f"rankmot_{bcol}_home"
                acol = f"rankmot_{bcol}_away"
                if hcol in out.columns:
                    _fill_if_missing(out, hcol, out["__home_k"].map(mref[bcol]))
                if acol in out.columns:
                    _fill_if_missing(out, acol, out["__away_k"].map(mref[bcol]))

    # management master
    mgpath = os.path.join(ROOT_DIR, "data", "manual", "team_management_master.csv")
    if os.path.exists(mgpath):
        try:
            mg = pd.read_csv(mgpath)
        except Exception:
            mg = None
        if mg is not None and not mg.empty and "team_name" in mg.columns:
            mg = mg.copy()
            mg["__team_k"] = _team_key_series(mg["team_name"])
            mgref = mg.drop_duplicates("__team_k", keep="last").set_index("__team_k")
            out["__home_k"] = _team_key_series(out.get("home_team", pd.Series(index=out.index, dtype="object")))
            out["__away_k"] = _team_key_series(out.get("away_team", pd.Series(index=out.index, dtype="object")))
            colmap = {
                "manager_name": "management_manager_name",
                "recent_injuries_suspensions_count": "management_recent_injuries_suspensions_count",
                "weather_influence_score": "management_weather_influence_score",
                "motivation_score": "management_motivation_score",
            }
            for src, dstbase in colmap.items():
                hcol = f"{dstbase}_home"
                acol = f"{dstbase}_away"
                if src in mgref.columns and hcol in out.columns:
                    _fill_if_missing(out, hcol, out["__home_k"].map(mgref[src]))
                if src in mgref.columns and acol in out.columns:
                    _fill_if_missing(out, acol, out["__away_k"].map(mgref[src]))

    # actual_result / is_correct の再計算
    if "actual_result" in out.columns:
        hs = _to_num(out.get("home_score"))
        aw = _to_num(out.get("away_score"))
        actual = pd.Series(pd.NA, index=out.index, dtype="object")
        actual = actual.mask(hs > aw, "H").mask(hs < aw, "A").mask((hs == aw) & hs.notna() & aw.notna(), "D")
        _fill_if_missing(out, "actual_result", actual)
    if "is_correct" in out.columns and "predicted_result" in out.columns and "actual_result" in out.columns:
        pred = out["predicted_result"].astype("string").str.upper()
        corr = (pred == out["actual_result"]).where(out["actual_result"].notna(), pd.NA)
        _fill_if_missing(out, "is_correct", corr)

    drop_cols = [c for c in out.columns if c.startswith("__") or c.endswith("__res") or c.endswith("__res_pd") or c.endswith("__fat") or c.endswith("__fat_res") or c.endswith("__w") or c.endswith("__w_res")]
    out = out.drop(columns=drop_cols, errors="ignore")
    return out


def build_round_views():
    cleaned_prediction_pairs = set()
    cleaned_backtest_pairs = set()
    cleaned_prediction_all_years = set()
    cleaned_backtest_all_years = set()
    toto_round_map = load_toto_round_map()
    toto_target_history = load_toto_target_history()
    pred_files = [f for f in os.listdir(ROOT_DIR) if f.endswith("_predictions.csv")]
    include_rounds_backtest = os.environ.get("INCLUDE_ROUNDS_BACKTEST", "1") == "1"
    backtest_files = [
        f
        for f in os.listdir(ROOT_DIR)
        if f.startswith("backtest_")
        and f.endswith(".csv")
        and (include_rounds_backtest or (not f.endswith("_rounds.csv")))
    ]
    backtest_files += [os.path.relpath(p, ROOT_DIR) for p in sorted(
        [os.path.join(ROOT_DIR, "data", f) for f in os.listdir(os.path.join(ROOT_DIR, "data")) if f.startswith("round") and f.endswith("_backtest.csv")]
    )]

    merged_pred_by_round = {}
    merged_backtest_by_round = {}
    backtest_round_supplement = {}
    results_round_supplement = {}
    trial_flag_supplement = {}
    league_team_pool = {}

    # 予測CSVからリーグごとのチーム集合を作る（結果CSV補完時の混入防止）
    for fname in pred_files:
        league, year = parse_league_year_from_predictions_filename(fname)
        if league == "na" or year == "na":
            continue
        path = os.path.join(ROOT_DIR, fname)
        try:
            pdf = pd.read_csv(path, usecols=lambda c: c in {"home_team", "away_team"})
        except Exception:
            continue
        teams = set()
        if "home_team" in pdf.columns:
            teams.update(pdf["home_team"].dropna().astype(str).str.strip().tolist())
        if "away_team" in pdf.columns:
            teams.update(pdf["away_team"].dropna().astype(str).str.strip().tolist())
        league_team_pool[(league, year)] = teams

    # 予想ページの試合数不足を避けるため、同リーグ/同節のbacktest行を補完候補として読み込む。
    backtest_candidates = {}
    for fname in sorted(backtest_files):
        path = os.path.join(ROOT_DIR, fname)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        league, year = parse_league_year_from_backtest_filename(fname)
        if league == "na" or year == "na":
            continue
        round_col = "節" if "節" in df.columns else None
        if not round_col:
            continue
        df = df.copy()
        df["_round_label"] = df[round_col].apply(round_label_from_section)
        for rnd, sub in df.groupby("_round_label"):
            team_pool = league_team_pool.get((league, year), set())
            if team_pool and "home_team" in sub.columns and "away_team" in sub.columns:
                sub = sub[
                    sub["home_team"].astype(str).str.strip().isin(team_pool)
                    & sub["away_team"].astype(str).str.strip().isin(team_pool)
                ]
            key = (league, year, rnd)
            payload = _without_columns(sub, ["_round_label"], copy=True)
            if key in backtest_round_supplement:
                backtest_round_supplement[key] = pd.concat(
                    [backtest_round_supplement[key], payload], ignore_index=True
                )
            else:
                backtest_round_supplement[key] = payload

    # さらに、結果CSVから同節の試合を補完候補として読み込む（予測未保持の試合を埋める）。
    for fname in pred_files:
        league, year = parse_league_year_from_predictions_filename(fname)
        if league == "na" or year == "na":
            continue
        result_path = os.path.join(ROOT_DIR, "data", f"{league}_{year}_latest_results.csv")
        if not os.path.exists(result_path):
            continue
        try:
            rdf = pd.read_csv(result_path)
        except Exception:
            continue
        round_col = "節" if "節" in rdf.columns else ("section" if "section" in rdf.columns else None)
        if not round_col:
            continue
        rdf = rdf.copy()
        rdf["_round_label"] = rdf[round_col].apply(round_label_from_section)
        for rnd, sub in rdf.groupby("_round_label"):
            team_pool = league_team_pool.get((league, year), set())
            if team_pool and "home_team" in sub.columns and "away_team" in sub.columns:
                sub = sub[
                    sub["home_team"].astype(str).str.strip().isin(team_pool)
                    & sub["away_team"].astype(str).str.strip().isin(team_pool)
                ]
            results_round_supplement[(league, year, rnd)] = _without_columns(sub, ["_round_label"], copy=True)

    trial_dir = os.path.join(ROOT_DIR, "data", "external_metrics", "trial_flags")
    if os.path.isdir(trial_dir):
        for path in sorted(glob.glob(os.path.join(trial_dir, "*.csv"))):
            try:
                tdf = pd.read_csv(path)
            except Exception:
                continue
            if tdf.empty or "match_id" not in tdf.columns:
                continue
            round_col = "節" if "節" in tdf.columns else ("section" if "section" in tdf.columns else None)
            if not round_col:
                continue
            tdf = tdf.copy()
            tdf["_round_label"] = tdf[round_col].apply(round_label_from_section)
            if "league" in tdf.columns:
                tdf["league"] = tdf["league"].fillna("").astype(str).str.strip().str.lower()
                miss_lg = tdf["league"].isin(["", "nan", "none", "nat", "<na>"])
                if miss_lg.any():
                    tdf.loc[miss_lg, "league"] = (
                        tdf.loc[miss_lg, "match_id"]
                        .astype(str)
                        .str.extract(r"^(j[123])_", flags=re.IGNORECASE)[0]
                        .str.lower()
                    )
            else:
                tdf["league"] = tdf["match_id"].astype(str).str.extract(r"^(j[123])_", flags=re.IGNORECASE)[0].str.lower()
            for rnd, sub in tdf.groupby("_round_label"):
                for lg, block in _iter_blocks_by_column(sub, "league"):
                    if not isinstance(lg, str) or not re.match(r"^j[123]$", lg):
                        continue
                    year = "na"
                    for mid in block["match_id"].dropna().astype(str):
                        _lg, _yr = parse_league_year_from_match_id(mid)
                        if _lg == lg and _yr != "na":
                            year = _yr
                            break
                    if year == "na":
                        continue
                    key = (lg, year, rnd)
                    payload = _without_columns(block, ["_round_label"], copy=True)
                    if key in trial_flag_supplement:
                        trial_flag_supplement[key] = pd.concat([trial_flag_supplement[key], payload], ignore_index=True)
                    else:
                        trial_flag_supplement[key] = payload

    for fname in pred_files:
        path = os.path.join(ROOT_DIR, fname)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        league, year = parse_league_year_from_predictions_filename(fname)
        round_col = "節" if "節" in df.columns else ("section" if "section" in df.columns else None)
        if not round_col:
            continue
        df = df.copy()
        df["_round_label"] = df[round_col].apply(round_label_from_section)
        for rnd, sub in df.groupby("_round_label"):
            out_df = _without_columns(sub, ["_round_label"])
            supp_key = (league, year, rnd)
            supp_df = backtest_round_supplement.get(supp_key)
            if supp_df is not None and not supp_df.empty:
                if "match_id" in out_df.columns and "match_id" in supp_df.columns:
                    missing = supp_df[~supp_df["match_id"].isin(out_df["match_id"])]
                    if not missing.empty:
                        out_df = pd.concat([out_df, missing], ignore_index=True)
                else:
                    out_df = pd.concat([out_df, supp_df], ignore_index=True).drop_duplicates()
            res_df = results_round_supplement.get(supp_key)
            if res_df is not None and not res_df.empty and "match_id" in out_df.columns and "match_id" in res_df.columns:
                missing_res = res_df[~res_df["match_id"].isin(out_df["match_id"])]
                if not missing_res.empty:
                    aligned = missing_res.reindex(columns=out_df.columns, fill_value=pd.NA)
                    # 共通列のみ結果CSVの値で埋める
                    common_cols = [c for c in out_df.columns if c in missing_res.columns]
                    for c in common_cols:
                        aligned[c] = missing_res[c].values
                    out_df = pd.concat([out_df, aligned], ignore_index=True)
            trial_df = trial_flag_supplement.get(supp_key)
            if trial_df is not None and not trial_df.empty:
                out_df = _merge_prediction_supplement(out_df, trial_df)
            out_df = _dedupe_prediction_rows(out_df)
            out_df = reorder_prediction_columns(out_df)
            pair_key = (league, year)
            if pair_key not in cleaned_prediction_pairs:
                if os.path.exists(HTML_DIR):
                    for old_name in os.listdir(HTML_DIR):
                        if old_name.startswith(f"predictions_round_{league}_{year}_") and old_name.endswith(".html"):
                            os.remove(os.path.join(HTML_DIR, old_name))
                if os.path.exists(CSV_DIR):
                    for old_name in os.listdir(CSV_DIR):
                        if old_name.startswith(f"predictions_round_{league}_{year}_") and old_name.endswith(".csv"):
                            os.remove(os.path.join(CSV_DIR, old_name))
                cleaned_prediction_pairs.add(pair_key)
            os.makedirs(HTML_DIR, exist_ok=True)
            out_name = f"predictions_round_{league}_{year}_{sanitize_round(rnd)}.html"
            out_path = os.path.join(HTML_DIR, out_name)
            title = f"Predictions {league.upper()} {year} {rnd}: {fname}"
            write_html_table(out_df, title, PRED_DESC_MAP, out_path)
            os.makedirs(CSV_DIR, exist_ok=True)
            csv_name = f"predictions_round_{league}_{year}_{sanitize_round(rnd)}.csv"
            csv_path = os.path.join(CSV_DIR, csv_name)
            out_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            merged_key = (year, rnd)
            merged_part = ensure_league_column(out_df, league)
            merged_part = reorder_prediction_columns(merged_part)
            merged_pred_by_round.setdefault(merged_key, []).append(merged_part)

    for (year, rnd), parts in merged_pred_by_round.items():
        if len({str(part["league"].iloc[0]).strip().lower() for part in parts if not part.empty and "league" in part.columns}) < 2:
            continue
        merged_df = pd.concat(parts, ignore_index=True)
        merged_df = _dedupe_prediction_rows(merged_df)
        merged_df = reorder_prediction_columns(merged_df)
        if "datetime_x" in merged_df.columns:
            merged_df["datetime_x"] = pd.to_datetime(merged_df["datetime_x"], errors="coerce")
            merged_df = merged_df.sort_values(["league", "datetime_x"], na_position="last").reset_index(drop=True)
        elif "datetime" in merged_df.columns:
            merged_df["datetime"] = pd.to_datetime(merged_df["datetime"], errors="coerce")
            merged_df = merged_df.sort_values(["league", "datetime"], na_position="last").reset_index(drop=True)
        else:
            merged_df = merged_df.sort_values(["league"]).reset_index(drop=True)

        if year not in cleaned_prediction_all_years:
            if os.path.exists(HTML_DIR):
                for old_name in os.listdir(HTML_DIR):
                    if old_name.startswith(f"predictions_round_all_{year}_") and old_name.endswith(".html"):
                        os.remove(os.path.join(HTML_DIR, old_name))
            if os.path.exists(CSV_DIR):
                for old_name in os.listdir(CSV_DIR):
                    if old_name.startswith(f"predictions_round_all_{year}_") and old_name.endswith(".csv"):
                        os.remove(os.path.join(CSV_DIR, old_name))
            cleaned_prediction_all_years.add(year)
        os.makedirs(HTML_DIR, exist_ok=True)
        out_name = f"predictions_round_all_{year}_{sanitize_round(rnd)}.html"
        out_path = os.path.join(HTML_DIR, out_name)
        title = f"Predictions J1+J2 {year} {rnd}"
        desc_map = dict(PRED_DESC_MAP)
        desc_map["league"] = "リーグ"
        write_html_table(merged_df, title, desc_map, out_path)
        os.makedirs(CSV_DIR, exist_ok=True)
        csv_name = f"predictions_round_all_{year}_{sanitize_round(rnd)}.csv"
        csv_path = os.path.join(CSV_DIR, csv_name)
        merged_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    for fname in backtest_files:
        path = os.path.join(ROOT_DIR, fname)
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        league, year = parse_league_year_from_backtest_filename(fname)
        if league == "na" or year == "na":
            continue
        round_col = "節" if "節" in df.columns else None
        if not round_col:
            continue
        df = df.copy()
        df = _filter_backtest_by_match_id_league(df, league)
        df["_round_label"] = df[round_col].apply(round_label_from_section)
        src_rank = 0 if not fname.endswith("_rounds.csv") else 1
        for rnd, sub in df.groupby("_round_label"):
            out_df = _without_columns(sub, ["_round_label"], copy=True)
            key = (league, year, rnd)
            backtest_candidates.setdefault(key, []).append((src_rank, fname, out_df))

    # roundスナップショット（購入評価用 predictions.csv）も補完候補に加える。
    # これらは detail 列が多く、*_rounds.csv の欠落補完に有効。
    snapshot_patterns = [
        os.path.join(ROOT_DIR, "data", "eval", "rounds", "round*", "snapshot", "predictions.csv"),
        os.path.join(ROOT_DIR, "data", "eval", "rounds", "round*", "snapshot_*", "predictions.csv"),
    ]
    snapshot_files = []
    for pat in snapshot_patterns:
        snapshot_files.extend(glob.glob(pat))
    for path in sorted(set(snapshot_files)):
        try:
            sdf = pd.read_csv(path)
        except Exception:
            continue
        if sdf.empty or "節" not in sdf.columns:
            continue
        if "match_id" not in sdf.columns:
            continue
        league_col = "league"
        sdf = sdf.copy()
        if league_col in sdf.columns:
            sdf[league_col] = sdf[league_col].astype(str).str.strip().str.lower()
        else:
            sdf[league_col] = sdf["match_id"].astype(str).str.extract(r"^(j[123])_", flags=re.IGNORECASE)[0].str.lower()
        sdf["_round_label"] = sdf["節"].apply(round_label_from_section)
        src_name = os.path.relpath(path, ROOT_DIR)
        for rnd, sub in sdf.groupby("_round_label"):
            for lg, block in _iter_blocks_by_column(sub, league_col):
                if not isinstance(lg, str) or not re.match(r"^j[123]$", lg):
                    continue
                # year は match_id から推定
                year = "na"
                for mid in block["match_id"].dropna().astype(str):
                    _lg, _yr = parse_league_year_from_match_id(mid)
                    if _lg == lg and _yr != "na":
                        year = _yr
                        break
                if year == "na":
                    continue
                out_df = _without_columns(block, ["_round_label"], copy=True)
                key = (lg, year, rnd)
                # rank=2: 通常backtest(0)より低いが、*_rounds(1)よりは情報量で勝てるようにする。
                backtest_candidates.setdefault(key, []).append((2, src_name, out_df))

    for (league, year, rnd), candidates in backtest_candidates.items():
        out_df = _merge_backtest_candidates(candidates)
        out_df = _enrich_backtest_with_results(out_df, league, year)
        out_df = _enrich_backtest_from_reference_sources(out_df, league, year)
        out_df = _keep_finished_backtest_rows(out_df)
        out_df = ensure_league_column(out_df, league)
        out_df = reorder_backtest_columns(out_df)
        pair_key = (league, year)
        if pair_key not in cleaned_backtest_pairs and os.path.exists(HTML_DIR):
            for old_name in os.listdir(HTML_DIR):
                if old_name.startswith(f"backtest_round_{league}_{year}_") and old_name.endswith(".html"):
                    os.remove(os.path.join(HTML_DIR, old_name))
            cleaned_backtest_pairs.add(pair_key)
        os.makedirs(HTML_DIR, exist_ok=True)
        out_name = f"backtest_round_{league}_{year}_{sanitize_round(rnd)}.html"
        out_path = os.path.join(HTML_DIR, out_name)
        title = f"Backtest {league.upper()} {year} {rnd}"
        write_html_table(out_df, title, BACKTEST_DESC_MAP, out_path)
        merged_key = (year, rnd)
        merged_part = ensure_league_column(out_df, league)
        merged_backtest_by_round.setdefault(merged_key, []).append(merged_part)

    for (year, rnd), parts in merged_backtest_by_round.items():
        if len({str(part["league"].iloc[0]).strip().lower() for part in parts if not part.empty and "league" in part.columns}) < 2:
            continue
        merged_df = pd.concat(parts, ignore_index=True)
        merged_df = _keep_finished_backtest_rows(merged_df)
        merged_df = _dedupe_merged_backtest_round(merged_df)
        merged_df = reorder_backtest_columns(merged_df)
        if "datetime" in merged_df.columns:
            merged_df["datetime"] = pd.to_datetime(merged_df["datetime"], errors="coerce")
            merged_df = merged_df.sort_values(["league", "datetime"], na_position="last").reset_index(drop=True)
        else:
            merged_df = merged_df.sort_values(["league"]).reset_index(drop=True)

        if year not in cleaned_backtest_all_years and os.path.exists(HTML_DIR):
            for old_name in os.listdir(HTML_DIR):
                if old_name.startswith(f"backtest_round_all_{year}_") and old_name.endswith(".html"):
                    os.remove(os.path.join(HTML_DIR, old_name))
            cleaned_backtest_all_years.add(year)
        os.makedirs(HTML_DIR, exist_ok=True)
        out_name = f"backtest_round_all_{year}_{sanitize_round(rnd)}.html"
        out_path = os.path.join(HTML_DIR, out_name)
        title = f"Backtest J1+J2 {year} {rnd}"
        desc_map = dict(BACKTEST_DESC_MAP)
        desc_map["league"] = "リーグ"
        write_html_table(merged_df, title, desc_map, out_path)

        round_no = extract_round_number(rnd)
        toto_round = toto_round_map.get((str(year), int(round_no))) if round_no is not None else None
        if toto_round:
            toto_df = build_toto_target_backtest_view(merged_df, year, toto_round, toto_target_history)
            if toto_df.empty:
                toto_df = merged_df.copy()
            toto_html_name = f"backtest_toto_all_{year}_toto{toto_round}.html"
            toto_html_path = os.path.join(HTML_DIR, toto_html_name)
            toto_title = f"Backtest TOTO {toto_round} ({year} {rnd})"
            write_html_table(toto_df, toto_title, desc_map, toto_html_path)
            os.makedirs(CSV_DIR, exist_ok=True)
            toto_csv_name = f"backtest_toto_all_{year}_toto{toto_round}.csv"
            toto_csv_path = os.path.join(CSV_DIR, toto_csv_name)
            toto_df.to_csv(toto_csv_path, index=False, encoding="utf-8-sig")

    print(f"出力: {HTML_DIR}")


if __name__ == "__main__":
    build_round_views()
