import os
import re
import pandas as pd
from report_view_utils import write_html_table


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
REPORT_DIR = os.path.abspath(os.path.dirname(__file__))
HTML_DIR = os.path.join(REPORT_DIR, "html")
CSV_DIR = os.path.join(REPORT_DIR, "csv")

PRED_DESC_MAP = {
    "節": "節",
    "section": "節",
    "date": "日付",
    "match_id": "試合ID",
    "datetime": "試合日時",
    "stadium": "スタジアム",
    "home_team": "ホーム",
    "away_team": "アウェイ",
    "home_score": "ホーム得点（予測時は空）",
    "away_score": "アウェイ得点（予測時は空）",
    "prob_home_win": "ホーム勝ち確率",
    "prob_draw": "引き分け確率",
    "prob_away_win": "アウェイ勝ち確率",
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
}


def sanitize_round(value):
    text = str(value).strip()
    return re.sub(r"[\\s/\\\\]+", "_", text)


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


def build_round_views():
    if os.path.exists(HTML_DIR):
        for old_name in os.listdir(HTML_DIR):
            if old_name.startswith("predictions_round_") and old_name.endswith(".html"):
                os.remove(os.path.join(HTML_DIR, old_name))
            if old_name.startswith("backtest_round_") and old_name.endswith(".html"):
                os.remove(os.path.join(HTML_DIR, old_name))
    if os.path.exists(CSV_DIR):
        for old_name in os.listdir(CSV_DIR):
            if old_name.startswith("predictions_round_") and old_name.endswith(".csv"):
                os.remove(os.path.join(CSV_DIR, old_name))
    pred_files = [f for f in os.listdir(ROOT_DIR) if f.endswith("_predictions.csv")]
    backtest_files = [f for f in os.listdir(ROOT_DIR) if f.startswith("backtest_") and f.endswith(".csv")]
    backtest_files += [os.path.relpath(p, ROOT_DIR) for p in sorted(
        [os.path.join(ROOT_DIR, "data", f) for f in os.listdir(os.path.join(ROOT_DIR, "data")) if f.startswith("round") and f.endswith("_backtest.csv")]
    )]

    merged_pred_by_round = {}
    merged_backtest_by_round = {}
    backtest_round_supplement = {}
    results_round_supplement = {}
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
        df["_round_label"] = df[round_col].apply(round_label_from_section)
        for rnd, sub in df.groupby("_round_label"):
            team_pool = league_team_pool.get((league, year), set())
            if team_pool and "home_team" in sub.columns and "away_team" in sub.columns:
                sub = sub[
                    sub["home_team"].astype(str).str.strip().isin(team_pool)
                    & sub["away_team"].astype(str).str.strip().isin(team_pool)
                ]
            key = (league, year, rnd)
            payload = sub.drop(columns=["_round_label"], errors="ignore").copy()
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
        rdf["_round_label"] = rdf[round_col].apply(round_label_from_section)
        for rnd, sub in rdf.groupby("_round_label"):
            team_pool = league_team_pool.get((league, year), set())
            if team_pool and "home_team" in sub.columns and "away_team" in sub.columns:
                sub = sub[
                    sub["home_team"].astype(str).str.strip().isin(team_pool)
                    & sub["away_team"].astype(str).str.strip().isin(team_pool)
                ]
            results_round_supplement[(league, year, rnd)] = sub.drop(columns=["_round_label"], errors="ignore").copy()

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
        df["_round_label"] = df[round_col].apply(round_label_from_section)
        for rnd, sub in df.groupby("_round_label"):
            out_df = sub.drop(columns=["_round_label"], errors="ignore")
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
            merged_part = out_df.copy()
            merged_part.insert(0, "league", league.upper())
            merged_pred_by_round.setdefault(merged_key, []).append(merged_part)

    for (year, rnd), parts in merged_pred_by_round.items():
        merged_df = pd.concat(parts, ignore_index=True)
        if "datetime_x" in merged_df.columns:
            merged_df["datetime_x"] = pd.to_datetime(merged_df["datetime_x"], errors="coerce")
            merged_df = merged_df.sort_values(["league", "datetime_x"], na_position="last").reset_index(drop=True)
        elif "datetime" in merged_df.columns:
            merged_df["datetime"] = pd.to_datetime(merged_df["datetime"], errors="coerce")
            merged_df = merged_df.sort_values(["league", "datetime"], na_position="last").reset_index(drop=True)
        else:
            merged_df = merged_df.sort_values(["league"]).reset_index(drop=True)

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
        df["_round_label"] = df[round_col].apply(round_label_from_section)
        for rnd, sub in df.groupby("_round_label"):
            out_df = sub.drop(columns=["_round_label"], errors="ignore")
            os.makedirs(HTML_DIR, exist_ok=True)
            out_name = f"backtest_round_{league}_{year}_{sanitize_round(rnd)}.html"
            out_path = os.path.join(HTML_DIR, out_name)
            title = f"Backtest {league.upper()} {year} {rnd}: {fname}"
            write_html_table(out_df, title, BACKTEST_DESC_MAP, out_path)
            merged_key = (year, rnd)
            merged_part = out_df.copy()
            merged_part.insert(0, "league", league.upper())
            merged_backtest_by_round.setdefault(merged_key, []).append(merged_part)

    for (year, rnd), parts in merged_backtest_by_round.items():
        merged_df = pd.concat(parts, ignore_index=True)
        if "datetime" in merged_df.columns:
            merged_df["datetime"] = pd.to_datetime(merged_df["datetime"], errors="coerce")
            merged_df = merged_df.sort_values(["league", "datetime"], na_position="last").reset_index(drop=True)
        else:
            merged_df = merged_df.sort_values(["league"]).reset_index(drop=True)

        os.makedirs(HTML_DIR, exist_ok=True)
        out_name = f"backtest_round_all_{year}_{sanitize_round(rnd)}.html"
        out_path = os.path.join(HTML_DIR, out_name)
        title = f"Backtest J1+J2 {year} {rnd}"
        desc_map = dict(BACKTEST_DESC_MAP)
        desc_map["league"] = "リーグ"
        write_html_table(merged_df, title, desc_map, out_path)

    print(f"出力: {HTML_DIR}")


if __name__ == "__main__":
    build_round_views()
