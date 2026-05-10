#!/usr/bin/env python3
import argparse
import html
import os
import re
import sys
from datetime import datetime

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def eval_base_dir(round_id: str) -> str:
    if str(round_id).startswith("toto"):
        return os.path.join(ROOT_DIR, "data", "eval", "toto_rounds", round_id)
    return os.path.join(ROOT_DIR, "data", "eval", "rounds", round_id)


SCENARIO_JA_BY_ID = {
    "01": "全P1",
    "02": "低1→P2",
    "03": "低1-2→P2",
    "04": "低1→P2",
    "05": "低1-2→P2",
    "06": "低1-3→P2",
    "07": "低1→P3/低2→P2",
    "08": "低1-2→P3/低3→P2",
    "09": "低1-2+中1→P3/低3→P2",
    "10": "低1-2+中2→P3/低3→P2",
}
SCENARIO_JA_BY_NAME = {
    "pattern 01": "全P1",
    "pattern 02": "低1→P2",
    "pattern 03": "低1-2→P2",
    "pattern 04": "低1→P2",
    "pattern 05": "低1-2→P2",
    "pattern 06": "低1-3→P2",
    "pattern 07": "低1→P3/低2→P2",
    "pattern 08": "低1-2→P3/低3→P2",
    "pattern 09": "低1-2+中1→P3/低3→P2",
    "pattern 10": "低1-2+中2→P3/低3→P2",
    "unique 01": "基準票",
    "unique 02": "ユニーク列挙",
    "unique 03": "ユニーク列挙",
    "unique 04": "ユニーク列挙",
    "unique 05": "ユニーク列挙",
    "unique 06": "ユニーク列挙",
    "unique 07": "ユニーク列挙",
    "unique 08": "ユニーク列挙",
    "unique 09": "ユニーク列挙",
    "unique 10": "ユニーク列挙",
    "base": "基準",
    "slight upset": "軽い波乱",
    "strong upset": "強い波乱",
    "draw bias": "引分寄せ",
    "home bias": "ホーム寄せ",
    "away bias": "アウェイ寄せ",
    "tight match": "拮抗波乱",
    "chaos": "カオス",
    "conservative": "保守",
    "balanced": "均衡",
}


def parse_args():
    p = argparse.ArgumentParser(description="採点済み buyplan HTML 生成")
    p.add_argument("--round", required=True, help="round02 / toto1608")
    p.add_argument("--buyplan", default=None, help="既定: data/eval/{rounds|toto_rounds}/{round}/snapshot/buyplan.csv")
    p.add_argument("--actual", default=None, help="既定: data/eval/{rounds|toto_rounds}/{round}/actual_results.csv")
    p.add_argument("--evaluation", default=None, help="既定: data/eval/{rounds|toto_rounds}/{round}/evaluation.csv")
    p.add_argument("--out", default=None, help="既定: data/eval/{rounds|toto_rounds}/{round}/buyplan_scored.html")
    return p.parse_args()


def detect_candidate_columns(df):
    out = []
    for c in df.columns:
        m = re.search(r"(?:ticket|候補)\s*0*(10|[1-9])", str(c), flags=re.IGNORECASE)
        if m:
            out.append((int(m.group(1)), c))
    out = sorted(set(out), key=lambda x: x[0])
    if not out:
        raise RuntimeError("候補列が検出できません")
    return out


def build_scenario_meta(df_buy, cand_cols):
    if df_buy.empty:
        return ""
    first = df_buy.iloc[0]
    parts = []
    for idx, _ in cand_cols:
        sid_col = f"scenario_id_{idx:02d}"
        sname_col = f"scenario_name_{idx:02d}"
        sid = str(first.get(sid_col, "")).strip()
        sname = str(first.get(sname_col, "")).strip()
        if (not sid or sid.lower() == "nan") and "scenario_id" in df_buy.columns:
            sid = str(first.get("scenario_id", "")).strip()
        if (not sname or sname.lower() == "nan") and "scenario_name" in df_buy.columns:
            sname = str(first.get("scenario_name", "")).strip()
        if sname.lower() == "nan":
            sname = ""
        if sid.lower() == "nan":
            sid = ""
        sja = SCENARIO_JA_BY_NAME.get(sname.lower(), SCENARIO_JA_BY_ID.get(sid, "未定義"))
        if sname:
            parts.append(f"候補{idx:02d}:{sname}（{sja}）")
        else:
            parts.append(f"候補{idx:02d}:（{sja}）")
    return " / ".join(parts)


def build_scenario_ja_by_candidate(df_buy, cand_cols):
    out = {}
    if df_buy.empty:
        return out
    first = df_buy.iloc[0]
    for idx, _ in cand_cols:
        sid_col = f"scenario_id_{idx:02d}"
        sname_col = f"scenario_name_{idx:02d}"
        sid = str(first.get(sid_col, "")).strip()
        sname = str(first.get(sname_col, "")).strip()
        if (not sid or sid.lower() == "nan") and "scenario_id" in df_buy.columns:
            sid = str(first.get("scenario_id", "")).strip()
        if (not sname or sname.lower() == "nan") and "scenario_name" in df_buy.columns:
            sname = str(first.get("scenario_name", "")).strip()
        if sid.lower() == "nan":
            sid = ""
        if sname.lower() == "nan":
            sname = ""
        out[idx] = SCENARIO_JA_BY_NAME.get(sname.lower(), SCENARIO_JA_BY_ID.get(sid, "未定義"))
    return out


def detect_prob_columns(df):
    candidates = [
        ("p_home", "p_draw", "p_away"),
        ("prob_home_win", "prob_draw", "prob_away_win"),
        ("prob_home", "prob_draw", "prob_away"),
    ]
    for cols in candidates:
        if set(cols).issubset(df.columns):
            return cols
    return None


def best_second_label(row, prob_cols):
    if not prob_cols:
        return ""
    home_col, draw_col, away_col = prob_cols
    labels = [("H", row.get(home_col)), ("D", row.get(draw_col)), ("A", row.get(away_col))]
    pairs = []
    for label, value in labels:
        try:
            pairs.append((label, float(value)))
        except (TypeError, ValueError):
            return ""
    ordered = sorted(pairs, key=lambda item: item[1], reverse=True)
    return f"{ordered[0][0]}/{ordered[1][0]}"


def infer_draw_pressure_count(df, prob_cols):
    if df.empty or not prob_cols:
        return 0
    home_col, draw_col, away_col = prob_cols
    count = 0
    for _, row in df.iterrows():
        try:
            p_home = float(row.get(home_col))
            p_draw = float(row.get(draw_col))
            p_away = float(row.get(away_col))
        except (TypeError, ValueError):
            continue
        ordered = sorted([p_home, p_draw, p_away], reverse=True)
        if p_draw >= ordered[1] or p_draw >= 0.30:
            count += 1
    return count


def candidate_draw_priority_label(draw_count, draw_pressure_count):
    if draw_pressure_count >= 3 and draw_count == 0:
        return "低優先"
    if draw_pressure_count >= 4 and draw_count <= 1:
        return "注意"
    return "通常"


def evaluation_row_map(ev: pd.DataFrame) -> dict:
    if ev.empty or "candidate_id" not in ev.columns:
        return {}
    out = {}
    for _, row in ev.iterrows():
        out[str(row["candidate_id"])] = row
    return out


def display_result_mark(value):
    if pd.isna(value):
        return ""
    try:
        fv = float(value)
    except (TypeError, ValueError):
        s = str(value).strip()
        return "" if s.lower() == "nan" else s
    if fv in {0.0, 1.0, 2.0}:
        return str(int(fv))
    return str(value).strip()


def is_resolved_result(value):
    return display_result_mark(value) in {"0", "1", "2"}


def main():
    args = parse_args()
    round_id = args.round
    base = eval_base_dir(round_id)
    buyplan_path = args.buyplan or os.path.join(base, "snapshot", "buyplan.csv")
    actual_path = args.actual or os.path.join(base, "actual_results.csv")
    eval_path = args.evaluation or os.path.join(base, "evaluation.csv")
    out_path = args.out or os.path.join(base, "buyplan_scored.html")

    buy = pd.read_csv(buyplan_path)
    actual = pd.read_csv(actual_path)
    ev = pd.read_csv(eval_path) if os.path.exists(eval_path) else pd.DataFrame()
    cand_cols = detect_candidate_columns(buy)
    scenario_meta = build_scenario_meta(buy, cand_cols)
    scenario_ja_by_candidate = build_scenario_ja_by_candidate(buy, cand_cols)
    prob_cols = detect_prob_columns(buy)

    merged = buy.merge(actual[["match_no", "result"]], on="match_no", how="left")
    merged = merged.sort_values("match_no").reset_index(drop=True)
    resolved_total = int(merged["result"].map(is_resolved_result).sum())
    unresolved_total = int(len(merged) - resolved_total)

    score_map = {}
    eval_row_map = evaluation_row_map(ev)
    avg_hit_rate_pct = None
    avg_hits = None
    avg_total = None
    avg_draw_precision_pct = None
    avg_draw_recall_pct = None
    if not ev.empty and {"candidate_id", "hits", "total"}.issubset(ev.columns):
        for _, r in ev.iterrows():
            score_map[str(r["candidate_id"])] = f"{int(r['hits'])}/{int(r['total'])}"
        hits = pd.to_numeric(ev["hits"], errors="coerce")
        total = pd.to_numeric(ev["total"], errors="coerce")
        valid = hits.notna() & total.notna()
        if valid.any():
            avg_hits = float(hits[valid].mean())
            avg_total = float(total[valid].mean())
    if not ev.empty:
        if "hit_rate" in ev.columns:
            hr = pd.to_numeric(ev["hit_rate"], errors="coerce").dropna()
            if len(hr) > 0:
                avg_hit_rate_pct = float(hr.mean() * 100.0)
        elif {"hits", "total"}.issubset(ev.columns):
            hits = pd.to_numeric(ev["hits"], errors="coerce")
            total = pd.to_numeric(ev["total"], errors="coerce")
            valid = (total > 0) & hits.notna() & total.notna()
            if valid.any():
                avg_hit_rate_pct = float((hits[valid] / total[valid]).mean() * 100.0)
        if {"draw_precision", "draw_recall"}.issubset(ev.columns):
            dp = pd.to_numeric(ev["draw_precision"], errors="coerce").dropna()
            dr = pd.to_numeric(ev["draw_recall"], errors="coerce").dropna()
            if len(dp) > 0:
                avg_draw_precision_pct = float(dp.mean() * 100.0)
            if len(dr) > 0:
                avg_draw_recall_pct = float(dr.mean() * 100.0)

    parts = []
    parts.append("<!doctype html><html lang='ja'><head><meta charset='utf-8'>")
    parts.append("<title>BuyPlan Scored</title>")
    parts.append(
        "<style>"
        "body{font-family:system-ui,-apple-system,sans-serif;margin:20px;}"
        "table{border-collapse:collapse;width:100%;font-size:12px;}"
        "th,td{border:1px solid #ddd;padding:6px;text-align:center;}"
        "th{background:#f5f5f5;} .left{text-align:left;} .ok{background:#e8f7ea;} .ng{background:#fff3f3;} .pending{background:#f7f7f7;color:#666;}"
        ".meta{margin-bottom:10px;color:#333;} .sub{font-size:12px;color:#555;}"
        "</style></head><body>"
    )
    parts.append(f"<h2>BuyPlan Scored {html.escape(round_id)}</h2>")
    parts.append(f"<div class='meta'>生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>")
    parts.append(f"<div class='sub'>scenario一覧: {html.escape(scenario_meta)}</div>")
    draw_count_map = {}
    zero_draw_candidates = []
    draw_pressure_count = infer_draw_pressure_count(merged, prob_cols)
    priority_label_map = {}
    for idx, col in cand_cols:
        d_count = int((merged[col].astype(str) == "0").sum()) if col in merged.columns else 0
        draw_count_map[idx] = d_count
        priority_label_map[idx] = candidate_draw_priority_label(d_count, draw_pressure_count)
        if d_count == 0:
            zero_draw_candidates.append(idx)
    if avg_hit_rate_pct is not None:
        parts.append(f"<div class='sub'>全候補平均的中率: <b>{avg_hit_rate_pct:.2f}%</b></div>")
    if avg_hits is not None:
        if avg_total is not None:
            parts.append(f"<div class='sub'>全候補平均的中数: <b>{avg_hits:.2f}/{avg_total:.0f}</b></div>")
        else:
            parts.append(f"<div class='sub'>全候補平均的中数: <b>{avg_hits:.2f}</b></div>")
    if avg_draw_precision_pct is not None:
        parts.append(f"<div class='sub'>全候補平均D的中率(draw precision): <b>{avg_draw_precision_pct:.2f}%</b></div>")
    if avg_draw_recall_pct is not None:
        parts.append(f"<div class='sub'>全候補平均D捕捉率(draw recall): <b>{avg_draw_recall_pct:.2f}%</b></div>")
    parts.append(
        f"<div class='sub'>buyplan={html.escape(buyplan_path)} / actual={html.escape(actual_path)} / eval={html.escape(eval_path)}</div>"
    )
    parts.append(f"<div class='sub'>結果反映: <b>{resolved_total}/{len(merged)}</b> 試合 / 未反映: <b>{unresolved_total}</b> 試合</div>")
    draw_count_text = " / ".join(f"候補{idx:02d}=D{draw_count_map.get(idx, 0)}本" for idx, _ in cand_cols)
    parts.append(f"<div class='sub'>候補別D本数: <b>{html.escape(draw_count_text)}</b></div>")
    priority_text = " / ".join(
        f"候補{idx:02d}={priority_label_map.get(idx, '通常')}" for idx, _ in cand_cols
    )
    parts.append(f"<div class='sub'>節内D気配: <b>{draw_pressure_count}試合</b> / 候補優先度: <b>{html.escape(priority_text)}</b></div>")
    if zero_draw_candidates:
        zero_text = " / ".join(f"候補{idx:02d}" for idx in zero_draw_candidates)
        parts.append(
            f"<div class='sub' style='color:#b3261e;'><b>Dなし候補に注意:</b> {html.escape(zero_text)} は D=0本です。"
            " 引分が複数出る節では上限が下がりやすいです。</div>"
        )
    parts.append("<table><thead><tr>")
    parts.append("<th>match_no</th><th class='left'>league</th><th class='left'>home_team</th><th class='left'>away_team</th><th>best/second</th><th>結果</th>")
    for idx, _ in cand_cols:
        sja = scenario_ja_by_candidate.get(idx, "未定義")
        d_count = draw_count_map.get(idx, 0)
        priority = priority_label_map.get(idx, "通常")
        parts.append(f"<th>候補{idx:02d}<br><small>{html.escape(sja)} / D{d_count} / {html.escape(priority)}</small></th>")
    parts.append("</tr></thead><tbody>")

    for _, r in merged.iterrows():
        parts.append("<tr>")
        parts.append(f"<td>{int(r['match_no'])}</td>")
        parts.append(f"<td class='left'>{html.escape(str(r.get('league','')))}</td>")
        parts.append(f"<td class='left'>{html.escape(str(r.get('home_team','')))}</td>")
        parts.append(f"<td class='left'>{html.escape(str(r.get('away_team','')))}</td>")
        parts.append(f"<td>{html.escape(best_second_label(r, prob_cols))}</td>")
        actual_mark = display_result_mark(r.get("result", ""))
        parts.append(f"<td><b>{html.escape(actual_mark)}</b></td>")
        for _, col in cand_cols:
            pred = str(r.get(col, "")).strip()
            if not actual_mark:
                cls = "pending"
                mark = ""
            else:
                cls = "ok" if pred == actual_mark else "ng"
                mark = " ✅" if cls == "ok" else ""
            parts.append(f"<td class='{cls}'>{html.escape(pred)}{mark}</td>")
        parts.append("</tr>")
    parts.append("</tbody>")

    parts.append("<tfoot><tr>")
    parts.append("<td colspan='6' class='left'><b>候補別命中数</b></td>")
    for idx, _ in cand_cols:
        cid = f"cand{idx:02d}"
        txt = score_map.get(cid, "-")
        parts.append(f"<td><b>{html.escape(txt)}</b></td>")
    parts.append("</tr>")
    if eval_row_map and {"draw_pred_count", "draw_hit_count", "draw_precision", "draw_recall"}.issubset(ev.columns):
        parts.append("<tr>")
        parts.append("<td colspan='6' class='left'><b>候補別D成績</b><br><small>予想D本数 / D的中数 / precision / recall</small></td>")
        for idx, _ in cand_cols:
            cid = f"cand{idx:02d}"
            row = eval_row_map.get(cid)
            if row is None:
                txt = "-"
            else:
                draw_pred_count = int(pd.to_numeric(row.get("draw_pred_count"), errors="coerce") or 0)
                draw_hit_count = int(pd.to_numeric(row.get("draw_hit_count"), errors="coerce") or 0)
                draw_precision = float(pd.to_numeric(row.get("draw_precision"), errors="coerce") or 0.0)
                draw_recall = float(pd.to_numeric(row.get("draw_recall"), errors="coerce") or 0.0)
                txt = f"{draw_pred_count}/{draw_hit_count}<br><small>{draw_precision*100:.0f}% / {draw_recall*100:.0f}%</small>"
            parts.append(f"<td><b>{txt}</b></td>")
        parts.append("</tr>")
    parts.append("</tfoot>")
    parts.append("</table></body></html>")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    print(f"[OK] scored html saved: {out_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
