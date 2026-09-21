#!/usr/bin/env python3
"""Freeze pre-match D-filter advice without changing prediction or BuyPlan inputs."""

from __future__ import annotations

import argparse
import hashlib
import html
import importlib.util
import json
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_PURCHASE_DIR = ROOT_DIR / "data" / "purchase_reference"
DEFAULT_OUT_ROOT = ROOT_DIR / "data" / "eval" / "d_filter_shadow"
FILTER_MODULE_PATH = Path(__file__).with_name("06_evaluate_d_filter_candidates.py")


def _load_filter_module():
    spec = importlib.util.spec_from_file_location("d_filter_candidates", FILTER_MODULE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load D filter module: {FILTER_MODULE_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FILTER = _load_filter_module()

POLICY_LABELS = {
    "false_draw_only": "False Drawのみ",
    "false_draw_plus_trap": "False Draw + Trap",
    "topology_guard": "Topology Guard",
}

ADVICE_LABELS = {
    "unanimous_side_change": "3案一致でサイド変更",
    "majority_side_cover": "多数案でサイド変更",
    "keep_draw": "3案一致でD維持",
    "split_keep_draw": "D維持・変更が分岐",
    "baseline_not_draw": "正式予想がD以外",
    "not_applicable_no_prediction": "対象外（正式予測なし）",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _escape(value: object) -> str:
    if pd.isna(value):
        return ""
    return html.escape(str(value), quote=True)


def _symbol_label(value: object) -> str:
    symbol = FILTER._symbol(value)
    return {0: "D", 1: "H", 2: "A"}.get(symbol, "-")


def _probability(value: object) -> str:
    number = pd.to_numeric(value, errors="coerce")
    return "-" if pd.isna(number) else f"{float(number) * 100:.1f}%"


def build_reference_html(reference: pd.DataFrame, *, round_id: str, created_at: datetime) -> str:
    """Build a human-readable, observation-only comparison report."""
    frame = reference.copy()
    if "match_no" in frame.columns:
        frame["_sort_match_no"] = pd.to_numeric(frame["match_no"], errors="coerce")
        frame = frame.sort_values("_sort_match_no", na_position="last").drop(columns="_sort_match_no")

    eligible = pd.to_numeric(
        frame.get("d_filter_eligible", pd.Series(1, index=frame.index)), errors="coerce"
    ).fillna(0).eq(1)
    eligible_count = int(eligible.sum())
    excluded_count = int((~eligible).sum())
    baseline_draws = int(frame.loc[eligible, "baseline_symbol"].map(FILTER._symbol).eq(0).sum())
    policy_counts = {
        policy: int(pd.to_numeric(frame[f"{policy}_changed"], errors="coerce").fillna(0).eq(1).sum())
        for policy in FILTER.POLICIES
    }
    advice_counts = frame["reference_advice"].fillna("").astype(str).value_counts().to_dict()

    parts = [
        "<!doctype html>",
        '<html lang="ja"><head><meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>D-filter参考比較 {_escape(round_id)}</title>",
        """<style>
        :root { color-scheme:light; --ink:#172033; --muted:#657086; --line:#dbe1ea;
          --panel:#f7f9fc; --amber:#b45309; --red:#b91c1c; }
        * { box-sizing:border-box; }
        body { margin:0; padding:24px; color:var(--ink); background:#eef2f7;
          font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans JP",sans-serif; }
        main { max-width:1500px; margin:auto; background:white; padding:24px; border-radius:14px;
          box-shadow:0 8px 28px rgba(15,23,42,.08); }
        h1 { margin:0 0 6px; font-size:26px; }
        .sub { color:var(--muted); font-size:13px; }
        .notice { margin:18px 0; padding:12px 14px; border-left:5px solid var(--amber);
          background:#fffbeb; line-height:1.55; }
        .cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin:16px 0; }
        .card { border:1px solid var(--line); background:var(--panel); border-radius:10px; padding:12px; }
        .card b { display:block; margin-top:4px; font-size:22px; }
        .legend { display:flex; gap:12px; flex-wrap:wrap; margin:12px 0; font-size:13px; color:var(--muted); }
        .table-wrap { overflow:auto; border:1px solid var(--line); border-radius:10px; }
        table { border-collapse:collapse; width:100%; min-width:1220px; font-size:13px; }
        th,td { border-bottom:1px solid var(--line); padding:8px 7px; text-align:center; white-space:nowrap; }
        th { position:sticky; top:0; z-index:1; background:#e9eef6; }
        td.left { text-align:left; }
        tr:hover td { background:#f8fafc; }
        .symbol { display:inline-flex; width:28px; height:28px; align-items:center; justify-content:center;
          border-radius:50%; font-weight:700; }
        .sym-H { background:#dbeafe; color:#1d4ed8; } .sym-D { background:#fef3c7; color:#92400e; }
        .sym-A { background:#dcfce7; color:#047857; } .sym-none { background:#e5e7eb; }
        .changed { outline:2px solid var(--red); outline-offset:1px; }
        .advice { font-weight:600; }
        footer { margin-top:14px; color:var(--muted); font-size:12px; }
        </style></head><body><main>""",
        f"<h1>D-filter参考比較 — {_escape(round_id)}</h1>",
        f'<div class="sub">生成日時: {_escape(created_at.isoformat(timespec="seconds"))}</div>',
        '<div class="notice"><b>観測専用レポートです。</b> 正式な predictions / buyplan / 10候補券は変更していません。'
        "過去の少数サンプルから作った参考ポリシーを、購入前の確認材料として並べています。"
        "正式予測が存在しない仮入力試合は、Dフィルター対象外として表示します。</div>",
        '<div class="cards">',
        f'<div class="card">対象試合<b>{len(frame)}</b></div>',
        f'<div class="card">D-filter評価対象<b>{eligible_count}</b></div>',
        f'<div class="card">対象外（正式予測なし）<b>{excluded_count}</b></div>',
        f'<div class="card">正式予想のD<b>{baseline_draws}</b></div>',
    ]
    for policy in FILTER.POLICIES:
        parts.append(
            f'<div class="card">{_escape(POLICY_LABELS[policy])} の変更<b>{policy_counts[policy]}</b></div>'
        )
    parts += [
        f'<div class="card">3案一致サイド変更<b>{int(advice_counts.get("unanimous_side_change", 0))}</b></div>',
        f'<div class="card">D判断が分岐<b>{int(advice_counts.get("split_keep_draw", 0))}</b></div>',
        "</div>",
        '<div class="legend"><span>H=ホーム勝ち</span><span>D=引き分け</span><span>A=アウェー勝ち</span>'
        '<span>赤枠=正式予想から変更する参考案</span></div>',
        '<div class="table-wrap"><table><thead><tr>',
        "<th>No.</th><th>リーグ</th><th>ホーム</th><th>アウェー</th>",
        "<th>H確率</th><th>D確率</th><th>A確率</th><th>正式</th>",
    ]
    for policy in FILTER.POLICIES:
        parts.append(f"<th>{_escape(POLICY_LABELS[policy])}</th>")
    parts += ["<th>参考判断</th><th>多数側</th><th>試合タイプ</th><th>D tier</th></tr></thead><tbody>"]

    for _, row in frame.iterrows():
        def symbol_cell(value: object, changed: bool = False) -> str:
            label = _symbol_label(value)
            classes = f"symbol sym-{label if label in {'H', 'D', 'A'} else 'none'}"
            if changed:
                classes += " changed"
            return f'<span class="{classes}">{label}</span>'

        parts += [
            "<tr>",
            f"<td>{_escape(row.get('match_no'))}</td>",
            f"<td>{_escape(row.get('league'))}</td>",
            f'<td class="left">{_escape(row.get("home_team"))}</td>',
            f'<td class="left">{_escape(row.get("away_team"))}</td>',
            f"<td>{_probability(row.get('p_home'))}</td>",
            f"<td>{_probability(row.get('p_draw'))}</td>",
            f"<td>{_probability(row.get('p_away'))}</td>",
            f"<td>{symbol_cell(row.get('baseline_symbol'))}</td>",
        ]
        for policy in FILTER.POLICIES:
            changed = bool(pd.to_numeric(row.get(f"{policy}_changed"), errors="coerce") == 1)
            parts.append(f"<td>{symbol_cell(row.get(f'{policy}_symbol'), changed)}</td>")
        advice = str(row.get("reference_advice") or "")
        majority = row.get("filter_majority_side")
        parts += [
            f'<td class="left advice">{_escape(ADVICE_LABELS.get(advice, advice))}</td>',
            f"<td>{symbol_cell(majority) if str(majority).strip() else '-'}</td>",
            f'<td class="left">{_escape(row.get("context_match_purchase_type", row.get("match_purchase_type", "")))}</td>',
            f"<td>{_escape(row.get('context_draw_purchase_tier', row.get('draw_purchase_tier', '')))}</td>",
            "</tr>",
        ]
    parts += [
        "</tbody></table></div>",
        '<footer>元データ: predictions_source.csv / buyplan_source.csv / d_filter_prematch.csv / '
        "buyplan_d_filter_reference.csv</footer>",
        "</main></body></html>",
    ]
    return "\n".join(parts)


def _first(row: pd.Series, names: list[str]):
    for name in names:
        if name not in row.index:
            continue
        value = row.get(name)
        if pd.notna(value) and str(value).strip() != "":
            return value
    return ""


def _normalize_prediction_row(row: pd.Series) -> pd.Series:
    out = row.copy()
    out["predicted_result"] = _first(
        row,
        ["predicted_result_main_symbol", "predicted_result_main", "predicted_result", "predicted_result_final"],
    )
    out["p_home"] = _first(row, ["prob_final_home", "prob_blend_home", "prob_home_win", "p_home"])
    out["p_draw"] = _first(row, ["prob_final_draw", "prob_blend_draw", "prob_draw", "p_draw"])
    out["p_away"] = _first(row, ["prob_final_away", "prob_blend_away", "prob_away_win", "p_away"])
    return out


def build_prematch(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for source_index, source in predictions.iterrows():
        row = _normalize_prediction_row(source)
        baseline = FILTER._symbol(row.get("predicted_result"))
        if baseline is None:
            raise ValueError(f"invalid predicted result at row={source_index}: {row.get('predicted_result')!r}")
        record: dict[str, object] = {
            "source_row": int(source_index),
            "match_no": source.get("match_no"),
            "league": source.get("league"),
            "round_label": _first(source, ["節", "section", "round"]),
            "match_id": source.get("match_id"),
            "datetime": source.get("datetime"),
            "home_team": source.get("home_team"),
            "away_team": source.get("away_team"),
            "p_home": row.get("p_home"),
            "p_draw": row.get("p_draw"),
            "p_away": row.get("p_away"),
            "baseline_symbol": baseline,
            "match_purchase_type": source.get("match_purchase_type"),
            "match_purchase_subtype": source.get("match_purchase_subtype"),
            "rank1_symbol": source.get("rank1_symbol"),
            "rank2_symbol": source.get("rank2_symbol"),
            "rank3_symbol": source.get("rank3_symbol"),
        }
        policy_symbols: list[int] = []
        for policy in FILTER.POLICIES:
            symbol, changed, reason = FILTER.apply_policy(row, policy)
            record[f"{policy}_symbol"] = symbol
            record[f"{policy}_changed"] = int(changed)
            record[f"{policy}_reason"] = reason
            policy_symbols.append(symbol)
        side_symbols = [symbol for symbol in policy_symbols if symbol in {1, 2}]
        unanimous = len(set(policy_symbols)) == 1
        side_counts = {side: side_symbols.count(side) for side in (1, 2)}
        majority_side = max(side_counts, key=side_counts.get) if max(side_counts.values(), default=0) >= 2 else ""
        if unanimous and policy_symbols[0] in {1, 2} and baseline == 0:
            advice = "unanimous_side_change"
        elif majority_side and baseline == 0:
            advice = "majority_side_cover"
        elif unanimous and policy_symbols[0] == 0:
            advice = "keep_draw"
        elif baseline == 0:
            advice = "split_keep_draw"
        else:
            advice = "baseline_not_draw"
        record["filter_unanimous"] = int(unanimous)
        record["filter_majority_side"] = majority_side
        record["reference_advice"] = advice
        rows.append(record)
    return pd.DataFrame(rows)


def _merge_reference(buyplan: pd.DataFrame, prematch: pd.DataFrame) -> pd.DataFrame:
    filter_cols = [
        "match_no",
        "home_team",
        "away_team",
        "baseline_symbol",
        *[f"{policy}_symbol" for policy in FILTER.POLICIES],
        *[f"{policy}_changed" for policy in FILTER.POLICIES],
        "filter_unanimous",
        "filter_majority_side",
        "reference_advice",
    ]
    left = buyplan.copy()
    right = prematch[filter_cols].copy()
    for frame in (left, right):
        frame["_match_no_key"] = pd.to_numeric(frame["match_no"], errors="coerce").astype("Int64")
        frame["_home_key"] = frame["home_team"].map(
            lambda value: unicodedata.normalize("NFKC", str(value or "")).replace("　", " ").strip()
        )
        frame["_away_key"] = frame["away_team"].map(
            lambda value: unicodedata.normalize("NFKC", str(value or "")).replace("　", " ").strip()
        )
    right = right.drop(columns=["match_no", "home_team", "away_team"])
    merge_keys = ["_home_key", "_away_key"]
    if right["_match_no_key"].notna().any():
        merge_keys.insert(0, "_match_no_key")
    else:
        right = right.drop(columns=["_match_no_key"])
    by_no = left.merge(right, on=merge_keys, how="left", validate="one_to_one")
    by_no = by_no.drop(columns=["_match_no_key", "_home_key", "_away_key"])
    missing = by_no["baseline_symbol"].isna()
    if missing.any():
        # A special/cup match may be present only as a provisional BuyPlan row and
        # have no official prediction.  It must remain visible in the report, but
        # must not be evaluated by D-filter policies using fabricated probabilities.
        text_columns = [
            "baseline_symbol",
            *[f"{policy}_symbol" for policy in FILTER.POLICIES],
            "filter_majority_side",
            "reference_advice",
        ]
        by_no[text_columns] = by_no[text_columns].astype("object")
        by_no.loc[missing, "baseline_symbol"] = ""
        for policy in FILTER.POLICIES:
            by_no.loc[missing, f"{policy}_symbol"] = ""
            by_no.loc[missing, f"{policy}_changed"] = 0
        by_no.loc[missing, "filter_unanimous"] = 0
        by_no.loc[missing, "filter_majority_side"] = ""
        by_no.loc[missing, "reference_advice"] = "not_applicable_no_prediction"
    by_no["d_filter_eligible"] = (~missing).astype(int)
    by_no["d_filter_exclusion_reason"] = ""
    by_no.loc[missing, "d_filter_exclusion_reason"] = "official_prediction_not_found"
    return by_no


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Freeze observation-only D-filter advice before kickoff")
    parser.add_argument("--round", required=True, dest="round_id", help="toto1648 / round04")
    parser.add_argument("--logical-season", default="2027")
    parser.add_argument("--predictions", default=str(DEFAULT_PURCHASE_DIR / "predictions.csv"))
    parser.add_argument("--buyplan", default=str(DEFAULT_PURCHASE_DIR / "buyplan.csv"))
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--label", default="prematch")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions).resolve()
    buyplan_path = Path(args.buyplan).resolve()
    missing = [str(path) for path in (predictions_path, buyplan_path) if not path.exists()]
    if missing:
        raise FileNotFoundError(f"required inputs are missing: {missing}")

    # Validate and build every derived table before creating the immutable folder.
    # A merge/input failure therefore cannot leave a partial snapshot behind.
    predictions = pd.read_csv(predictions_path, low_memory=False)
    buyplan = pd.read_csv(buyplan_path, low_memory=False)
    prematch = build_prematch(predictions)
    reference = _merge_reference(buyplan, prematch)

    created_at = datetime.now().astimezone()
    stamp = created_at.strftime("%Y%m%d_%H%M%S_%f")
    safe_label = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in args.label).strip("_") or "prematch"
    out_dir = Path(args.out_root).resolve() / str(args.logical_season) / args.round_id / f"{stamp}_{safe_label}"
    out_dir.mkdir(parents=True, exist_ok=False)
    source_predictions = out_dir / "predictions_source.csv"
    source_buyplan = out_dir / "buyplan_source.csv"
    shutil.copy2(predictions_path, source_predictions)
    shutil.copy2(buyplan_path, source_buyplan)
    prematch_path = out_dir / "d_filter_prematch.csv"
    reference_path = out_dir / "buyplan_d_filter_reference.csv"
    prematch.to_csv(prematch_path, index=False, encoding="utf-8-sig")
    reference.to_csv(reference_path, index=False, encoding="utf-8-sig")
    html_path = out_dir / "buyplan_d_filter_reference.html"
    html_path.write_text(
        build_reference_html(reference, round_id=args.round_id, created_at=created_at),
        encoding="utf-8",
    )

    manifest = {
        "schema_version": 1,
        "created_at": created_at.isoformat(timespec="seconds"),
        "round_id": args.round_id,
        "logical_season": str(args.logical_season),
        "label": safe_label,
        "prediction_rows": len(predictions),
        "buyplan_rows": len(buyplan),
        "policies": list(FILTER.POLICIES),
        "official_logic_changed": False,
        "files": {},
    }
    for path in (source_predictions, source_buyplan, prematch_path, reference_path, html_path):
        manifest["files"][path.name] = {"size": path.stat().st_size, "sha256": _sha256(path)}
    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] D-filter prematch shadow: {out_dir}")
    print(f"[OK] 参照HTML: {html_path}")
    print(f"[INFO] predictions={len(predictions)} buyplan={len(buyplan)} official_logic_changed=0")


if __name__ == "__main__":
    main()
