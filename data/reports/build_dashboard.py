import csv
import os
import re
from datetime import datetime
from html import escape


REPORT_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.join(REPORT_DIR, "..", ".."))
HTML_DIR = os.path.join(REPORT_DIR, "html")


def _extract_round_num(label):
    m = re.search(r"第([0-9０-９]+)節", label)
    return int(m.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))) if m else 9999


def _extract_battle_num(label):
    m = re.search(r"第([0-9０-９]+)戦", label)
    return int(m.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))) if m else 9999


def _extract_day_num(label):
    m = re.search(r"第([0-9０-９]+)日", label)
    return int(m.group(1).translate(str.maketrans("０１２３４５６７８９", "0123456789"))) if m else 9999


def _season_label(year):
    if year == "2026_2027":
        return "2026/27"
    if year == "2026_special":
        return "2026 特殊期"
    return str(year).replace("_", "/")


def _prediction_display(opt, include_season=True):
    league, year, round_label, _, _ = opt
    parts = []
    if include_season:
        parts.append(_season_label(year))
    if league != "ALL":
        parts.append(league)
    parts.append(round_label)
    return " ".join(parts)


def _round_order(opt):
    return (
        _extract_round_num(opt[2]),
        _extract_battle_num(opt[2]),
        _extract_day_num(opt[2]),
        opt[2],
    )


def _next_round_numbers(operation_season):
    """Return the nearest not-yet-played round for each operational league."""
    today = datetime.now().date()
    rounds = set()
    for league in ("j1", "j2"):
        path = os.path.join(ROOT_DIR, f"{league}_{operation_season}_predictions.csv")
        if not os.path.isfile(path):
            continue
        candidates = []
        try:
            with open(path, encoding="utf-8-sig", newline="") as handle:
                for row in csv.DictReader(handle):
                    raw_date = str(row.get("datetime", "")).strip()[:10]
                    try:
                        match_date = datetime.strptime(raw_date, "%Y-%m-%d").date()
                    except ValueError:
                        continue
                    round_no = _extract_round_num(str(row.get("節", "")))
                    if match_date >= today and round_no != 9999:
                        candidates.append((match_date, round_no))
        except OSError:
            continue
        if candidates:
            rounds.add(min(candidates)[1])
    return rounds


def _parse_round_file_meta(filename, prefix):
    body = filename.replace(prefix, "").replace(".html", "")
    m = re.match(r"^(j[123]|all)_(\d{4}(?:_special|_\d{4})?)_(.+)$", body)
    if m:
        raw_league = m.group(1).lower()
        league = "ALL" if raw_league == "all" else raw_league.upper()
        year = m.group(2)
        round_label = m.group(3)
    else:
        league = "NA"
        year = "NA"
        round_label = body
    return league, year, round_label


def _build_round_options(files, prefix):
    options = []
    for f in files:
        league, year, round_label = _parse_round_file_meta(f, prefix)
        display = f"{league} {year} {round_label}"
        options.append((league, year, round_label, display, f))
    # まとめページがある場合、日別ページ(第◯節第◯日 / 第◯戦第◯日)は候補から除外
    aggregate_keys = set()
    for league, year, round_label, _, _ in options:
        m = re.match(r"^(第\d+[節戦])$", round_label)
        if m:
            aggregate_keys.add((league, year, m.group(1)))
    filtered = []
    for item in options:
        league, year, round_label, _, _ = item
        m = re.match(r"^(第\d+[節戦])第\d+日$", round_label)
        if m and (league, year, m.group(1)) in aggregate_keys:
            continue
        filtered.append(item)
    options = filtered
    # リーグ年付き新形式がある場合は旧形式(NA)を非表示にする
    if any(opt[0] != "NA" and opt[1] != "NA" for opt in options):
        options = [opt for opt in options if not (opt[0] == "NA" and opt[1] == "NA")]
    options.sort(
        key=lambda x: (
            x[0],
            x[1],
            _extract_round_num(x[2]),
            _extract_battle_num(x[2]),
            _extract_day_num(x[2]),
            x[2],
            x[4],
        )
    )
    return options


def _collapse_pred_options_by_round(options):
    def league_priority(league):
        if league == "ALL":
            return 0
        if league == "J1":
            return 1
        if league == "J2":
            return 2
        if league == "J3":
            return 3
        return 9

    grouped = {}
    for opt in options:
        league, year, round_label, _, file_name = opt
        key = (year, round_label)
        rank_key = (
            league_priority(league),
            -(int(year) if str(year).isdigit() else 0),
            file_name,
        )
        if key not in grouped or rank_key < grouped[key][0]:
            grouped[key] = (rank_key, opt)

    collapsed = []
    for (_, round_label), (_, opt) in grouped.items():
        league, year, _, _, file_name = opt
        collapsed.append((league, year, round_label, f"{year} {round_label}", file_name))
    collapsed.sort(key=lambda x: (_extract_round_num(x[2]), _extract_battle_num(x[2]), _extract_day_num(x[2]), x[2]))
    return collapsed


def _latest_option(options):
    if not options:
        return None

    def _key(opt):
        league, year, round_label, _, _ = opt
        if str(year) == "2026_2027":
            year_num = 2027
        elif str(year) == "2026_special":
            year_num = 2026
        else:
            year_num = int(year) if str(year).isdigit() else -1
        league_rank = 1 if league == "ALL" else 0
        return (
            year_num,
            _extract_round_num(round_label),
            _extract_battle_num(round_label),
            _extract_day_num(round_label),
            league_rank,
        )

    return max(options, key=_key)


def _latest_option_prefer_all(options):
    if not options:
        return None
    all_options = [opt for opt in options if opt[0] == "ALL"]
    if all_options:
        return _latest_option(all_options)
    return _latest_option(options)


def _detect_operation_season():
    """Return the season used by the operational commands shown on the dashboard."""
    configured = os.environ.get("SEASON_YEAR", "").strip()
    if re.fullmatch(r"\d{4}", configured):
        return configured

    years = []
    try:
        root_names = os.listdir(ROOT_DIR)
    except OSError:
        root_names = []
    for name in root_names:
        match = re.match(r"^j[12]_(\d{4})_predictions(?:_.+)?\.csv$", name, flags=re.IGNORECASE)
        if match:
            years.append(int(match.group(1)))

    if os.path.isdir(HTML_DIR):
        for name in os.listdir(HTML_DIR):
            match = re.match(
                r"^(?:predictions|backtest)_round_(?:j[123]|all)_(\d{4})(?:_special|_\d{4})?_",
                name,
                flags=re.IGNORECASE,
            )
            if match:
                years.append(int(match.group(1)))

    # 新シーズン情報がまだ無い環境でも、従来の運用年で表示できるようにする。
    return str(max(years)) if years else str(datetime.now().year)


def build_dashboard():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    operation_season = _detect_operation_season()
    operation_leagues = os.environ.get("LEAGUES", "j1 j2").strip() or "j1 j2"
    if os.path.exists(HTML_DIR):
        all_pred_links = sorted([f for f in os.listdir(HTML_DIR) if f.startswith("predictions_round_") and f.endswith(".html")])
        all_back_links = sorted([f for f in os.listdir(HTML_DIR) if f.startswith("backtest_round_") and f.endswith(".html")])
        all_back_toto_links = sorted([f for f in os.listdir(HTML_DIR) if f.startswith("backtest_toto_") and f.endswith(".html")])
    else:
        all_pred_links = []
        all_back_links = []
        all_back_toto_links = []

    round_pred_links = _collapse_pred_options_by_round(_build_round_options(all_pred_links, "predictions_round_"))
    round_back_links = _build_round_options(all_back_links, "backtest_round_")
    round_back_toto_links = [(f, f) for f in all_back_toto_links]
    latest_pred_opt = _latest_option(round_pred_links)
    latest_back_opt = _latest_option_prefer_all(round_back_links)
    latest_pred = latest_pred_opt[4] if latest_pred_opt else None
    latest_back = latest_back_opt[4] if latest_back_opt else None
    latest_back_toto = all_back_toto_links[-1] if all_back_toto_links else None
    current_prediction_year = latest_pred_opt[1] if latest_pred_opt else None
    current_pred_links = [opt for opt in round_pred_links if opt[1] == current_prediction_year]
    archive_pred_links = [opt for opt in round_pred_links if opt[1] != current_prediction_year]
    current_pred_links.sort(key=_round_order, reverse=True)
    archive_pred_links.sort(key=lambda opt: (str(opt[1]), _round_order(opt)), reverse=True)
    current_season_label = _season_label(current_prediction_year) if current_prediction_year else "未検出"
    next_round_numbers = _next_round_numbers(operation_season)
    first_active_round = min(next_round_numbers) if next_round_numbers else None
    near_pred_links = [
        opt for opt in current_pred_links
        if first_active_round is not None and first_active_round <= _extract_round_num(opt[2]) <= first_active_round + 2
    ]
    future_pred_links = [
        opt for opt in current_pred_links
        if first_active_round is not None and _extract_round_num(opt[2]) > first_active_round + 2
    ]
    past_pred_links = [
        opt for opt in current_pred_links
        if first_active_round is None or _extract_round_num(opt[2]) < first_active_round
    ]
    near_pred_links.sort(key=_round_order)
    future_pred_links.sort(key=_round_order)
    past_pred_links.sort(key=_round_order, reverse=True)
    if near_pred_links:
        latest_pred = near_pred_links[0][4]
    html = []
    html.append("<!doctype html>")
    html.append("<html lang='ja'>")
    html.append("<head>")
    html.append("<meta charset='utf-8'>")
    html.append("<title>v2.0 予想・運用ダッシュボード</title>")
    html.append("<style>")
    html.append("body{font-family:system-ui,-apple-system,sans-serif;margin:0;background:#f3f6fa;color:#172033;}")
    html.append("main{max-width:1180px;margin:auto;padding:24px;}")
    html.append(".box{background:#fff;border:1px solid #d6dee8;border-radius:12px;padding:18px;margin-bottom:16px;box-shadow:0 2px 8px #1720330d;}")
    html.append(".hero{background:#172f4f;color:#fff;border:0}.hero .muted{color:#c9d8ea;}")
    html.append(".grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(260px,1fr));gap:14px;}")
    html.append(".row{margin:9px 0;}")
    html.append(".label{font-weight:600; margin-right:8px;}")
    html.append("a{color:#1769aa;text-decoration:none;font-weight:600}.hero a{color:#fff}")
    html.append("select{min-width:320px;max-width:100%;padding:9px;border:1px solid #9facbb;border-radius:7px;background:#fff}")
    html.append("button,.button{display:inline-block;padding:9px 13px;border:0;border-radius:7px;background:#1769aa;color:#fff;cursor:pointer;margin-left:5px}")
    html.append("code{white-space:normal}.muted{color:#617184}.warning{border-left:5px solid #d99000;background:#fff8e8;padding:12px}")
    html.append("details{margin-top:10px}summary{cursor:pointer;font-weight:700;padding:5px 0}")
    html.append("@media(max-width:640px){main{padding:12px}select{min-width:0;width:100%;margin-bottom:8px}button{margin-left:0;margin-right:4px}}")
    html.append("</style>")
    html.append("</head>")
    html.append("<body><main>")
    html.append("<div class='box hero'>")
    html.append("<h1>v2.0 予想・運用ダッシュボード</h1>")
    html.append(f"<div class='row'>現行予想シーズン: <b>{escape(current_season_label)}</b></div>")
    html.append(f"<div class='row muted'>生成: {now} / 予想本体とBuyPlanの確認、前向き検証の入口</div>")
    html.append("</div>")

    html.append("<div class='grid'>")
    html.append("<section class='box'><h2>現在の予想</h2>")
    if latest_pred:
        html.append(f"<div class='row'><a class='button' href='html/{escape(latest_pred)}'>直近の予想を開く</a></div>")
    else:
        html.append("<div class='row'>最新予想なし</div>")
    html.append("<div class='row'><a href='html/predictions_view.html'>予想一覧レポート</a></div>")
    html.append("<div class='row'><a href='html/report_view.html'>解析内訳レポート</a></div>")
    html.append("<button onclick='reloadNoCache()'>再読込</button></section>")
    html.append("<section class='box'><h2>BuyPlan v2.0</h2>")
    html.append("<div class='row'>正式版の購入候補とD分類を確認します。</div>")
    html.append("<div class='row'><a class='button' href='../purchase_reference/buyplan.html'>buyplan.htmlを開く</a></div>")
    html.append("<div class='row muted'>対象toto13試合が一致しない場合は生成時に安全停止します。</div></section>")
    html.append("</div>")

    html.append("<section class='box'>")
    html.append(f"<h2>現行シーズンの節別予想 <small class='muted'>{escape(current_season_label)}</small></h2>")
    html.append("<div class='row'>")
    if current_pred_links:
        html.append("<select id='predRound'><option value=''>予想節を選択</option>")
        for group_label, options in (("直近・次の3節", near_pred_links), ("今後の日程", future_pred_links), ("終了した節", past_pred_links)):
            if not options:
                continue
            html.append(f"<optgroup label='{group_label}'>")
            for opt in options:
                round_no = _extract_round_num(opt[2])
                marker = " ◀ 次回対象" if round_no in next_round_numbers else ""
                label = _prediction_display(opt, include_season=False) + marker
                html.append(f"<option value='html/{escape(opt[4])}'>{escape(label)}</option>")
            html.append("</optgroup>")
        html.append("</select><button onclick=\"openRound('predRound')\">表示</button>")
        html.append("<button onclick=\"downloadPredRoundCsv()\">buyplan用CSV出力</button>")
    else:
        html.append("（未生成）")
    html.append("</div><div class='row muted'>新しい節から表示します。CSV出力は選択節だけを含みます。</div></section>")

    html.append("<section class='box'><h2>v2.0の前向き検証</h2>")
    html.append("<div class='warning'>v1の過去バックテストとv2.0は予測方式が異なるため、直接比較しません。v2.0は正式運用後の固定保存データで評価します。</div>")
    html.append("<div class='row muted'>記録対象: 候補01、10候補平均、最高的中、portfolio cover、H/D/A確率品質、D分類、入力欠損。</div></section>")

    html.append("<section class='box'><h2>過去資料・v1開発履歴</h2>")
    html.append("<div class='warning'>以下は開発経過と原因調査用です。現行v2.0の通常運用には使用しません。</div>")
    html.append("<details><summary>過去シーズンの予想を開く</summary><div class='row'>")
    if archive_pred_links:
        html.append("<select id='archivePredRound'><option value=''>過去の予想を選択</option>")
        for opt in archive_pred_links:
            html.append(f"<option value='html/{escape(opt[4])}'>{escape(_prediction_display(opt))}</option>")
        html.append("</select><button onclick=\"openRound('archivePredRound')\">表示</button>")
    else:
        html.append("（過去予想なし）")
    html.append("</div></details>")
    html.append("<details><summary>v1バックテストを開く</summary>")
    html.append("<div class='row'><a href='html/backtest_view.html'>v1バックテスト一覧</a></div>")
    if round_back_links:
        html.append("<div class='row'><select id='backRound'><option value=''>節別v1バックテスト</option>")
        for _, _, _, display, f in round_back_links:
            html.append(f"<option value='html/{escape(f)}'>{escape(display)}</option>")
        html.append("</select><button onclick=\"openRound('backRound')\">表示</button></div>")
    if round_back_toto_links:
        html.append("<div class='row'><select id='backTotoRound'><option value=''>toto回v1バックテスト</option>")
        for f, display in round_back_toto_links:
            html.append(f"<option value='html/{escape(f)}'>{escape(display)}</option>")
        html.append("</select><button onclick=\"openRound('backTotoRound')\">表示</button></div>")
    html.append("</details></section>")

    html.append("<details class='box'><summary>保守・再生成コマンド</summary>")
    html.append("<div class='row'><code>python data/reports/view_predictions.py</code></div>")
    html.append("<div class='row'><code>python data/reports/view_report_json.py</code></div>")
    html.append("<div class='row'><code>python data/reports/build_round_views.py</code></div>")
    html.append(f"<div class='row'><code>SEASON_YEAR={operation_season} LEAGUES='{escape(operation_leagues)}' ./scripts/run_batch_weekly.sh</code></div>")
    html.append(f"<div class='row'><code>SEASON_YEAR={operation_season} LEAGUES='{escape(operation_leagues)}' ./scripts/run_batch_matchday.sh</code></div>")
    html.append("</details>")

    html.append("<script>")
    html.append("function openRound(selectId){")
    html.append("  const sel = document.getElementById(selectId);")
    html.append("  if (!sel || !sel.value) { alert('節を選択してください'); return; }")
    html.append("  window.location.href = sel.value;")
    html.append("}")
    html.append("function downloadPredRoundCsv(){")
    html.append("  const sel = document.getElementById('predRound');")
    html.append("  if (!sel || !sel.value) { alert('節を選択してください'); return; }")
    html.append("  const htmlPath = sel.value;")
    html.append("  const file = htmlPath.split('/').pop();")
    html.append("  const csvPath = 'csv/' + file.replace('.html', '.csv');")
    html.append("  const stem = file.replace('.html', '').replace(/^predictions_round_/, '');")
    html.append("  const a = document.createElement('a');")
    html.append("  a.href = csvPath;")
    html.append("  a.download = 'purchase_reference_predictions_' + stem + '.csv';")
    html.append("  document.body.appendChild(a);")
    html.append("  a.click();")
    html.append("  document.body.removeChild(a);")
    html.append("}")
    html.append("function reloadNoCache(){")
    html.append("  const u = new URL(window.location.href);")
    html.append("  u.searchParams.set('t', Date.now().toString());")
    html.append("  window.location.href = u.toString();")
    html.append("}")
    html.append("</script>")
    html.append("</main></body></html>")

    out_path = os.path.join(REPORT_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"HTML出力: {out_path}")


if __name__ == "__main__":
    build_dashboard()
