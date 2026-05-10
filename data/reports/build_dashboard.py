import os
import re
from datetime import datetime


REPORT_DIR = os.path.abspath(os.path.dirname(__file__))
HTML_DIR = os.path.join(REPORT_DIR, "html")


def _extract_round_num(label):
    m = re.search(r"第(\d+)節", label)
    return int(m.group(1)) if m else 9999


def _extract_battle_num(label):
    m = re.search(r"第(\d+)戦", label)
    return int(m.group(1)) if m else 9999


def _extract_day_num(label):
    m = re.search(r"第(\d+)日", label)
    return int(m.group(1)) if m else 9999


def _parse_round_file_meta(filename, prefix):
    body = filename.replace(prefix, "").replace(".html", "")
    m = re.match(r"^(j[123]|all)_(\d{4})_(.+)$", body)
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
        key = round_label
        rank_key = (
            league_priority(league),
            -(int(year) if str(year).isdigit() else 0),
            file_name,
        )
        if key not in grouped or rank_key < grouped[key][0]:
            grouped[key] = (rank_key, opt)

    collapsed = []
    for round_label, (_, opt) in grouped.items():
        league, year, _, _, file_name = opt
        collapsed.append((league, year, round_label, round_label, file_name))
    collapsed.sort(key=lambda x: (_extract_round_num(x[2]), _extract_battle_num(x[2]), _extract_day_num(x[2]), x[2]))
    return collapsed


def _latest_option(options):
    if not options:
        return None

    def _key(opt):
        league, year, round_label, _, _ = opt
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


def build_dashboard():
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
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
    html = []
    html.append("<!doctype html>")
    html.append("<html lang='ja'>")
    html.append("<head>")
    html.append("<meta charset='utf-8'>")
    html.append("<title>作業管理ページ</title>")
    html.append("<style>")
    html.append("body{font-family:system-ui, -apple-system, sans-serif; margin:24px;}")
    html.append(".box{border:2px solid #333; padding:16px; margin-bottom:16px;}")
    html.append(".row{margin:8px 0;}")
    html.append(".label{font-weight:600; margin-right:8px;}")
    html.append("a{color:#1a73e8; text-decoration:none;}")
    html.append("</style>")
    html.append("</head>")
    html.append("<body>")
    html.append("<h2>作業管理ページ</h2>")
    html.append(f"<div class='row'>更新: {now}</div>")

    html.append("<div class='box'>")
    html.append("<div class='row'><span class='label'>準備</span>週次バッチ起動: <code>/scripts/run_batch_weekly.sh</code></div>")
    html.append("<div class='row'><span class='label'>準備</span>試合直前バッチ: <code>/scripts/run_batch_matchday.sh</code></div>")
    html.append("<div class='row'><span class='label'>手動実行(ログ保存)</span><code>SEASON_YEAR=2026 LEAGUES='j1 j2' ./scripts/run_batch_weekly.sh 2>&1 | tee logs/run_batch_weekly.log</code></div>")
    html.append("<div class='row'><span class='label'>手動実行(ログ保存)</span><code>SEASON_YEAR=2026 LEAGUES='j1 j2' ./scripts/run_batch_matchday.sh 2>&1 | tee logs/run_batch_matchday.log</code></div>")
    html.append("<div class='row'><span class='label'>ログ確認</span><code>tail -n 80 logs/run_batch_weekly.log</code> / <code>tail -n 80 logs/run_batch_matchday.log</code></div>")
    html.append("<div class='row'><span class='label'>結果確認</span><code>tail -n 20 j1_2026_predictions.csv</code> / <code>tail -n 20 j2_2026_predictions.csv</code></div>")
    html.append("<div class='row'><span class='label'>品質確認</span><code>ls data/reports/merge_qc/j1_2026</code> / <code>ls data/reports/merge_qc/j2_2026</code></div>")
    html.append("</div>")

    html.append("<div class='box'>")
    html.append("<div class='row'><span class='label'>予想</span>入力: 節 / リーグ / 年</div>")
    html.append("</div>")

    html.append("<div class='box'>")
    html.append("<div class='row'><span class='label'>予想レポート</span><a href='html/predictions_view.html'>結果予想レポート</a></div>")
    html.append("<div class='row'><span class='label'>解析内訳</span><a href='html/report_view.html'>解析内訳等レポート</a></div>")
    html.append("<div class='row'><span class='label'>バックテスト</span><a href='html/backtest_view.html'>バックテスト</a></div>")
    html.append("<div class='row'><span class='label'>最新表示</span>")
    if latest_pred:
        html.append(f"<a href='html/{latest_pred}'>最新の予想を開く</a>")
    else:
        html.append("最新予想なし")
    html.append(" / ")
    if latest_back:
        html.append(f"<a href='html/{latest_back}'>最新のバックテストを開く</a>")
    else:
        html.append("最新バックテストなし")
    html.append(" / ")
    if latest_back_toto:
        html.append(f"<a href='html/{latest_back_toto}'>最新のtoto回バックテストを開く</a>")
    else:
        html.append("最新toto回バックテストなし")
    html.append(" <button onclick='reloadNoCache()'>再読込(キャッシュ回避)</button></div>")
    html.append("</div>")

    html.append("<div class='box'>")
    html.append("<div class='row'><span class='label'>節別リンク</span>以下から節別HTMLを選択</div>")

    html.append("<div class='row'><span class='label'>予想</span>")
    if round_pred_links:
        html.append("<select id='predRound'>")
        html.append("<option value=''>選択してください</option>")
        for _, _, _, display, f in round_pred_links:
            html.append(f"<option value='html/{f}'>{display}</option>")
        html.append("</select>")
        html.append("<button onclick=\"openRound('predRound')\">表示</button>")
        html.append("<button onclick=\"downloadPredRoundCsv()\">buyplan用CSV出力</button>")
    else:
        html.append("（未生成）")
    html.append("</div>")

    html.append("<div class='row'><span class='label'>バックテスト</span>")
    if round_back_links:
        html.append("<select id='backRound'>")
        html.append("<option value=''>選択してください</option>")
        for _, _, _, display, f in round_back_links:
            html.append(f"<option value='html/{f}'>{display}</option>")
        html.append("</select>")
        html.append("<button onclick=\"openRound('backRound')\">表示</button>")
    else:
        html.append("（未生成）")
    html.append("</div>")

    html.append("<div class='row'><span class='label'>toto回BT</span>")
    if round_back_toto_links:
        html.append("<select id='backTotoRound'>")
        html.append("<option value=''>選択してください</option>")
        for f, display in round_back_toto_links:
            html.append(f"<option value='html/{f}'>{display}</option>")
        html.append("</select>")
        html.append("<button onclick=\"openRound('backTotoRound')\">表示</button>")
    else:
        html.append("（未生成）")
    html.append("</div>")
    html.append("</div>")

    html.append("<div class='box'>")
    html.append("<div class='row'><span class='label'>メモ</span>予想CSV出力は選択した節だけを含む buyplan 用CSV</div>")
    html.append("<div class='row'><span class='label'>メモ</span>各レポートHTMLは下記スクリプトで更新</div>")
    html.append("<div class='row'><code>python data/reports/view_predictions.py</code></div>")
    html.append("<div class='row'><code>python data/reports/view_report_json.py</code></div>")
    html.append("<div class='row'><code>python data/reports/view_backtest.py</code></div>")
    html.append("<div class='row'><code>python data/reports/build_round_views.py</code></div>")
    html.append("</div>")

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
    html.append("</body></html>")

    out_path = os.path.join(REPORT_DIR, "index.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html))
    print(f"HTML出力: {out_path}")


if __name__ == "__main__":
    build_dashboard()
