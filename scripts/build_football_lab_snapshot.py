#!/usr/bin/env python3
import argparse
import csv
import json
import re
import unicodedata
from html import unescape
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
OUTPUT_DIR = BASE_DIR / "data" / "external_metrics"
RAW_DIR = OUTPUT_DIR / "raw"


TEAM_ALIAS_RAW_MAP = {
    "鹿島アントラーズ": "鹿島",
    "水戸ホーリーホック": "水戸",
    "浦和レッズ": "浦和",
    "ジェフユナイテッド千葉": "千葉",
    "柏レイソル": "柏",
    "ＦＣ東京": "FC東京",
    "FC東京": "FC東京",
    "G大阪": "Ｇ大阪",
    "C大阪": "Ｃ大阪",
    "川崎F": "川崎Ｆ",
    "東京V": "東京Ｖ",
    "東京ヴェルディ": "東京Ｖ",
    "ＦＣ町田ゼルビア": "町田",
    "FC町田ゼルビア": "町田",
    "川崎フロンターレ": "川崎Ｆ",
    "横浜Ｆ・マリノス": "横浜FM",
    "横浜F・マリノス": "横浜FM",
    "清水エスパルス": "清水",
    "名古屋グランパス": "名古屋",
    "京都サンガF.C.": "京都",
    "京都サンガFC": "京都",
    "ガンバ大阪": "Ｇ大阪",
    "セレッソ大阪": "Ｃ大阪",
    "ヴィッセル神戸": "神戸",
    "ファジアーノ岡山": "岡山",
    "サンフレッチェ広島": "広島",
    "アビスパ福岡": "福岡",
    "Ｖ・ファーレン長崎": "長崎",
    "V・ファーレン長崎": "長崎",
    "北海道コンサドーレ札幌": "札幌",
    "ベガルタ仙台": "仙台",
    "ブラウブリッツ秋田": "秋田",
    "モンテディオ山形": "山形",
    "ザスパ群馬": "群馬",
    "いわきＦＣ": "いわき",
    "いわきFC": "いわき",
    "ＲＢ大宮アルディージャ": "大宮",
    "RB大宮アルディージャ": "大宮",
    "横浜ＦＣ": "横浜FC",
    "湘南ベルマーレ": "湘南",
    "ヴァンラーレ八戸": "八戸",
    "ヴァンフォーレ甲府": "甲府",
    "アルビレックス新潟": "新潟",
    "栃木シティ": "栃木C",
    "栃木シティFC": "栃木C",
    "栃木シティＦＣ": "栃木C",
    "栃木SC": "栃木SC",
    "栃木ＳＣ": "栃木SC",
    "SC相模原": "相模原",
    "ＳＣ相模原": "相模原",
    "FC岐阜": "岐阜",
    "ＦＣ岐阜": "岐阜",
    "AC長野パルセイロ": "長野",
    "ＡＣ長野パルセイロ": "長野",
    "松本山雅FC": "松本",
    "松本山雅ＦＣ": "松本",
    "カターレ富山": "富山",
    "ツエーゲン金沢": "金沢",
    "ジュビロ磐田": "磐田",
    "藤枝ＭＹＦＣ": "藤枝",
    "藤枝MYFC": "藤枝",
    "徳島ヴォルティス": "徳島",
    "ＦＣ今治": "今治",
    "FC今治": "今治",
    "サガン鳥栖": "鳥栖",
    "大分トリニータ": "大分",
    "テゲバジャーロ宮崎": "宮崎",
    "レノファ山口ＦＣ": "山口",
    "レノファ山口FC": "山口",
    "ギラヴァンツ北九州": "北九州",
    "ガイナーレ鳥取": "鳥取",
    "カマタマーレ讃岐": "讃岐",
    "鹿児島ユナイテッドFC": "鹿児島",
    "鹿児島ユナイテッドＦＣ": "鹿児島",
    "福島ユナイテッドFC": "福島",
    "福島ユナイテッドＦＣ": "福島",
    "高知ユナイテッドSC": "高知",
    "高知ユナイテッドＳＣ": "高知",
    "コンサドーレ札幌": "札幌",
    "ロアッソ熊本": "熊本",
    "愛媛ＦＣ": "愛媛",
    "愛媛FC": "愛媛",
    "ＦＣ琉球": "琉球",
    "FC琉球": "琉球",
}

TEAM_ALIAS_MAP = {
    unicodedata.normalize("NFKC", str(k)).strip(): v
    for k, v in TEAM_ALIAS_RAW_MAP.items()
}

WIDE_TABLE_MAPPINGS = {
    "ゴール期待値": {
        "期待値": "flab_expected_for_xg",
        "ゴール": "flab_expected_for_goals",
        "差分": "flab_expected_for_diff",
        "成績順位": "flab_expected_for_rank",
    },
    "被ゴール期待値": {
        "期待値": "flab_expected_against_xg",
        "被ゴール": "flab_expected_against_goals",
        "差分": "flab_expected_against_diff",
        "成績順位": "flab_expected_against_rank",
    },
    "チャンス構築率": {
        "攻撃回数": "flab_chance_attack_count",
        "シュート": "flab_chance_shots",
        "チャンス構築率": "flab_chance_build_rate",
        "ゴール": "flab_chance_goals",
        "シュート成功率": "flab_chance_shot_conversion",
    },
    "被チャンス構築率": {
        "被攻撃回数": "flab_chance_allowed_attack_count",
        "被シュート": "flab_chance_allowed_shots",
        "被チャンス構築率": "flab_chance_allowed_build_rate",
        "被ゴール": "flab_chance_allowed_goals",
        "被シュート成功率": "flab_chance_allowed_shot_conversion",
    },
}

WIDE_CHART_MAPPINGS = {
    "ゴール期待値": {
        "ゴール": "flab_expected_for_goals",
        "期待値": "flab_expected_for_xg",
    },
    "被ゴール期待値": {
        "被ゴール": "flab_expected_against_goals",
        "期待値": "flab_expected_against_xg",
    },
    "チャンス構築率": {
        "シュート成功率": "flab_chance_shot_conversion",
        "チャンス構築率": "flab_chance_build_rate",
    },
    "被チャンス構築率": {
        "被シュート成功率": "flab_chance_allowed_shot_conversion",
        "被チャンス構築率": "flab_chance_allowed_build_rate",
    },
    "ボール保持率": {
        "攻撃CBP": "flab_possession_attack_cbp",
        "保持率": "flab_possession_rate",
    },
    "ボール保持率と攻撃CBP": {
        "攻撃CBP": "flab_possession_attack_cbp",
        "保持率": "flab_possession_rate",
    },
    "アクチュアルプレーイングタイム": {
        "1試合平均総走行距離": "flab_actual_play_distance",
        "アクチュアルプレーイングタイム": "flab_actual_play_time",
    },
}


def normalize_text(value):
    return unicodedata.normalize("NFKC", str(value or "")).strip()


def canonical_team_name(value):
    text = normalize_text(value)
    return TEAM_ALIAS_MAP.get(text, text)


def normalize_team_key(value):
    return canonical_team_name(value).upper().replace(" ", "")


def ensure_dirs():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def load_allowed_teams_for_league(league, season):
    paths = [
        BASE_DIR / "data" / "manual" / f"{league}_allowed_teams_{season}.csv",
        BASE_DIR / "data" / f"{league}_{season}_upcoming.csv",
        BASE_DIR / "data" / f"{league}_{season}_latest_results.csv",
    ]
    teams = set()
    for path in paths:
        if not path.exists():
            continue
        with path.open(encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "team_name" in row:
                    teams.add(normalize_team_key(row["team_name"]))
                    continue
                for col in ("home_team", "away_team"):
                    if row.get(col):
                        teams.add(normalize_team_key(row[col]))
        if teams:
            return teams
    return teams


def strip_tags(text):
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    text = normalize_text(text)
    return text


def clean_header(text):
    text = strip_tags(text)
    return re.sub(r"\s+", "", text)


def clean_metric_label(text):
    text = clean_header(text)
    text = text.replace("(%)", "").replace("（％）", "").replace("％", "").replace("%", "")
    text = text.replace("ボール保持率", "保持率")
    return text


def parse_title_info(title_text, fallback_league):
    season = ""
    league = fallback_league.lower()
    page_label = title_text
    m = re.search(r"(\d{4})\s+J[123](?:J[23])?百年構想リーグ\s+(.+?)\s*\|", title_text)
    if m:
        season = m.group(1)
        page_label = m.group(2)
    return season, league, page_label


def parse_asof_date(html_text, fallback_date):
    m = re.search(r'<em class="uiDate">(.*?)</em>', html_text, re.S)
    if not m:
        return fallback_date
    text = strip_tags(m.group(1))
    m2 = re.search(r"(\d{4})\.(\d{1,2})\.(\d{1,2})", text)
    if not m2:
        return fallback_date
    return f"{m2.group(1)}{int(m2.group(2)):02d}{int(m2.group(3)):02d}"


def find_preceding_section_title(matches, position):
    current = ""
    for match in matches:
        if match.start() > position:
            break
        current = strip_tags(match.group(1))
    return current


def extract_chart_section_titles(html_text):
    titles = []
    pattern = re.compile(
        r'<h3 class="boxHeader"><span>(.*?)</span></h3>\s*<div id="ccs\d+"',
        re.S,
    )
    for match in pattern.finditer(html_text):
        titles.append(strip_tags(match.group(1)))
    return titles


def parse_js_token(token):
    token = token.strip()
    if not token:
        return ""
    if token[0] in ("'", '"') and token[-1] == token[0]:
        token = token[1:-1]
        token = token.replace("\\'", "'").replace('\\"', '"')
        return normalize_text(token)
    try:
        if "." in token:
            return float(token)
        return int(token)
    except ValueError:
        return normalize_text(token)


def split_top_level(text, delimiter=","):
    parts = []
    buf = []
    level = 0
    quote = ""
    escaped = False
    for ch in text:
        if escaped:
            buf.append(ch)
            escaped = False
            continue
        if quote:
            buf.append(ch)
            if ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            continue
        if ch in ("'", '"'):
            buf.append(ch)
            quote = ch
            continue
        if ch == "[":
            level += 1
            buf.append(ch)
            continue
        if ch == "]":
            level -= 1
            buf.append(ch)
            continue
        if ch == delimiter and level == 0:
            parts.append("".join(buf).strip())
            buf = []
            continue
        buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


def parse_js_array_rows(body):
    rows = []
    level = 0
    start = None
    quote = ""
    escaped = False
    for i, ch in enumerate(body):
        if escaped:
            escaped = False
            continue
        if quote:
            if ch == "\\":
                escaped = True
            elif ch == quote:
                quote = ""
            continue
        if ch in ("'", '"'):
            quote = ch
            continue
        if ch == "[":
            level += 1
            if level == 1:
                start = i + 1
            continue
        if ch == "]":
            if level == 1 and start is not None:
                rows.append(body[start:i].strip())
                start = None
            level -= 1
    parsed = []
    for row in rows:
        parsed.append([parse_js_token(tok) for tok in split_top_level(row)])
    return parsed


def extract_chart_rows(html_text, meta):
    chart_titles = extract_chart_section_titles(html_text)
    chart_rows = []
    for idx, match in enumerate(re.finditer(r"google\.visualization\.arrayToDataTable\(\[(.*?)\]\);", html_text, re.S), 1):
        section_title = chart_titles[idx - 1] if idx - 1 < len(chart_titles) else meta["page_label"]
        rows = parse_js_array_rows(match.group(1))
        if len(rows) < 2:
            continue
        header = [normalize_text(v) for v in rows[0]]
        if len(header) < 4 or header[0] != "ID":
            continue
        metric_x = clean_metric_label(header[1])
        metric_y = clean_metric_label(header[2])
        for row in rows[1:]:
            if len(row) < 4:
                continue
            team_name = canonical_team_name(row[3])
            chart_rows.append({
                **meta,
                "section_title": section_title,
                "chart_index": idx,
                "metric_x_name": metric_x,
                "metric_x_value": row[1],
                "metric_y_name": metric_y,
                "metric_y_value": row[2],
                "team_short": normalize_text(row[0]),
                "team_name": team_name,
                "team_key": normalize_team_key(team_name),
            })
    return chart_rows


def extract_team_tables(html_text, meta):
    section_matches = list(re.finditer(r'<h3 class="boxHeader"><span>(.*?)</span></h3>', html_text, re.S))
    team_tables = []
    for idx, match in enumerate(re.finditer(r'<table[^>]*class="([^"]*statsTbl[^"]*)"[^>]*>(.*?)</table>', html_text, re.S), 1):
        if 'class="tName"' not in match.group(2):
            continue
        section_title = find_preceding_section_title(section_matches, match.start())
        headers = [clean_header(v) for v in re.findall(r"<th[^>]*>(.*?)</th>", match.group(2), re.S)]
        headers = [h for h in headers if h]
        tbody_match = re.search(r"<tbody>(.*?)</tbody>", match.group(2), re.S)
        if not tbody_match:
            continue
        trs = re.findall(r"<tr[^>]*>(.*?)</tr>", tbody_match.group(1), re.S)
        for tr in trs:
            tds = re.findall(r"<td[^>]*>(.*?)</td>", tr, re.S)
            if not tds:
                continue
            team_name = ""
            values = []
            for cell in tds:
                cell_text = strip_tags(cell)
                if 'class="tName"' in cell or "class='tName'" in cell:
                    team_name = canonical_team_name(cell_text)
                elif team_name:
                    values.append(cell_text)
            if not team_name:
                continue
            team_tables.append({
                **meta,
                "section_title": section_title,
                "table_index": idx,
                "team_name": team_name,
                "team_key": normalize_team_key(team_name),
                "headers_json": json.dumps(headers, ensure_ascii=False),
                "values_json": json.dumps(values, ensure_ascii=False),
            })
    return team_tables


def maybe_float(value):
    text = normalize_text(value)
    if not text:
        return ""
    text = text.replace("%", "").replace(",", "")
    try:
        return float(text)
    except ValueError:
        return text


def build_wide_metrics(table_rows, chart_rows):
    wide = {}
    for row in chart_rows:
        section_map = WIDE_CHART_MAPPINGS.get(row["section_title"])
        if not section_map:
            continue
        key = row["team_key"]
        entry = wide.setdefault(key, {
            "snapshot_date": row["snapshot_date"],
            "asof_date": row["asof_date"],
            "league": row["league"],
            "season": row["season"],
            "team_name": row["team_name"],
            "team_key": key,
        })
        if row["metric_x_name"] in section_map:
            entry[section_map[row["metric_x_name"]]] = maybe_float(row["metric_x_value"])
        if row["metric_y_name"] in section_map:
            entry[section_map[row["metric_y_name"]]] = maybe_float(row["metric_y_value"])

    for row in table_rows:
        section_map = WIDE_TABLE_MAPPINGS.get(row["section_title"])
        if not section_map:
            continue
        headers = json.loads(row["headers_json"])
        values = json.loads(row["values_json"])
        key = row["team_key"]
        entry = wide.setdefault(key, {
            "snapshot_date": row["snapshot_date"],
            "asof_date": row["asof_date"],
            "league": row["league"],
            "season": row["season"],
            "team_name": row["team_name"],
            "team_key": key,
        })
        mapped_headers = []
        for h in headers:
            cleaned = clean_header(h)
            if cleaned in section_map:
                mapped_headers.append(section_map[cleaned])
        for col, val in zip(mapped_headers, values):
            entry[col] = maybe_float(val)
    return list(wide.values())


def read_prediction_rows(prediction_csv):
    with prediction_csv.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def build_match_comparison(prediction_rows, team_metrics):
    metric_map = {row["team_key"]: row for row in team_metrics}
    output = []
    for row in prediction_rows:
        out = dict(row)
        home_key = normalize_team_key(row.get("home_team", ""))
        away_key = normalize_team_key(row.get("away_team", ""))
        home = metric_map.get(home_key, {})
        away = metric_map.get(away_key, {})
        metric_cols = sorted({
            k for item in (home, away) for k in item.keys()
            if k.startswith("flab_")
        })
        for col in metric_cols:
            hv = home.get(col, "")
            av = away.get(col, "")
            out[f"{col}_home"] = hv
            out[f"{col}_away"] = av
            if isinstance(hv, (int, float)) and isinstance(av, (int, float)):
                out[f"{col}_diff"] = hv - av
            else:
                out[f"{col}_diff"] = ""
        output.append(out)
    return output


def filter_team_metrics(team_metrics, league, season):
    allowed = load_allowed_teams_for_league(league, season)
    if not allowed:
        return team_metrics
    return [row for row in team_metrics if row["team_key"] in allowed]


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([])
        return
    fieldnames = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_league_dir(league_dir, prediction_csv=None):
    league_dir = league_dir.resolve()
    snapshot_date = league_dir.parent.name
    fallback_league = league_dir.name.lower()
    chart_rows = []
    table_rows = []
    for html_path in sorted(league_dir.glob("*.htm*")):
        html_text = html_path.read_text(encoding="utf-8", errors="ignore")
        title_match = re.search(r"<title>(.*?)</title>", html_text, re.S)
        title_text = strip_tags(title_match.group(1)) if title_match else html_path.stem
        season, league, page_label = parse_title_info(title_text, fallback_league)
        meta = {
            "snapshot_date": snapshot_date,
            "asof_date": parse_asof_date(html_text, snapshot_date),
            "league": league,
            "season": season,
            "source_file": html_path.name,
            "page_label": page_label,
            "title": title_text,
        }
        chart_rows.extend(extract_chart_rows(html_text, meta))
        table_rows.extend(extract_team_tables(html_text, meta))

    wide_metrics = build_wide_metrics(table_rows, chart_rows)
    league_for_filter = wide_metrics[0]["league"] if wide_metrics else fallback_league
    season_for_filter = wide_metrics[0]["season"] if wide_metrics else ""
    wide_metrics = filter_team_metrics(wide_metrics, league_for_filter, season_for_filter)

    chart_out = RAW_DIR / f"football_lab_chart_rows_{snapshot_date}_{fallback_league}.csv"
    table_out = RAW_DIR / f"football_lab_table_rows_{snapshot_date}_{fallback_league}.csv"
    team_out = OUTPUT_DIR / f"football_lab_team_metrics_{snapshot_date}_{fallback_league}.csv"
    write_csv(chart_out, chart_rows)
    write_csv(table_out, table_rows)
    write_csv(team_out, wide_metrics)

    compare_out = None
    if prediction_csv:
        prediction_rows = read_prediction_rows(prediction_csv)
        compare_rows = build_match_comparison(prediction_rows, wide_metrics)
        compare_out = OUTPUT_DIR / (
            f"{prediction_csv.stem}_football_lab_compare_{snapshot_date}_{fallback_league}.csv"
        )
        write_csv(compare_out, compare_rows)

    return {
        "chart_out": chart_out,
        "table_out": table_out,
        "team_out": team_out,
        "compare_out": compare_out,
        "chart_rows": len(chart_rows),
        "table_rows": len(table_rows),
        "team_rows": len(wide_metrics),
    }


def main():
    parser = argparse.ArgumentParser(description="Football LAB保存HTMLから外部比較データを生成する")
    parser.add_argument("--league-dir", required=True, help="external _contents/YYYYMMDD/j1 のような保存先")
    parser.add_argument("--prediction-csv", help="比較対象の予想CSV。指定時は試合単位比較CSVも出力")
    args = parser.parse_args()

    ensure_dirs()
    result = process_league_dir(
        Path(args.league_dir),
        Path(args.prediction_csv) if args.prediction_csv else None,
    )
    print(json.dumps({
        "chart_out": str(result["chart_out"]),
        "table_out": str(result["table_out"]),
        "team_out": str(result["team_out"]),
        "compare_out": str(result["compare_out"]) if result["compare_out"] else "",
        "chart_rows": result["chart_rows"],
        "table_rows": result["table_rows"],
        "team_rows": result["team_rows"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
