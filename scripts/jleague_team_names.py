"""Small, side-effect-free team-name normalizer for supplemental tooling."""

from __future__ import annotations

import re
import unicodedata


def _key(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return re.sub(r"[\s・･\.．]", "", text).upper()


_ALIASES = {
    # J1
    "浦和レッズ": "浦和", "横浜F・マリノス": "横浜FM", "横浜ＦＭ": "横浜FM",
    "FC東京": "FC東京", "ＦＣ東京": "FC東京", "東京ヴェルディ": "東京V",
    "名古屋グランパス": "名古屋", "ガンバ大阪": "G大阪", "Ｇ大阪": "G大阪",
    "鹿島アントラーズ": "鹿島", "サンフレッチェ広島": "広島", "ヴィッセル神戸": "神戸",
    "川崎フロンターレ": "川崎F", "川崎Ｆ": "川崎F", "セレッソ大阪": "C大阪",
    "Ｃ大阪": "C大阪", "アビスパ福岡": "福岡", "京都サンガ": "京都",
    "京都サンガF.C.": "京都", "清水エスパルス": "清水", "V・ファーレン長崎": "長崎",
    "ジェフ千葉": "千葉", "柏レイソル": "柏", "水戸ホーリーホック": "水戸",
    "ファジアーノ岡山": "岡山", "FC町田ゼルビア": "町田", "ＦＣ町田ゼルビア": "町田",
    # J2 / cup participants
    "北海道コンサドーレ札幌": "札幌", "ヴァンラーレ八戸": "八戸", "ベガルタ仙台": "仙台",
    "ブラウブリッツ秋田": "秋田", "モンテディオ山形": "山形", "いわきFC": "いわき",
    "栃木シティ": "栃木C", "栃木Ｃ": "栃木C", "RB大宮アルディージャ": "大宮",
    "横浜FC": "横浜FC", "湘南ベルマーレ": "湘南", "ヴァンフォーレ甲府": "甲府",
    "アルビレックス新潟": "新潟", "カターレ富山": "富山", "ジュビロ磐田": "磐田",
    "藤枝MYFC": "藤枝", "徳島ヴォルティス": "徳島", "FC今治": "今治",
    "サガン鳥栖": "鳥栖", "大分トリニータ": "大分", "テゲバジャーロ宮崎": "宮崎",
    "愛媛FC": "愛媛", "愛媛ＦＣ": "愛媛", "ガイナーレ鳥取": "鳥取",
}

_LOOKUP = {_key(k): v for k, v in _ALIASES.items()}


def canonical_team_name(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value or "")).strip()
    return _LOOKUP.get(_key(text), text)

