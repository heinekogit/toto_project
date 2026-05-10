#!/usr/bin/env python3
import argparse
import os
import re
import shutil
import sys
import unicodedata

import pandas as pd


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def normalize_team_text(v):
    if pd.isna(v):
        return ""
    s = unicodedata.normalize("NFKC", str(v)).strip()
    s = s.replace("　", " ")
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


def build_card_set(df):
    if df.empty or "home_team" not in df.columns or "away_team" not in df.columns:
        return set()
    work = df[["home_team", "away_team"]].copy()
    work["home_key"] = work["home_team"].map(normalize_team_text)
    work["away_key"] = work["away_team"].map(normalize_team_text)
    return {(r["home_key"], r["away_key"]) for _, r in work.iterrows()}


def parse_args():
    p = argparse.ArgumentParser(description="snapshot内 buyplan を predictions 基準へ自動整形")
    p.add_argument("--snapshot-dir", required=True)
    p.add_argument("--python", default=os.path.join(ROOT_DIR, "scripts", ".venv", "bin", "python"))
    return p.parse_args()


def backup_if_needed(src, dst):
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"[INFO] snapshot元buyplanを退避: {dst}")


def restore_if_needed(src, dst):
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"[INFO] snapshot元buyplanを復元: {dst}")


def main():
    args = parse_args()
    snapshot_dir = os.path.abspath(args.snapshot_dir)
    pred_path = os.path.join(snapshot_dir, "predictions.csv")
    buyplan_csv = os.path.join(snapshot_dir, "buyplan.csv")
    buyplan_html = os.path.join(snapshot_dir, "buyplan.html")
    source_csv = os.path.join(snapshot_dir, "buyplan.source.csv")
    source_html = os.path.join(snapshot_dir, "buyplan.source.html")

    if not os.path.exists(pred_path):
        raise FileNotFoundError(f"predictions.csv not found: {pred_path}")
    if not os.path.exists(buyplan_csv):
        raise FileNotFoundError(f"buyplan.csv not found: {buyplan_csv}")

    backup_if_needed(buyplan_csv, source_csv)
    backup_if_needed(buyplan_html, source_html)
    restore_if_needed(source_csv, buyplan_csv)
    restore_if_needed(source_html, buyplan_html)

    pred_df = pd.read_csv(pred_path)
    buy_df = pd.read_csv(buyplan_csv)
    pred_cards = build_card_set(pred_df)
    buy_cards = build_card_set(buy_df)

    if pred_cards and pred_cards == buy_cards and len(pred_df) == len(buy_df):
        print("[OK] snapshot buyplan は predictions と整合しています")
        return

    print(
        "WARN: snapshot buyplan と predictions のカード集合が不一致です。"
        f" pred_only={len(pred_cards - buy_cards)} buy_only={len(buy_cards - pred_cards)}. "
        "評価対象は buyplan.csv を優先し、predictions.csv は節番号確認用として扱います。"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
