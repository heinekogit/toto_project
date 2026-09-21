#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime


def eval_base_dir(round_id: str) -> str:
    if str(round_id).startswith("toto"):
        return os.path.join("data", "eval", "toto_rounds", round_id)
    return os.path.join("data", "eval", "rounds", round_id)


def parse_args():
    p = argparse.ArgumentParser(description="購入時点スナップショット保存")
    p.add_argument("--round", required=True, help="round02 / toto1608 のような評価ID")
    p.add_argument("--srcdir", default="data/purchase_reference")
    p.add_argument("--outdir", default=None, help="既定: data/eval/{rounds|toto_rounds}/{round}/snapshot")
    p.add_argument("--logical-season", default="", help="toto節リスト上のシーズン。例: 2027")
    p.add_argument("--prediction-season", default="", help="予測ファイル上のシーズン。例: 2026")
    p.add_argument("--j1-source", default="", help="J1予測の元snapshot（記録専用）")
    p.add_argument("--j2-source", default="", help="J2予測の元snapshot（記録専用）")
    p.add_argument("--football-lab-snapshot", default="", help="Football LAB保存日。例: 20260802")
    return p.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    args = parse_args()
    round_id = args.round
    outdir = args.outdir or os.path.join(eval_base_dir(round_id), "snapshot")
    srcdir = args.srcdir

    required = ["predictions.csv", "buyplan.csv"]
    optional = ["buyplan.html", "predictions_buyplan_context.csv"]

    os.makedirs(os.path.dirname(outdir), exist_ok=True)
    if os.path.exists(outdir):
        raise RuntimeError(f"snapshot出力先が既に存在します（安全停止）: {outdir}")
    os.makedirs(outdir, exist_ok=False)

    copied = []
    for fn in required:
        src = os.path.join(srcdir, fn)
        dst = os.path.join(outdir, fn)
        if not os.path.exists(src):
            raise FileNotFoundError(f"必要ファイルが見つかりません: {src}")
        if os.path.exists(dst):
            raise RuntimeError(f"出力先に同名ファイルが既に存在: {dst}")
        shutil.copy2(src, dst)
        copied.append(dst)

    for fn in optional:
        src = os.path.join(srcdir, fn)
        dst = os.path.join(outdir, fn)
        if os.path.exists(src):
            if os.path.exists(dst):
                raise RuntimeError(f"出力先に同名ファイルが既に存在: {dst}")
            shutil.copy2(src, dst)
            copied.append(dst)

    manifest = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "round_id": round_id,
        "logical_season": str(args.logical_season).strip(),
        "prediction_season": str(args.prediction_season).strip(),
        "j1_source": os.path.abspath(args.j1_source) if args.j1_source else "",
        "j2_source": os.path.abspath(args.j2_source) if args.j2_source else "",
        "football_lab_snapshot": str(args.football_lab_snapshot).strip(),
        "files": {
            os.path.basename(path): {
                "size": os.path.getsize(path),
                "sha256": sha256_file(path),
            }
            for path in copied
        },
    }
    manifest_path = os.path.join(outdir, "manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
        f.write("\n")
    copied.append(manifest_path)

    print(f"[OK] snapshot saved: {outdir}")
    for p in copied:
        print(p)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
