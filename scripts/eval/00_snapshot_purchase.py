#!/usr/bin/env python3
import argparse
import os
import shutil
import sys


def eval_base_dir(round_id: str) -> str:
    if str(round_id).startswith("toto"):
        return os.path.join("data", "eval", "toto_rounds", round_id)
    return os.path.join("data", "eval", "rounds", round_id)


def parse_args():
    p = argparse.ArgumentParser(description="購入時点スナップショット保存")
    p.add_argument("--round", required=True, help="round02 / toto1608 のような評価ID")
    p.add_argument("--srcdir", default="data/purchase_reference")
    p.add_argument("--outdir", default=None, help="既定: data/eval/{rounds|toto_rounds}/{round}/snapshot")
    return p.parse_args()


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

    print(f"[OK] snapshot saved: {outdir}")
    for p in copied:
        print(p)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
