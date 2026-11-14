#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
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
    optional = [
        "buyplan.html", "predictions_buyplan_context.csv",
        "next_predictions.csv", "shadow_buyplan.csv", "shadow_buyplan.html",
        "shadow_buyplan_b.csv", "shadow_buyplan_b.html", "manifest.json",
        "shadow_buyplan_c.csv", "shadow_buyplan_c.html",
        "shadow_condition_s_buyplan.csv", "shadow_condition_s_buyplan.html",
        'shadow_buyplan_v25.csv', 'shadow_buyplan_v25.html', 'shadow_v25_features.csv', 'shadow_v25_candidates.csv', 'shadow_v25_audit.json', 'shadow_v25_reference.json',
    ]

    if os.path.exists(outdir):
        raise RuntimeError(f"snapshot出力先が既に存在します（安全停止）: {outdir}")
    for fn in required:
        if not os.path.isfile(os.path.join(srcdir, fn)):
            raise FileNotFoundError(f"必要ファイルが見つかりません: {fn}")
    engine_path = os.path.join(srcdir, "manifest.json")
    if os.path.isfile(os.path.join(srcdir, "next_predictions.csv")) and not os.path.isfile(engine_path):
        raise FileNotFoundError("v2生成manifest.jsonがありません。BuyPlanを再生成してください")
    if os.path.isfile(engine_path):
        with open(engine_path, encoding="utf-8") as f:
            engine = json.load(f)
        if engine.get("round_id") != round_id:
            raise ValueError("生成manifestのtoto回が保存対象と一致しません")
        if args.logical_season and str(engine.get("logical_season")) != args.logical_season:
            raise ValueError("生成manifestのシーズンが保存対象と一致しません")
        if engine.get("input_sha256") != sha256_file(os.path.join(srcdir, "predictions.csv")):
            raise ValueError("生成後にpredictions.csvが変更されています")
        for name, expected in engine.get("output_sha256", {}).items():
            if name not in required + optional:
                raise ValueError(f"未知の生成ファイル: {name}")
            source = os.path.join(srcdir, name)
            if not os.path.isfile(source) or sha256_file(source) != expected:
                raise ValueError(f"生成後にファイルが変更・削除されています: {name}")
        if engine.get("production_switched"):
            required += ["next_predictions.csv", "shadow_buyplan.csv", "shadow_buyplan.html"]
            if engine.get("official_version") in {"v2.3", "v2.4"}:
                required += ["shadow_buyplan_b.csv", "shadow_buyplan_b.html"]
            if "shadow_c_observation" in engine:
                required += ["shadow_buyplan_c.csv", "shadow_buyplan_c.html"]
            if "shadow_v25_observation" in engine:
                required += ['shadow_buyplan_v25.csv', 'shadow_buyplan_v25.html', 'shadow_v25_features.csv', 'shadow_v25_candidates.csv', 'shadow_v25_audit.json', 'shadow_v25_reference.json']
            for fn in required:
                if not os.path.isfile(os.path.join(srcdir, fn)):
                    raise FileNotFoundError(f"必要ファイルが見つかりません: {fn}")
    parent = os.path.dirname(os.path.abspath(outdir))
    os.makedirs(parent, exist_ok=True)
    copied = []
    with tempfile.TemporaryDirectory(prefix=".snapshot-", dir=parent) as stage:
        for fn in dict.fromkeys(required + optional):
            source = os.path.join(srcdir, fn)
            if not os.path.isfile(source):
                continue
            name = "buyplan_engine_manifest.json" if fn == "manifest.json" else fn
            shutil.copy2(source, os.path.join(stage, name))
            copied.append(name)
        manifest = {
            "schema_version": 2,
            "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "round_id": round_id,
            "logical_season": str(args.logical_season).strip(),
            "prediction_season": str(args.prediction_season).strip(),
            "j1_source": os.path.abspath(args.j1_source) if args.j1_source else "",
            "j2_source": os.path.abspath(args.j2_source) if args.j2_source else "",
            "football_lab_snapshot": str(args.football_lab_snapshot).strip(),
            "files": {
                name: {"size": os.path.getsize(os.path.join(stage, name)),
                       "sha256": sha256_file(os.path.join(stage, name))}
                for name in copied
            },
        }
        with open(os.path.join(stage, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write("\n")
        # Reserve the final directory exclusively; no existing snapshot is replaced.
        os.mkdir(outdir)
        try:
            for name in copied + ["manifest.json"]:
                os.replace(os.path.join(stage, name), os.path.join(outdir, name))
        except Exception:
            shutil.rmtree(outdir)
            raise
    copied = [os.path.join(outdir, name) for name in copied + ["manifest.json"]]

    print(f"[OK] snapshot saved: {outdir}")
    for p in copied:
        print(p)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)
