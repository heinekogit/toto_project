#!/usr/bin/env python3
"""Generate a normalized, opt-in travel matrix without replacing production data."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

from jleague_team_names import canonical_team_name

ROOT = Path(__file__).resolve().parents[1]


def haversine(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    radius = 6371.0
    p1, p2 = math.radians(a_lat), math.radians(b_lat)
    dp, dl = math.radians(b_lat-a_lat), math.radians(b_lon-a_lon)
    x = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return radius * 2 * math.atan2(math.sqrt(x), math.sqrt(1-x))


def build(output: Path) -> Path:
    frames = [pd.read_csv(ROOT / "data/manual/stadiums.csv", encoding="utf-8-sig")]
    supplemental = ROOT / "data/manual/stadiums_supplemental.csv"
    if supplemental.exists():
        frames.append(pd.read_csv(supplemental, encoding="utf-8-sig"))
    stadiums = pd.concat(frames, ignore_index=True)
    stadiums["team"] = stadiums.team.map(canonical_team_name)
    stadiums["lat"] = pd.to_numeric(stadiums.lat, errors="coerce")
    stadiums["lon"] = pd.to_numeric(stadiums.lon, errors="coerce")
    stadiums = stadiums.dropna(subset=["team", "lat", "lon"])
    conflicts = stadiums.groupby("team")[["lat", "lon"]].nunique().max(axis=1)
    if (conflicts > 1).any():
        raise ValueError(f"conflicting home coordinates: {conflicts[conflicts > 1].index.tolist()}")
    coords = stadiums.drop_duplicates("team").set_index("team")[["lat", "lon"]].sort_index()
    teams = coords.index.tolist()
    matrix = pd.DataFrame(index=teams, columns=teams, dtype=float)
    for away in teams:
        for home in teams:
            matrix.loc[away, home] = round(haversine(coords.at[away, "lat"], coords.at[away, "lon"],
                                                     coords.at[home, "lat"], coords.at[home, "lon"]), 1)
    matrix.index.name = "ホーム　／　アウェイ"
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output, sep="\t", encoding="utf-8-sig")
    return output


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path,
                   default=ROOT / "data/manual/team_travel_distances_normalized_all.csv")
    args = p.parse_args()
    path = build(args.output)
    matrix = pd.read_csv(path, sep="\t", encoding="utf-8-sig", index_col=0)
    print(f"OK: 正規化距離行列 {len(matrix)}クラブ: {path.resolve()}")
    print("NOTE: data/manual/team_travel_distances.csv は変更していません。")


if __name__ == "__main__":
    main()
