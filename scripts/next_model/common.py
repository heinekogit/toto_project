from __future__ import annotations

import hashlib
import math
import unicodedata
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
VALIDATION_ROUNDS = (
    "toto1644", "toto1645", "toto1647", "toto1648",
    "toto1649", "toto1650", "toto1653", "toto1654",
)
PREMATCH_PROVEN_ROUNDS = {"toto1649", "toto1650", "toto1653", "toto1654"}


def normalize_text(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value)).strip()


def match_key(home: object, away: object) -> str:
    return f"{normalize_text(home)}|{normalize_text(away)}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def number(value: object, default: float = math.nan) -> float:
    parsed = pd.to_numeric(value, errors="coerce")
    return default if pd.isna(parsed) else float(parsed)


def result_symbol(value: object) -> str:
    numeric = pd.to_numeric(value, errors="coerce")
    if pd.isna(numeric) or int(numeric) not in {0, 1, 2}:
        return ""
    return {1: "H", 0: "D", 2: "A"}[int(numeric)]
