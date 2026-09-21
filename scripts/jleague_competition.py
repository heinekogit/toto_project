"""Resolve competition IDs used by the J.League Data Site search form."""

from http_retry import post_with_retry


COMPETITIONS_URL = "https://data.j-league.or.jp/SFMS01/competitions"


def _parse_ids(raw):
    return [value.strip() for value in str(raw or "").split(",") if value.strip()]


def resolve_competition_ids(
    competition_year,
    competition_frame_ids,
    override=None,
    *,
    post_func=post_with_retry,
):
    """Return a comma-separated competition ID string.

    ``override`` retains the COMPETITION_IDS escape hatch. Automatic lookup is
    intentionally strict: the Data Site form endpoint accepts one year/frame
    pair and a normal league season must resolve to exactly one competition.
    """
    override_ids = _parse_ids(override)
    if override_ids:
        resolved = ",".join(dict.fromkeys(override_ids))
        print(f"[COMPETITION_ID] source=env resolved={resolved}")
        return resolved

    year = str(competition_year).strip()
    frame_ids = _parse_ids(competition_frame_ids)
    if not year:
        raise RuntimeError("大会ID自動解決に必要なcompetition_yearが空です。")
    if len(frame_ids) != 1:
        raise RuntimeError(
            "大会IDの自動解決ではcompetition_frame_idsを1件だけ指定してください: "
            f"value={competition_frame_ids!r}"
        )
    frame_id = frame_ids[0]

    response = post_func(
        COMPETITIONS_URL,
        data={"competition_year": year, "competition_frame_id": frame_id},
        timeout=(5, 20),
        max_retries=3,
    )
    try:
        payload = response.json()
    except (TypeError, ValueError) as exc:
        raise RuntimeError("大会ID自動解決の応答がJSONではありません。") from exc

    if not isinstance(payload, dict) or payload.get("error") is not False:
        raise RuntimeError(f"大会ID自動解決でエラー応答を受信しました: {payload!r}")
    rows = payload.get("data")
    if not isinstance(rows, list):
        raise RuntimeError(f"大会ID自動解決のdata形式が不正です: {payload!r}")

    candidates = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        parent = str(row.get("parentValue", "")).strip()
        value = str(row.get("selectValue", "")).strip()
        if parent == frame_id and value:
            candidates.append(value)
    candidates = list(dict.fromkeys(candidates))

    if len(candidates) != 1:
        raise RuntimeError(
            "大会IDを一意に自動解決できませんでした: "
            f"season={year} frame_id={frame_id} candidates={candidates}"
        )

    resolved = candidates[0]
    print(
        f"[COMPETITION_ID] source=data-site season={year} "
        f"frame_id={frame_id} resolved={resolved}"
    )
    return resolved
