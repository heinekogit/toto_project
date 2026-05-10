import argparse
import os
import logging
import pandas as pd
from datetime import timedelta

from io_utils import load_stadiums, load_matches, merge_matches_with_stadiums
from open_meteo_client import OpenMeteoClient
from feature_builder import hourly_json_to_df, build_features_for_match


def setup_logger(log_path):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch weather features from Open-Meteo.")
    parser.add_argument("--stadiums", required=True, help="スタジアム一覧ファイル（Excel/CSV）")
    parser.add_argument("--stadiums-sheet", default="stadiums", help="スタジアム一覧のシート名")
    parser.add_argument("--matches", required=True, help="試合日程ファイル（Excel/CSV）")
    parser.add_argument("--matches-sheet", default=None, help="試合日程のシート名")
    parser.add_argument("--out", default="weather_features.csv", help="出力CSVファイル名")
    parser.add_argument("--cache-dir", default="weather_raw", help="生データキャッシュ保存先")
    parser.add_argument(
        "--lookahead-days",
        type=int,
        default=None,
        help="基準日から何日先までを天候取得対象にするか（未指定で全件）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_logger(os.path.join("logs", "weather_fetch.log"))
    # 天候取得成功率の最小閾値（未指定時 0.7）
    min_success_ratio = float(os.environ.get("WEATHER_MIN_SUCCESS_RATIO", "0.7"))

    stadiums = load_stadiums(args.stadiums, sheet=args.stadiums_sheet)
    matches = load_matches(args.matches, sheet=args.matches_sheet)

    merged, key = merge_matches_with_stadiums(matches, stadiums)

    if args.lookahead_days is not None:
        asof_str = os.environ.get("WEATHER_ASOF_DATE")
        asof = pd.to_datetime(asof_str).normalize() if asof_str else pd.Timestamp.now().normalize()
        horizon = asof + timedelta(days=args.lookahead_days)
        kickoff_ts = pd.to_datetime(merged.get("kickoff_jst"), errors="coerce")
        before_count = len(merged)
        merged = merged[(kickoff_ts >= asof) & (kickoff_ts <= horizon)].copy()
        logging.info(
            "WEATHER_TARGET_FILTER: asof=%s lookahead_days=%s before=%s after=%s",
            asof.date(),
            args.lookahead_days,
            before_count,
            len(merged),
        )

    required_cols = ["match_id", "kickoff_jst", "stadium_name", "lat", "lon"]
    for col in required_cols:
        if col not in merged.columns:
            raise ValueError(f"必要な列が見つかりません: {col}")

    client = OpenMeteoClient(cache_dir=args.cache_dir)

    records = []
    total = 0
    eligible_total = 0
    success = 0
    required_feature_keys = ["is_rain", "is_heavy_rain", "is_strong_wind"]
    error_messages = []

    for _, row in merged.iterrows():
        total += 1
        match_id = row.get("match_id")
        kickoff = row.get("kickoff_jst")
        stadium_name = row.get("stadium_name")
        lat = row.get("lat")
        lon = row.get("lon")

        base_record = {
            "match_id": match_id,
            "kickoff_jst": kickoff,
            "stadium_name": stadium_name,
            "lat": lat,
            "lon": lon,
            "weather_fetch_ok": 0,
        }

        if pd.isna(kickoff) or pd.isna(lat) or pd.isna(lon):
            msg = f"欠損: match_id={match_id}, stadium={stadium_name}, lat/lon or kickoff"
            logging.warning(msg)
            error_messages.append(msg)
            records.append(base_record)
            continue
        eligible_total += 1

        try:
            datasets = client.fetch_hourly_range(kickoff, float(lat), float(lon))
            hourly_df = pd.concat([hourly_json_to_df(d) for d in datasets], ignore_index=True)
            features = build_features_for_match(row, hourly_df)
            base_record.update(features)
            fetch_ok = all((k in features) and pd.notna(features.get(k)) for k in required_feature_keys)
            if fetch_ok:
                base_record["weather_fetch_ok"] = 1
                success += 1
        except Exception as e:
            msg = f"取得失敗: match_id={match_id}, stadium={stadium_name}, lat={lat}, lon={lon}, err={e}"
            logging.warning(msg)
            error_messages.append(msg)
        records.append(base_record)

    if total == 0:
        raise RuntimeError("WEATHER_FETCH: total=0（対象試合が0件）")

    if eligible_total == 0:
        raise RuntimeError("WEATHER_FETCH: eligible_total=0（lat/lon+kickoff が揃った対象が0件）")

    success_ratio = success / eligible_total
    overall_missing_ratio = (total - success) / total
    logging.info(
        "WEATHER_FETCH: success=%s eligible_total=%s total=%s success_ratio=%.3f overall_missing_ratio=%.3f threshold=%.3f",
        success,
        eligible_total,
        total,
        success_ratio,
        overall_missing_ratio,
        min_success_ratio,
    )

    df_out = pd.DataFrame(records)
    df_out["weather_missing_ratio"] = overall_missing_ratio
    df_out.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"出力: {args.out}")
    print(
        f"WEATHER_FETCH: success={success} eligible_total={eligible_total} "
        f"total={total} success_ratio={success_ratio:.3f}"
    )
    print(f"WEATHER_FETCH_ERRORS: count={len(error_messages)}")
    if error_messages:
        print("WEATHER_FETCH_ERRORS_DETAIL_BEGIN")
        for msg in error_messages:
            print(msg)
        print("WEATHER_FETCH_ERRORS_DETAIL_END")

    # 動作確認観点:
    # - total=20 success=0 なら success_ratio=0.0 で閾値未満 -> 例外停止
    # - total>0 かつ success_ratio>=閾値 なら通常完了
    if success_ratio < min_success_ratio:
        raise RuntimeError(
            f"WEATHER_FETCH: success_ratio below threshold "
            f"(success={success}, eligible_total={eligible_total}, total={total}, ratio={success_ratio:.3f}, "
            f"threshold={min_success_ratio:.3f}, error_count={len(error_messages)})"
        )


if __name__ == "__main__":
    main()
