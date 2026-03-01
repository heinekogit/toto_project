import importlib.util
import os
import pandas as pd


def _load_correction_module():
    path = os.path.join(os.path.dirname(__file__), "24_apply_home_away_correction.py")
    spec = importlib.util.spec_from_file_location("home_away_correction", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    m = _load_correction_module()
    output_csv = os.environ.get(
        "OUTPUT_CSV",
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "data",
            "reports",
            "metrics",
            "home_away_correction_test_output.csv",
        ),
    )

    # ダミー試合データ + 予想確率
    df = pd.DataFrame(
        [
            {
                "match_id": "dummy_001",
                "home_team": "TeamA",
                "away_team": "TeamB",
                "home_advantage_diff": 0.80,
                "prob_home_win": 0.46,
                "prob_draw": 0.30,
                "prob_away_win": 0.24,
            },
            {
                "match_id": "dummy_002",
                "home_team": "TeamC",
                "away_team": "TeamD",
                "home_advantage_diff": -0.55,
                "prob_home_win": 0.33,
                "prob_draw": 0.34,
                "prob_away_win": 0.33,
            },
            {
                "match_id": "dummy_003",
                "home_team": "TeamE",
                "away_team": "TeamF",
                "home_advantage_diff": 0.10,
                "prob_home_win": 0.38,
                "prob_draw": 0.29,
                "prob_away_win": 0.33,
            },
        ]
    )

    print("=== BEFORE ===")
    print(
        df[["match_id", "home_advantage_diff", "prob_home_win", "prob_draw", "prob_away_win"]].to_string(index=False)
    )

    corrected = m.apply_home_advantage_diff_correction_to_dataframe(
        df=df,
        diff_col="home_advantage_diff",
        home_col="prob_home_win",
        draw_col="prob_draw",
        away_col="prob_away_win",
        k=0.03,
        max_abs_delta=0.05,
        out_home_col="prob_home_win_after",
        out_draw_col="prob_draw_after",
        out_away_col="prob_away_win_after",
    )

    corrected["sum_before"] = (
        corrected["prob_home_win"] + corrected["prob_draw"] + corrected["prob_away_win"]
    ).round(6)
    corrected["sum_after"] = (
        corrected["prob_home_win_after"] + corrected["prob_draw_after"] + corrected["prob_away_win_after"]
    ).round(6)

    print("\n=== AFTER ===")
    print(
        corrected[
            [
                "match_id",
                "home_advantage_diff",
                "prob_home_win_after",
                "prob_draw_after",
                "prob_away_win_after",
                "sum_before",
                "sum_after",
            ]
        ].to_string(index=False)
    )

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    corrected.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()
