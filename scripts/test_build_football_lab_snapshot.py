import unittest

from build_football_lab_snapshot import (
    build_wide_metrics,
    extract_team_tables,
    maybe_float,
    validate_chance_metrics,
)


META = {
    "snapshot_date": "20260919",
    "asof_date": "20260914",
    "league": "j1",
    "season": "2026",
    "source_file": "chance.html",
    "page_label": "チャンス構築率",
    "title": "fixture",
}


def chance_table(section, labels, values):
    heads = "<th></th><th></th>" + "".join(
        f"<th>{label}</th><th></th>" for label in labels
    )
    cells = "".join(
        f"<td>{value}</td><td>({rank}位)</td>" for value, rank in values
    )
    return f"""
    <h3 class="boxHeader"><span>{section}</span></h3>
    <table class="statsTbl10"><thead><tr>{heads}</tr></thead><tbody><tr>
      <td class="tEmblem"></td><td class="tName"><span class="sp">鹿島</span></td>
      {cells}
    </tr></tbody></table>
    """


class FootballLabChanceTableTest(unittest.TestCase):
    def test_zero_metrics_are_preserved_and_missing_stays_empty(self):
        for value in (0, 0.0, "0.0", "0.0%"):
            with self.subTest(value=value):
                self.assertEqual(maybe_float(value), 0.0)
        for value in (None, "", " "):
            with self.subTest(value=value):
                self.assertEqual(maybe_float(value), "")

    def test_value_and_rank_columns_stay_aligned(self):
        html = chance_table(
            "チャンス構築率",
            ["攻撃回数", "シュート", "チャンス構築率", "ゴール", "シュート成功率"],
            [(117.4, 4), (16.3, 3), ("13.9%", 4), (1.9, 3), ("11.4%", 9)],
        ) + chance_table(
            "被チャンス構築率",
            ["被攻撃回数", "被シュート", "被チャンス構築率", "被ゴール", "被シュート成功率"],
            [(113.0, 13), (10.4, 4), ("9.2%", 4), (1.6, 13), ("15.1%", 19)],
        )
        rows = extract_team_tables(html, META)
        metrics = build_wide_metrics(rows, [])
        self.assertEqual(len(metrics), 1)
        row = metrics[0]
        self.assertEqual(row["flab_chance_attack_count"], 117.4)
        self.assertEqual(row["flab_chance_shots"], 16.3)
        self.assertEqual(row["flab_chance_build_rate"], 13.9)
        self.assertEqual(row["flab_chance_goals"], 1.9)
        self.assertEqual(row["flab_chance_shot_conversion"], 11.4)
        self.assertEqual(row["flab_chance_allowed_shots"], 10.4)
        self.assertEqual(row["flab_chance_shots_rank"], 3)
        self.assertEqual(row["flab_chance_allowed_shots_rank"], 4)
        validate_chance_metrics(metrics)

    def test_rank_text_in_numeric_field_stops_build(self):
        with self.assertRaisesRegex(RuntimeError, "numeric value required"):
            validate_chance_metrics([{"team_name": "鹿島", "flab_chance_attack_count": "(4位)"}])

    def test_unrelated_wide_row_is_not_treated_as_chance_team(self):
        validate_chance_metrics([{"team_name": "選手名", "flab_cbp_attack_points": 10.0}])


if __name__ == "__main__":
    unittest.main()
