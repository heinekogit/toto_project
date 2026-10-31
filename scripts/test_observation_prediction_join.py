import importlib.util
from pathlib import Path
import unittest
import pandas as pd

spec = importlib.util.spec_from_file_location('observation_join', Path(__file__).parent/'eval/04_accumulate_observations.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ObservationJoinTests(unittest.TestCase):
    def test_fullwidth_and_halfwidth_names(self):
        buy = pd.Series(dict(match_no=1,home_team='C大阪',away_team='横浜ＦM',league='J1'))
        predictions = pd.DataFrame([dict(home_team='Ｃ大阪',away_team='横浜FM',league='j1')])
        self.assertEqual(module._join_prediction(buy,predictions).home_team,'Ｃ大阪')

    def test_match_number_does_not_override_wrong_card(self):
        buy = pd.Series(dict(match_no=1,home_team='C大阪',away_team='横浜FM'))
        predictions = pd.DataFrame([dict(match_no=1,home_team='浦和',away_team='鹿島')])
        with self.assertRaises(RuntimeError): module._join_prediction(buy,predictions)

    def test_ambiguous_and_reversed_cards_rejected(self):
        buy = pd.Series(dict(home_team='C大阪',away_team='横浜FM'))
        for rows in [[dict(home_team='横浜FM',away_team='C大阪')],
                     [dict(home_team='C大阪',away_team='横浜FM')]*2]:
            with self.assertRaises(RuntimeError): module._join_prediction(buy,pd.DataFrame(rows))


if __name__=='__main__': unittest.main()
