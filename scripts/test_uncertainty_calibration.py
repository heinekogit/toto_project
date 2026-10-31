import unittest

import numpy as np
import pandas as pd

from uncertainty_calibration import adjust, apply_to_frame


class UncertaintyCalibrationTests(unittest.TestCase):
    def test_rain_signal_can_be_tuned_without_changing_fatigue_signal(self):
        frame = pd.DataFrame([{
            "prob_final_home": 0.6, "prob_final_draw": 0.25, "prob_final_away": 0.15,
            "home_total_fatigue_score": 0.0, "away_total_fatigue_score": 0.0,
            "is_rain": True, "is_heavy_rain": False, "is_strong_wind": False,
            "weather_missing": False,
        }])
        default = apply_to_frame(frame, 0.25, 0.25)
        stronger = apply_to_frame(frame, 0.25, 0.25, rain_signal=0.7)
        self.assertEqual(default.uncertainty_fatigue_signal.iloc[0], stronger.uncertainty_fatigue_signal.iloc[0])
        self.assertGreater(stronger.uncertainty_weather_signal.iloc[0], default.uncertainty_weather_signal.iloc[0])
        self.assertGreater(stronger.uncertainty_calibration_mix.iloc[0], default.uncertainty_calibration_mix.iloc[0])

    def test_monotone_uncertainty_preserves_rank_and_probability_sum(self):
        probs = np.array([[0.60, 0.25, 0.15], [0.20, 0.50, 0.30]])
        adjusted, mix = adjust(probs, np.array([0.8, 0.8]), np.array([1.0, 0.0]), 0.25, 0.25)
        self.assertTrue(np.allclose(adjusted.sum(axis=1), 1.0))
        self.assertTrue(np.array_equal(adjusted.argmax(axis=1), probs.argmax(axis=1)))
        self.assertTrue(np.all(mix > 0))
        self.assertTrue(np.all(mix <= 0.25))
        before_entropy = -(probs * np.log(probs)).sum(axis=1)
        after_entropy = -(adjusted * np.log(adjusted)).sum(axis=1)
        self.assertTrue(np.all(after_entropy > before_entropy))

    def test_missing_weather_retains_fatigue_and_is_flagged(self):
        frame = pd.DataFrame([{
            "prob_final_home": 0.6, "prob_final_draw": 0.25, "prob_final_away": 0.15,
            "prob_home_win": 0.6, "prob_draw": 0.25, "prob_away_win": 0.15,
            "home_total_fatigue_score": 4.0, "away_total_fatigue_score": 8.0,
            "is_rain": None, "is_heavy_rain": None, "is_strong_wind": None,
            "weather_missing": True,
        }])
        result = apply_to_frame(frame, 0.25, 0.25)
        self.assertFalse(result.uncertainty_weather_known.iloc[0])
        self.assertEqual(result.uncertainty_weather_signal.iloc[0], 0.0)
        self.assertGreater(result.uncertainty_calibration_mix.iloc[0], 0.0)
        self.assertAlmostEqual(result.prob_final_home.iloc[0], result.prob_home_win.iloc[0])
        with self.assertRaises(ValueError):
            apply_to_frame(result, 0.25, 0.25)


if __name__ == "__main__":
    unittest.main()
