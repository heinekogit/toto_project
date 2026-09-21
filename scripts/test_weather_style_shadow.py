import unittest
import pandas as pd
from weather_style_shadow import apply_weather_uncertainty_shadow, FEATURES


class StyleWeatherTest(unittest.TestCase):
    def frame(self):
        row={'prob_final_home':.5,'prob_final_draw':.25,'prob_final_away':.25,
             'is_rain':True,'is_heavy_rain':True,'is_strong_wind':True,'weather_missing':False}
        for features in FEATURES.values():
            for field,lo,hi in features:
                row[field+'_home']=hi
                row[field+'_away']=lo
        return pd.DataFrame([row])

    def test_favourite_loses_probability_to_both_alternatives(self):
        out=apply_weather_uncertainty_shadow(self.frame())
        self.assertLess(out.prob_final_home.iloc[0], .5)
        self.assertGreater(out.prob_final_away.iloc[0], .25)
        self.assertGreater(out.prob_final_draw.iloc[0], .25)
        self.assertGreater(out.shadow_entropy_delta.iloc[0], 0)
        self.assertAlmostEqual(out[['prob_final_home','prob_final_draw','prob_final_away']].sum(axis=1).iloc[0],1)

    def test_dry_and_missing_weather_no_change(self):
        for missing in [False,True]:
            x=self.frame();x['weather_missing']=missing
            if not missing:x[['is_rain','is_heavy_rain','is_strong_wind']]=False
            out=apply_weather_uncertainty_shadow(x)
            self.assertEqual(out.prob_final_home.iloc[0],.5)

    def test_missing_style_uses_baseline_split(self):
        x=self.frame().drop(columns='stats_1試合平均パス数_home')
        out=apply_weather_uncertainty_shadow(x)
        self.assertFalse(out.shadow_style_assessable.iloc[0])
        self.assertGreater(out.shadow_probability_shift.iloc[0],0)

    def test_disabled_baseline_is_exact(self):
        x=self.frame();out=apply_weather_uncertainty_shadow(x,0,0)
        self.assertEqual(out.prob_final_home.iloc[0],.5)

    def test_outcomes_do_not_affect_pre_match_scores(self):
        x=self.frame();a=apply_weather_uncertainty_shadow(x);x['actual_result']='H';b=apply_weather_uncertainty_shadow(x)
        self.assertEqual(a.prob_final_home.iloc[0],b.prob_final_home.iloc[0])

    def test_side_favourite_only_does_not_move_draw_favourite(self):
        x=self.frame();x[['prob_final_home','prob_final_draw','prob_final_away']]=[.30,.45,.25]
        out=apply_weather_uncertainty_shadow(x, side_favourite_only=True)
        self.assertEqual(out.shadow_probability_shift.iloc[0],0)
        self.assertEqual(out.prob_final_draw.iloc[0],.45)

    def test_intermediate_model_probabilities_are_preserved(self):
        x=self.frame();x[['prob_main_home','prob_main_draw','prob_main_away']]=[.60,.20,.20]
        out=apply_weather_uncertainty_shadow(x)
        self.assertEqual(out.prob_main_home.iloc[0],.60)

    def test_second_application_is_rejected(self):
        once=apply_weather_uncertainty_shadow(self.frame())
        with self.assertRaises(ValueError):
            apply_weather_uncertainty_shadow(once)


if __name__=='__main__':unittest.main()
