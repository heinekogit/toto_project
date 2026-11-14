import unittest
import numpy as np
import pandas as pd
from analyze_phase35_high_draw_structure import BASE, conditions, match_controls

class FrozenConditionTests(unittest.TestCase):
    def setUp(self):
        self.d=pd.read_csv(BASE/'diagnostics_78.csv')
    def test_nested_membership_and_boundaries(self):
        c=conditions(self.d)
        self.assertEqual([int(c[k].sum()) for k in ['strict','central','inclusive']],[4,5,8])
        self.assertTrue((~c['strict']|c['central']).all())
        self.assertTrue((~c['central']|c['inclusive']).all())
        self.assertEqual(int((c['inclusive']&~c['strict']).sum()),4)
    def test_no_outcome_leakage(self):
        d=self.d.copy();a=conditions(d)
        for col in ['actual_symbol','actual_D','actual_home','actual_away']:
            d[col]=d[col].iloc[::-1].to_numpy()
        for key,value in conditions(d).items():pd.testing.assert_series_equal(value,a[key])
        for frame in [self.d,d]:
            frame['inclusive']=conditions(frame)['inclusive'];frame['HA_gap']=abs(frame.baseline_P_H-frame.baseline_P_A)
            frame['card']=frame.home_team+'-'+frame.away_team
        for mode in ['PD_HA','T_ratio']:
            x=match_controls(self.d,mode);y=match_controls(d,mode)
            pd.testing.assert_frame_equal(x[['round','target_no','control_no','matched']],y[['round','target_no','control_no','matched']])
            z=x[x.matched]
            self.assertFalse(z.duplicated(['round','control_no']).any())
    def test_rounded_display_not_selection_input(self):
        c=conditions(self.d)['inclusive']; shuffled=self.d.sample(frac=1,random_state=1)
        pd.testing.assert_series_equal(c,conditions(shuffled)['inclusive'].sort_index())

if __name__=='__main__':unittest.main()
