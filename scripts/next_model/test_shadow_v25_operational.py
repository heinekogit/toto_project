import unittest,tempfile,json
from pathlib import Path
import numpy as np
import pandas as pd
from shadow_v25_operational import build,write,ROOT,FILES

class C95OperationalTest(unittest.TestCase):
    def test_frozen_1670_reproduction_and_no_mutation(self):
        f=pd.read_csv(ROOT/'data/reports/toto1670_fixed_exact13/forecast_input.csv')
        official=pd.read_csv(ROOT/'data/purchase_reference/buyplan.csv')
        f0=f.copy(deep=True);o0=official.copy(deep=True)
        result=build(f,official);pd.testing.assert_frame_equal(f,f0);pd.testing.assert_frame_equal(official,o0)
        expected=pd.read_csv(ROOT/'data/reports/toto1670_draw_count_cover/C95_tickets.csv')
        for j in range(1,11):self.assertEqual(result[0][f'ticket{j:02}'].map({1:'H',0:'D',2:'A'}).tolist(),expected[f'{j:02}'].tolist())
        a=result[3];self.assertGreaterEqual(a['p13'],a['p13_floor']);self.assertEqual(a['mode'],'C95');self.assertFalse(a['probabilities_changed'])
        for s in [0,1,2]:np.testing.assert_array_equal((official[[f'ticket{j:02}' for j in range(1,11)]]==s).sum(axis=1),(result[0][[f'ticket{j:02}' for j in range(1,11)]]==s).sum(axis=1))
        with tempfile.TemporaryDirectory() as tmp:
            write(result,tmp,'toto1670');self.assertTrue(all((Path(tmp)/name).is_file() for name in FILES))
            self.assertEqual(json.loads((Path(tmp)/FILES[5]).read_text())['mode'],'C95')
            self.assertIn('C95シャドー',(Path(tmp)/FILES[1]).read_text())

if __name__=='__main__':unittest.main()
