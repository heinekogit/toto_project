import unittest
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
from shadow_v25_legacy import build,write,FILES,ROOT,REFERENCE

class OperationalTest(unittest.TestCase):
    def test_production_input_unchanged_fixed_reference_and_output(self):
        source=ROOT/'data/eval/toto_rounds/toto1668/snapshot'
        f=pd.read_csv(source/'next_predictions.csv');p=pd.read_csv(source/'buyplan.csv')
        f0=f.copy(deep=True);p0=p.copy(deep=True);ref=REFERENCE.read_bytes()
        bundle=build(f,p);again=build(f,p)
        pd.testing.assert_frame_equal(f,f0);pd.testing.assert_frame_equal(p,p0)
        pd.testing.assert_frame_equal(bundle[0],again[0]);self.assertEqual(ref,REFERENCE.read_bytes())
        for s in [0,1,2]:
            np.testing.assert_array_equal((p[[f'ticket{i:02d}' for i in range(1,11)]]==s).sum(axis=1),(bundle[0][[f'ticket{i:02d}' for i in range(1,11)]]==s).sum(axis=1))
        with tempfile.TemporaryDirectory() as d:
            write(bundle,d,'toto1668')
            self.assertTrue(all((Path(d)/n).exists() for n in FILES))
            self.assertEqual((Path(d)/FILES[-1]).read_bytes(),ref)
            self.assertIn('購入対象外', (Path(d)/FILES[1]).read_text())
    def test_bad_reference_stops_without_fallback(self):
        source=ROOT/'data/eval/toto_rounds/toto1668/snapshot'
        with self.assertRaises(FileNotFoundError):build(pd.read_csv(source/'next_predictions.csv'),pd.read_csv(source/'buyplan.csv'),reference_path='/tmp/no_v25_reference_exists.json')

if __name__=='__main__':unittest.main()
