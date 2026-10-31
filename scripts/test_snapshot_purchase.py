import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT = Path(__file__).parent / "eval/00_snapshot_purchase.py"


class SnapshotTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.out = self.root / "saved"
        for name in ("predictions.csv", "buyplan.csv", "next_predictions.csv", "shadow_buyplan.csv", "shadow_buyplan.html"):
            (self.source / name).write_text("fixture\n")
        self.engine = dict(round_id="toto9999", logical_season=2027, production_switched=True,
                           input_sha256=hashlib.sha256(b"fixture\n").hexdigest())
        self.write_engine()

    def write_engine(self):
        (self.source / "manifest.json").write_text(json.dumps(self.engine))

    def run_snapshot(self):
        return subprocess.run([sys.executable, str(SCRIPT), "--round", "toto9999", "--srcdir", str(self.source),
                               "--outdir", str(self.out), "--logical-season", "2027"], capture_output=True, text=True)

    def test_preserves_engine_and_all_hashes_and_no_overwrite(self):
        result = self.run_snapshot()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.out / "buyplan_engine_manifest.json").read_bytes(), (self.source / "manifest.json").read_bytes())
        manifest = json.loads((self.out / "manifest.json").read_text())
        for name, record in manifest["files"].items():
            self.assertEqual(record["sha256"], hashlib.sha256((self.out / name).read_bytes()).hexdigest())
        self.assertNotEqual(self.run_snapshot().returncode, 0)

    def test_missing_required_leaves_no_snapshot(self):
        (self.source / "buyplan.csv").unlink()
        self.assertNotEqual(self.run_snapshot().returncode, 0)
        self.assertFalse(self.out.exists())

    def test_round_mismatch(self):
        self.engine["round_id"] = "toto9998"
        self.write_engine()
        self.assertNotEqual(self.run_snapshot().returncode, 0)
        self.assertFalse(self.out.exists())

    def test_input_changed(self):
        (self.source / "predictions.csv").write_text("changed")
        self.assertNotEqual(self.run_snapshot().returncode, 0)
        self.assertFalse(self.out.exists())

    def test_output_changed(self):
        self.engine["output_sha256"] = {"buyplan.csv": "wrong"}
        self.write_engine()
        self.assertNotEqual(self.run_snapshot().returncode, 0)
        self.assertFalse(self.out.exists())

    def test_v23_requires_and_preserves_both_shadows(self):
        self.engine["official_version"] = "v2.3"
        self.write_engine()
        self.assertNotEqual(self.run_snapshot().returncode, 0)
        for name in ("shadow_buyplan_b.csv", "shadow_buyplan_b.html"):
            (self.source / name).write_text("shadow B fixture\n")
        self.assertEqual(self.run_snapshot().returncode, 0)
        self.assertEqual((self.out / "shadow_buyplan_b.csv").read_text(), "shadow B fixture\n")

    def test_v24_requires_and_preserves_both_shadows(self):
        self.engine["official_version"] = "v2.4"
        self.write_engine()
        self.assertNotEqual(self.run_snapshot().returncode, 0)
        for name in ("shadow_buyplan_b.csv", "shadow_buyplan_b.html"):
            (self.source / name).write_text("shadow B fixture\n")
        result = self.run_snapshot()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.out / "shadow_buyplan_b.csv").read_text(), "shadow B fixture\n")

    def test_shadow_c_is_required_when_declared_and_archived(self):
        self.engine["shadow_c_observation"] = {"generator": "free_coverage_shadow_c_v1"}
        self.write_engine()
        self.assertNotEqual(self.run_snapshot().returncode, 0)
        for name in ("shadow_buyplan_c.csv", "shadow_buyplan_c.html"):
            (self.source / name).write_text("shadow C fixture\n")
        result = self.run_snapshot()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual((self.out / "shadow_buyplan_c.csv").read_text(), "shadow C fixture\n")
