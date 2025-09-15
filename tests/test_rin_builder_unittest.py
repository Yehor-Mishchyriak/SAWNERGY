import unittest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import patch, MagicMock
from tests.util_import import load_core_sawnergy, load_module

# Wire up pseudo package and load the modules under their expected names
load_core_sawnergy()
rin_builder = load_module("sawnergy.rin.rin_builder", "rin_builder.py")

# Aliases to target correct patch paths inside rin_builder module
RB_MOD = "sawnergy.rin.rin_builder"
RIN_UTIL = f"{RB_MOD}.rin_util"
SAW_UTIL = f"{RB_MOD}.sawnergy_util"

class TestRINBuilder_Init(unittest.TestCase):
    def test_init_uses_locate_cpptraj_and_stores_path(self):
        with patch(f"{RIN_UTIL}.locate_cpptraj", return_value="/opt/amber/bin/cpptraj") as loc:
            rb = rin_builder.RINBuilder(cpptraj_path=None)
            self.assertEqual(rb.cpptraj, "/opt/amber/bin/cpptraj")
            loc.assert_called_once_with(explicit=None, verify=True)

    def test_init_accepts_string_path_and_passes_to_locate(self):
        with patch(f"{RIN_UTIL}.locate_cpptraj", return_value="/x/cpptraj") as loc:
            rb = rin_builder.RINBuilder(cpptraj_path="/y/cpptraj")
            self.assertEqual(rb.cpptraj, "/x/cpptraj")
            # ensure Path conversion was acceptable to locate_cpptraj
            args, kwargs = loc.call_args
            self.assertTrue(hasattr(args[0], "resolve"))  # got Path
            self.assertEqual(kwargs.get("verify", True), True)

class TestRINBuilder_NumberFrames(unittest.TestCase):
    def test_get_number_frames_parses_integer(self):
        with patch(f"{RIN_UTIL}.run_cpptraj", return_value="Frames: 123\n"):
            rb = rin_builder.RINBuilder(cpptraj_path=None)
            n = rb._get_number_frames("top.prmtop", "traj.nc", subprocess_env=None, timeout=3.0)
            self.assertEqual(n, 123)

    def test_get_number_frames_raises_on_bad_output(self):
        with patch(f"{RIN_UTIL}.run_cpptraj", return_value="Frames: not_an_int"):
            rb = rin_builder.RINBuilder(cpptraj_path=None)
            with self.assertRaises(RuntimeError):
                rb._get_number_frames("top.prmtop", "traj.nc")

class TestRINBuilder_Composition(unittest.TestCase):
    def test_get_atomic_composition_success_and_cleans_tempfile(self):
        # Mock parser to return a dict that includes our molecule_id
        fake_hierarchy = {1: {"RES1": {1,2,3}, "RES2": {4}}}
        mock_parser = MagicMock(return_value=fake_hierarchy)

        # Prepare a real temp file path the function will try to unlink
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        tmpfile = Path(tmpdir.name) / "composition.dat"
        tmpfile.write_text("dummy", encoding="utf-8")

        # Patch temporary_file to return our path; run_cpptraj doesn't need to do anything
        with patch(f"{SAW_UTIL}.temporary_file", return_value=tmpfile), \
             patch(f"{RIN_UTIL}.run_cpptraj", return_value="OK"), \
             patch(f"{RIN_UTIL}.CpptrajMaskParser.hierarchize_molecular_composition", mock_parser):
            rb = rin_builder.RINBuilder(cpptraj_path=None)
            comp = rb._get_atomic_composition_of_molecule(
                "top", "traj", molecule_id=1, subprocess_env=None, timeout=2.0
            )
            self.assertEqual(comp, fake_hierarchy[1])
            # The temp file should be removed in finally:
            self.assertFalse(tmpfile.exists(), "Expected temp composition file to be unlinked")

    def test_get_atomic_composition_missing_key_raises_KeyError(self):
        fake_hierarchy = {2: {"RESZ": {9}}}
        mock_parser = MagicMock(return_value=fake_hierarchy)
        tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(tmpdir.cleanup)
        tmpfile = Path(tmpdir.name) / "composition.dat"
        tmpfile.write_text("dummy", encoding="utf-8")
        with patch(f"{SAW_UTIL}.temporary_file", return_value=tmpfile), \
             patch(f"{RIN_UTIL}.run_cpptraj", return_value="OK"), \
             patch(f"{RIN_UTIL}.CpptrajMaskParser.hierarchize_molecular_composition", mock_parser):
            rb = rin_builder.RINBuilder(cpptraj_path=None)
            with self.assertRaises(KeyError):
                rb._get_atomic_composition_of_molecule("top", "traj", molecule_id=1)

class TestRINBuilder_PairwiseEnergies(unittest.TestCase):
    def _compose_output(self, emap_vals, vmap_vals):
        # Craft cpptraj output matching regex blocks expected by _elec_vdw_pattern
        emap = " ".join(map(str, emap_vals))
        vmap = " ".join(map(str, vmap_vals))
        return (
            "[printdata PW[EMAP] square2d noheader]\n"
            f"{emap}\n"
            "[printdata PW[VMAP] square2d noheader]\n"
            f"{vmap}\n"
        )

    def test_calc_avg_atomic_interactions_happy_path(self):
        # 2x2 blocks -> 4 values each; output sums EMAP+VMAP as float32
        out = self._compose_output([1,2,3,4], [10,20,30,40])
        captured_script = {}
        def fake_run_cpptraj(*, script=None, env=None, timeout=None, **kwargs):
            captured_script["script"] = script
            return out

        with patch(f"{RIN_UTIL}.run_cpptraj", side_effect=fake_run_cpptraj):
            rb = rin_builder.RINBuilder(cpptraj_path=None)
            M = rb._calc_avg_atomic_interactions_in_frames(
                frame_range=(1,2),
                topology_file="top",
                trajectory_file="traj",
                molecule_id=1,
                subprocess_env={"OMP_NUM_THREADS":"1"},
                timeout=1.5
            )
            self.assertEqual(M.shape, (2,2))
            self.assertEqual(M.dtype, np.float32)
            np.testing.assert_array_equal(M, np.array([[11,22],[33,44]], dtype=np.float32))
            # Script should contain both printdata lines (sanity of script composition)
            self.assertIn("printdata PW[EMAP] square2d noheader", captured_script["script"])
            self.assertIn("printdata PW[VMAP] square2d noheader", captured_script["script"])

    def test_calc_avg_atomic_interactions_size_mismatch_raises(self):
        out = self._compose_output([1,2,3,4], [10,20,30])  # VMAP shorter
        with patch(f"{RIN_UTIL}.run_cpptraj", return_value=out):
            rb = rin_builder.RINBuilder(cpptraj_path=None)
            with self.assertRaises(ValueError):
                rb._calc_avg_atomic_interactions_in_frames((1,1), "t", "y", 1)

    def test_calc_avg_atomic_interactions_non_square_raises(self):
        # 6 values cannot form a square
        out = self._compose_output([1,2,3,4,5,6], [10,20,30,40,50,60])
        with patch(f"{RIN_UTIL}.run_cpptraj", return_value=out):
            rb = rin_builder.RINBuilder(cpptraj_path=None)
            with self.assertRaises(ValueError):
                rb._calc_avg_atomic_interactions_in_frames((1,1), "t", "y", 1)

    def test_calc_avg_atomic_interactions_missing_blocks_raises(self):
        # Missing EMAP/VMAP markers entirely
        with patch(f"{RIN_UTIL}.run_cpptraj", return_value="some other output"):
            rb = rin_builder.RINBuilder(cpptraj_path=None)
            with self.assertRaises(ValueError):
                rb._calc_avg_atomic_interactions_in_frames((1,1), "t", "y", 1)

class TestRINBuilder_COMs(unittest.TestCase):
    def test_get_residue_COMs_per_frame_validates_frame_range(self):
        rb = rin_builder.RINBuilder(cpptraj_path=None)
        with self.assertRaises(ValueError):
            rb._get_residue_COMs_per_frame(
                frame_range=(5, 3),  # end < start -> invalid
                topology_file="t",
                trajectory_file="y",
                molecule_id=1,
                number_residues=2,
            )

    # NOTE: The full parsing of COM blocks depends on longer, truncated function body.
    # If you want, we can extend this with a crafted cpptraj output once full parser is available.

if __name__ == "__main__":
    unittest.main()
