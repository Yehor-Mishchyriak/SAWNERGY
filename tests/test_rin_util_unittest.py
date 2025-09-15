import unittest
from tests.util_import import load_core_sawnergy, load_module

load_core_sawnergy()
rin_util = load_module("sawnergy.rin.rin_util", "rin_util.py")

class TestCpptrajScript(unittest.TestCase):
    def test_from_add_redirect_render(self):
        s1 = rin_util.CpptrajScript.from_cmd("parm foo.prmtop")
        s2 = s1 + "trajin traj.nc"
        s3 = s2 > "out.txt"
        text = s3.render()
        self.assertIn("parm foo.prmtop", text)
        self.assertIn("trajin traj.nc out out.txt", text)
        # 'run' should be auto-inserted if missing
        self.assertIn("\nrun\n", text)

    def test_render_does_not_duplicate_run(self):
        s = rin_util.CpptrajScript.from_cmd("run")
        txt = s.render()
        # Should contain exactly one 'run' (no duplication)
        self.assertEqual(txt.count("\nrun\n"), 1)

    def test_COM_STDOUT_shape(self):
        s = rin_util.COM_STDOUT(1)
        txt = s.render()
        self.assertIn("dataset legend $R COM$i", txt)
        self.assertIn("printdata COMX*", txt)

if __name__ == "__main__":
    unittest.main()
