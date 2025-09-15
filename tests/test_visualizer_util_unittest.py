import unittest
import os
import sys
from unittest.mock import patch
from tests.util_import import load_module

vizu = load_module("visualizer_util", "visualizer_util.py")

class TestVisualizerUtil(unittest.TestCase):
    def test_map_groups_to_colors_basic(self):
        N = 5
        groups = (([1, 3], vizu.RED), ([5], vizu.BLUE))
        colors = vizu.map_groups_to_colors(N, groups, default_color=vizu.GRAY, one_based=True)
        self.assertEqual(len(colors), N)
        self.assertNotEqual(colors[0][:3], colors[1][:3])  # position 1 (1-based) colored
        self.assertEqual(colors[2][:3], colors[0][:3])     # 1 & 3 same color
        self.assertNotEqual(colors[4][:3], colors[0][:3])  # 5th is BLUE

    def test_map_groups_to_colors_oob(self):
        N = 4
        with self.assertRaises(IndexError):
            vizu.map_groups_to_colors(N, (([5], vizu.RED),), default_color=vizu.GRAY)

    def test_ensure_backend_switches_in_headless_linux(self):
        calls = {}
        def fake_use(backend, force=False):
            calls["backend"] = backend
            calls["force"] = force

        with patch.dict(os.environ, {"DISPLAY": ""}, clear=False), \
             patch.dict(os.environ, {}, clear=False), \
             patch.object(sys, "platform", "linux-amd64", create=True), \
             patch("matplotlib.use", side_effect=fake_use):
            vizu.ensure_backend(show=True)
        self.assertEqual(calls.get("backend"), "Agg")
        self.assertTrue(calls.get("force"))

    def test_warm_start_matplotlib_smoke(self):
        # Should not raise; it swallows exceptions internally
        vizu.warm_start_matplotlib()

if __name__ == "__main__":
    unittest.main()
