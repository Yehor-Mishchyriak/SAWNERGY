import unittest
import tempfile
import numpy as np
from pathlib import Path
from tests.util_import import ensure_package, load_module, load_core_sawnergy

# Prepare pseudo package and core deps
load_core_sawnergy()
walker = load_module("sawnergy.walker", "walker.py")
saw = load_module("sawnergy.sawnergy_util", "sawnergy_util.py")  # already loaded, idempotent
wu  = load_module("sawnergy.walker_util", "walker_util.py")      # already loaded, idempotent

class TestWalker(unittest.TestCase):
    def test_init_and_basic_helpers(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "rin_store"
            # Build a tiny transitions stack
            T, N = 3, 5
            attr = [(np.abs(np.random.randn(N, N)).astype(np.float32)) for _ in range(T)]
            repu = [(np.abs(np.random.randn(N, N)).astype(np.float32)) for _ in range(T)]

            with saw.ArrayStorage(base, mode="a") as st:
                st.write(attr, to_block_named="ATTRACTIVE_transitions", arrays_per_chunk=T)
                st.write(repu, to_block_named="REPULSIVE_transitions", arrays_per_chunk=T)

            # Instantiate
            W = walker.Walker(base, seed=123)
            try:
                self.assertEqual(W.node_count, N)
                self.assertEqual(W.time_stamp_count, T)

                # Verify vector extraction shape
                v = W._extract_prob_vector(node=0, time_stamp=0, interaction_type="attr")
                self.assertEqual(v.shape, (N,))

                # Invalid interaction type
                with self.assertRaises(ValueError):
                    _ = W._matrices_of_interaction_type("nope")
            finally:
                W.close()  # idempotent by design

if __name__ == "__main__":
    unittest.main()
