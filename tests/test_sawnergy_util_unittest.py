import unittest
import tempfile
import warnings
import numpy as np
from pathlib import Path
from tests.util_import import load_module

# Load into a pseudo-package where it's normally used
saw = load_module("sawnergy_util", "sawnergy_util.py")

class TestArrayStorage(unittest.TestCase):
    def test_write_and_read_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "store"
            with saw.ArrayStorage(p, mode="a") as st:
                arrs = [np.arange(6, dtype=np.int16).reshape(2, 3) for _ in range(3)]
                with warnings.catch_warnings(record=True) as w:
                    warnings.simplefilter("always")
                    st.write(arrs, to_block_named="BLOCK1")
                    self.assertTrue(any("Defaulting to 10" in str(x.message) for x in w))

                data_all = st.read("BLOCK1")
                self.assertEqual(data_all.shape, (3, 2, 3))

                sl = st.read("BLOCK1", slice(1, 3))
                self.assertEqual(sl.shape, (2, 2, 3))

                t = st.read("BLOCK1", (2, 0))
                self.assertEqual(t.shape, (2, 2, 3))

                one = st.read("BLOCK1", 1)
                self.assertEqual(one.shape, (2, 3))

    def test_chunk_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "store"
            with saw.ArrayStorage(p, mode="a") as st:
                arrs = [np.zeros((2, 2), dtype=np.float32) for _ in range(4)]
                st.write(arrs, to_block_named="B", arrays_per_chunk=4)
                with self.assertRaises(RuntimeError):
                    st.write([np.zeros((2, 2), dtype=np.float32)], to_block_named="B", arrays_per_chunk=2)

    def test_dtype_or_shape_mismatch_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "store"
            with saw.ArrayStorage(p, mode="a") as st:
                a0 = [np.zeros((2, 2), dtype=np.float32) for _ in range(2)]
                st.write(a0, to_block_named="B2")
                with self.assertRaises(ValueError):
                    st.write([np.zeros((3, 2), dtype=np.float32)], to_block_named="B2")
                with self.assertRaises(ValueError):
                    st.write([np.zeros((2, 2), dtype=np.int16)], to_block_named="B2")

    def test_missing_block_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "store"
            with saw.ArrayStorage(p, mode="a") as st:
                with self.assertRaises(KeyError):
                    st.read("NOPE")

    def test_zipstore_requires_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            zip_path = Path(tmp) / "store.zip"
            zip_path.write_bytes(b"")  # minimal placeholder
            with self.assertRaises(ValueError):
                saw.ArrayStorage(zip_path, mode="a")

if __name__ == "__main__":
    unittest.main()
