import unittest
import numpy as np
from tests.util_import import load_module

wu = load_module("walker_util", "walker_util.py")

class TestSharedNDArray(unittest.TestCase):
    def test_create_view_and_modify(self):
        data = np.arange(12, dtype=np.int32).reshape(3, 4)
        obj = wu.SharedNDArray.create(shape=data.shape, dtype=data.dtype, from_array=data)
        try:
            r0 = obj.array
            self.assertFalse(r0.flags.writeable)

            w = obj.view(readonly=False)
            w[0, 0] = 999
            r1 = obj.array
            self.assertEqual(r1[0, 0], 999)

            v = obj[slice(1, 3)]
            self.assertEqual(v.shape, (2, 4))
        finally:
            obj.close()
            obj.unlink()

    def test_len_and_repr(self):
        obj = wu.SharedNDArray.create(shape=(7, 2), dtype=np.float32)
        try:
            self.assertEqual(len(obj), 7)
            text = repr(obj)
            self.assertIn("SharedNDArray", text)
            self.assertIn("shape=(7, 2)", text)
        finally:
            obj.close()
            obj.unlink()

    def test_1d_int_index_rejected(self):
        obj = wu.SharedNDArray.create(shape=(5,), dtype=np.int16)
        try:
            with self.assertRaises(TypeError):
                _ = obj[0]
        finally:
            obj.close()
            obj.unlink()

    def test_attach_roundtrip(self):
        data = np.arange(6, dtype=np.int16).reshape(2, 3)
        obj = wu.SharedNDArray.create(shape=data.shape, dtype=data.dtype, from_array=data)
        try:
            name = obj.name
            attach = wu.SharedNDArray.attach(name=name, shape=data.shape, dtype=data.dtype)
            try:
                np.testing.assert_array_equal(attach.array, obj.array)
            finally:
                attach.close()
        finally:
            obj.close()
            obj.unlink()

if __name__ == "__main__":
    unittest.main()
