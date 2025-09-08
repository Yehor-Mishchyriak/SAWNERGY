from multiprocessing import shared_memory
import numpy as np
import logging

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class SharedNDArray:

    def __init__(self,
                shm: shared_memory.SharedMemory,
                shape: tuple[int, ...],
                dtype: np.dtype,
                *,
                default_readonly: bool = True):
        self.shm   = shm
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)

        self._default_readonly = default_readonly

    def __len__(self) -> int:
        if len(self.shape) == 0:
            raise TypeError("len() of unsized object")
        return self.shape[0]

    def __repr__(self):
        return f"SharedNDArray(name={self.name!r}, shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, ids: int | slice | None = None):
        """Axis-0 only, no-copy guaranteed.
        - None      -> full view
        - slice     -> view
        - int       -> view (requires ndim >= 2); for 1D, use slice(i, i+1)
        """
        arr = self.array 
        if ids is None:
            return arr
        if isinstance(ids, slice):
            return arr[ids, ...]
        if isinstance(ids, int):
            if arr.ndim == 1:
                raise TypeError(
                    "No-copy view for 1D int indexing is impossible. "
                    "Use slice(i, i+1) to get a 1-row view."
                )
            return arr[ids, ...]
        raise TypeError("Only axis-0 int/slice/None are allowed for no-copy access.")

    @classmethod
    def attach(cls, name: str, shape, dtype):
        shm = shared_memory.SharedMemory(name=name, create=False)
        return cls(shm, shape, dtype)

    @classmethod
    def create(cls, shape, dtype, *, from_array=None, name: str | None = None):
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        shm = shared_memory.SharedMemory(create=True, size=nbytes, name=name)
        obj = cls(shm, shape, dtype)

        view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        if from_array is not None:
            src = np.ascontiguousarray(from_array, dtype=dtype)
            if src.shape != tuple(shape):
                raise ValueError(f"shape mismatch: {src.shape} vs {shape}")
            view[...] = src
        else:
            view.fill(0)
        return obj

    def close(self) -> None:
        self.shm.close()

    def unlink(self) -> None:
        self.shm.unlink()

    def view(self, *, readonly: bool | None = None) -> np.ndarray: # if readonly is False, arr is mutable
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        ro = self._default_readonly if readonly is None else readonly
        if ro: arr.flags.writeable = False
        return arr

    @property
    def name(self) -> str:
        return self.shm.name

    @property
    def array(self) -> np.ndarray:
        return self.view(readonly=self._default_readonly)


if __name__ == "__main__":
    pass
