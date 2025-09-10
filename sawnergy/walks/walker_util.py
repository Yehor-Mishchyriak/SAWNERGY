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
    """NumPy-facing wrapper over a raw :class:`multiprocessing.shared_memory.SharedMemory`.

    This class does **not** own any data itself; it wraps an OS-level shared
    memory segment and exposes it as a NumPy array via zero-copy views
    (shape/dtype provided by the caller). The underlying buffer is just a
    flat byte block; dimensionality and strides come from the views you
    construct.

    Usage model:
      - Create a segment in the parent with :meth:`create`, optionally seeding
        from an existing array (copied once, C-contiguous).
      - Pass ``(name, shape, dtype)`` to workers and attach with :meth:`attach`.
      - Obtain a view with :py:meth:`view` or the :py:attr:`array` property.
        Views are read-only by default unless ``default_readonly=False`` or
        ``view(readonly=False)`` is requested.
      - Every process that opened the segment must call :meth:`close`.
        Exactly one process should call :meth:`unlink` after all others have
        closed to destroy the OS resource.

    Indexing:
      - ``__getitem__`` strictly supports **axis-0** basic indexing
        (``None``, ``slice``, or ``int``). This guarantees **no-copy** views.
        Fancy indexing (index arrays/boolean masks) is intentionally disallowed.
      - For 1D arrays, ``int`` indexing would yield a NumPy scalar (not a view),
        so it is rejected; use ``slice(i, i+1)`` for a one-row view instead.

    Concurrency:
      - Multiple readers are safe by design.
      - If multiple writers may overlap, synchronize externally (e.g., a
        :class:`multiprocessing.Lock`). The class does not implement locking.

    Notes:
      - The writeability flag is **per-view**. Marking one view read-only does
        not prevent other processes (or other views) from writing.
      - Shape/dtype are trusted by :meth:`attach`—they must match what was used
        at creation time; no runtime validation is performed here.
    """

    def __init__(self,
                shm: shared_memory.SharedMemory,
                shape: tuple[int, ...],
                dtype: np.dtype,
                *,
                default_readonly: bool = True):
        """Construct a wrapper over an existing shared memory handle.

        Args:
            shm: An open :class:`SharedMemory` handle (already created/attached).
            shape: Target array shape used for all views into this buffer.
            dtype: Target NumPy dtype used for all views into this buffer.
            default_readonly: If ``True``, views returned by :py:attr:`array`
                are marked read-only; override per-call via :py:meth:`view`.

        Remarks:
            This constructor does not allocate memory; it only stores metadata.
            Use :meth:`create` to allocate a new segment, or :meth:`attach`
            to connect to an existing one by name.
        """
        self.shm   = shm
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        self._default_readonly = default_readonly
        _logger.debug(
            "SharedNDArray.__init__(name=%r, shape=%s, dtype=%s, default_readonly=%s)",
            getattr(self.shm, "name", None), self.shape, self.dtype, self._default_readonly
        )

    def __len__(self) -> int:
        """Return the size of axis 0 (NumPy semantics).

        Returns:
            The number of elements along the first dimension.

        Raises:
            TypeError: If the wrapped array is 0-D (unsized).
        """
        if len(self.shape) == 0:
            _logger.error("len() called on 0-D array (shape=%s)", self.shape)
            raise TypeError("len() of unsized object")
        length = self.shape[0]
        _logger.debug("__len__ -> %d", length)
        return length

    def __repr__(self):
        """Debug-friendly representation showing name/shape/dtype."""
        return f"SharedNDArray(name={self.name!r}, shape={self.shape}, dtype={self.dtype})"

    def __getitem__(self, ids: int | slice | None = None):
        """Axis-0 only, no-copy guaranteed.
        - None      -> full view
        - slice     -> view
        - int       -> view (requires ndim >= 2); for 1D, use slice(i, i+1)
        """
        _logger.debug("__getitem__(ids=%r)", ids)
        arr = self.array 
        if ids is None:
            _logger.debug("__getitem__: returning full view")
            return arr
        if isinstance(ids, slice):
            _logger.debug("__getitem__: slice=%s", ids)
            return arr[ids, ...]
        if isinstance(ids, int):
            if arr.ndim == 1:
                _logger.error(
                    "__getitem__: 1D int indexing requested (idx=%r) -> would copy; raising",
                    ids
                )
                raise TypeError(
                    "No-copy view for 1D int indexing is impossible. "
                    "Use slice(i, i+1) to get a 1-row view."
                )
            _logger.debug("__getitem__: int=%d", ids)
            return arr[ids, ...]
        _logger.error("__getitem__: unsupported key type %s", type(ids).__name__)
        raise TypeError("Only axis-0 int/slice/None are allowed for no-copy access.")

    @classmethod
    def attach(cls, name: str, shape, dtype):
        """Attach to an existing shared memory segment by name.

        Args:
            name: System-wide shared memory name (as returned by :py:attr:`name`).
            shape: Shape to interpret the buffer with (must match creator).
            dtype: Dtype to interpret the buffer with (must match creator).

        Returns:
            A :class:`SharedNDArray` bound to the named segment.

        Raises:
            FileNotFoundError: If no segment with ``name`` exists.
            PermissionError: If the segment exists but cannot be opened.

        Notes:
            This method trusts ``shape`` and ``dtype``; it does not verify that
            they match the original settings. Passing inconsistent metadata
            results in undefined views.
        """
        _logger.debug("SharedNDArray.attach(name=%r, shape=%s, dtype=%s)", name, shape, np.dtype(dtype))
        shm = shared_memory.SharedMemory(name=name, create=False)
        obj = cls(shm, shape, dtype)
        _logger.debug("Attached to shared memory: name=%r", obj.name)
        return obj

    @classmethod
    def create(cls, shape, dtype, *, from_array=None, name: str | None = None):
        """Create a new shared memory segment and wrap it.

        The allocated buffer is sized exactly as ``prod(shape) * dtype.itemsize``.
        If ``from_array`` is provided, its contents are copied into the buffer
        after being coerced to a C-contiguous array of ``dtype``. Otherwise the
        buffer is zero-initialized.

        Args:
            shape: Desired array shape.
            dtype: Desired NumPy dtype.
            from_array: Optional source array to seed the buffer. Must match
                ``shape`` after coercion to ``dtype``; copied as C-contiguous.
            name: Optional OS-visible name for the segment. If omitted, a unique
                name is generated.

        Returns:
            A :class:`SharedNDArray` bound to the newly created segment.

        Raises:
            ValueError: If ``from_array`` shape does not match ``shape`` after
                dtype coercion.
        """
        dtype = np.dtype(dtype)
        nbytes = int(np.prod(shape)) * dtype.itemsize
        _logger.debug("SharedNDArray.create(shape=%s, dtype=%s, name=%r, nbytes=%d)", shape, dtype, name, nbytes)
        shm = shared_memory.SharedMemory(create=True, size=nbytes, name=name)
        obj = cls(shm, shape, dtype)

        view = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
        if from_array is not None:
            src = np.ascontiguousarray(from_array, dtype=dtype)
            if src.shape != tuple(shape):
                _logger.error("create: source shape %s does not match %s", src.shape, shape)
                raise ValueError(f"shape mismatch: {src.shape} vs {shape}")
            view[...] = src
            _logger.debug("create: seeded from array (shape=%s, dtype=%s)", src.shape, src.dtype)
        else:
            view.fill(0)
            _logger.debug("create: zero-initialized buffer")
        _logger.debug("create: created shared segment name=%r", obj.name)
        return obj

    def close(self) -> None:
        """Detach this process from the shared memory segment.

        Call this in **every** process that opened/attached the segment.
        After closing, any existing views into the buffer must **not** be used
        unless you first copy them (e.g., ``np.array(view, copy=True)``).
        """
        _logger.debug("close(): name=%r", self.name)
        self.shm.close()

    def unlink(self) -> None:
        """Destroy the shared memory segment (OS resource).

        Call exactly **once** globally after all participating processes have
        called :meth:`close`. After unlinking, the ``name`` may be reused by
        the OS for new segments.
        """
        _logger.debug("unlink(): name=%r", self.name)
        self.shm.unlink()

    def view(self, *, readonly: bool | None = None) -> np.ndarray: # if readonly is False, arr is mutable
        """Return a zero-copy NumPy view over the shared buffer.

        Args:
            readonly: If ``True``, the returned view is marked read-only.
                If ``False``, the view is writable. If ``None`` (default),
                the behavior follows ``self._default_readonly``.

        Returns:
            A NumPy ndarray that directly references the shared bytes using
            the stored ``shape`` and ``dtype``.

        Notes:
            - The writeability flag is **per-view**; it does not affect other
              views or other processes.
            - Basic slicing of the returned array yields further views that
              inherit the writeability flag; fancy indexing creates copies.
        """
        arr = np.ndarray(self.shape, dtype=self.dtype, buffer=self.shm.buf)
        ro = self._default_readonly if readonly is None else readonly
        _logger.debug("view(readonly=%r) -> resolved_readonly=%r", readonly, ro)
        if ro:
            arr.flags.writeable = False
        return arr

    @property
    def name(self) -> str:
        """System-wide name of the underlying shared memory segment."""
        return self.shm.name

    @property
    def array(self) -> np.ndarray:
        """Default zero-copy view honoring ``default_readonly``."""
        _logger.debug("array property accessed (default_readonly=%r)", self._default_readonly)
        return self.view(readonly=self._default_readonly)

# *----------------------------------------------------*
#                       FUNCTIONS
# *----------------------------------------------------*

def l1_norm(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    s = float(np.sum(X))
    if not np.isfinite(s) or s <= 0.0:
        return np.zeros_like(X)
    return X / s

def apply_on_axis0(X: np.ndarray, func):
    X = np.asarray(X)
    out0 = func(X[0])
    # the 0th axis has to have as many dims as the X array has along the 0th axis;
    # as for the other axes, they coincide in dimensionality with the output of func
    out = np.empty((X.shape[0],) + np.shape(out0), dtype=np.asarray(out0).dtype)
    out[0] = out0
    for i in range(1, X.shape[0]):
        out[i] = func(X[i])
    return out

def cosine_similarity(A: np.ndarray, eps: float = 1e-12):

    def inner(B: np.ndarray):
        nonlocal A
        nonlocal eps

        A = np.asarray(A)
        B = np.asarray(B)
        if A.shape != B.shape:
            raise ValueError(f"shapes must match, got {A.shape} vs {B.shape}")

        a = A.ravel()
        b = B.ravel()

        denom = np.linalg.norm(a) * np.linalg.norm(b)
        if denom < eps:
            return 0.0
        return (float(a @ b / denom) + 1) / 2 # translate from [-1, 1] to [0, 2] to [0, 1]

    return inner


if __name__ == "__main__":
    pass
