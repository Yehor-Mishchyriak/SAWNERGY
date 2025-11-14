from __future__ import annotations

# third-pary
import numpy as np
# built-in
from pathlib import Path
from typing import Literal
from concurrent.futures import ProcessPoolExecutor
import logging
import os
# local
from . import walker_util
from .. import sawnergy_util

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class Walker:
    """Random-walk sampler over time-indexed **transition** matrices.

    Loads per-timestamp stacks of attractive/repulsive **transition** matrices
    (shape ``(T, N, N)``) previously written by the RIN builder, exposes
    sampling for random walks (RW) and self-avoiding walks (SAW), and can
    optionally advance a *time* coordinate using cosine-similarity between
    transition slices (time-aware walks).

    Matrices live in OS shared memory via :class:`walker_util.SharedNDArray`,
    so multiple processes can read the same buffers zero-copy. Each `Walker`
    instance owns a dedicated :class:`numpy.random.Generator` seeded from a
    master seed for reproducibility.
    """

    def __init__(self,
                 RIN_path: str | Path,
                 *,
                 seed: int | None = None) -> None:
        """Initialize shared matrices and RNG.

        Data source:
            Transition dataset names are resolved from the archive attributes
            ``'attractive_transitions_name'`` and ``'repulsive_transitions_name'``.
            If either attribute is ``None``, that channel is unavailable.

        Args:
            RIN_path: Path to an ``ArrayStorage`` archive (.zip) containing
                **transition** matrices written by the builder.
            seed: Optional master seed for this instance's RNG. If ``None``,
                a random 32-bit seed is chosen.

        Raises:
            ValueError: If no matrices are found or arrays are not rank-3.
            RuntimeError: If attractive/repulsive shapes differ or matrices
                are not square along the last two axes.
        """
        _logger.info("Initializing Walker from %s", RIN_path)

        # Load numpy arrays from read-only storage
        with sawnergy_util.ArrayStorage(RIN_path, mode="r") as storage:
            attr_name = storage.get_attr("attractive_transitions_name")
            repuls_name = storage.get_attr("repulsive_transitions_name")
            attr_matrices  : np.ndarray | None = (
                storage.read(attr_name, slice(None)) if attr_name is not None else None
            )
            repuls_matrices: np.ndarray | None = (
                storage.read(repuls_name, slice(None)) if repuls_name is not None else None
            )

        _logger.debug(
            "Loaded matrices | attr: shape=%s dtype=%s | repuls: shape=%s dtype=%s",
            getattr(attr_matrices, "shape", None), getattr(attr_matrices, "dtype", None),
            getattr(repuls_matrices, "shape", None), getattr(repuls_matrices, "dtype", None),
        )

        # Shape & consistency checks (expect (T, N, N))
        if (attr_matrices is not None) and (repuls_matrices is not None):
            if attr_matrices.ndim != 3 or repuls_matrices.ndim != 3:
                _logger.error(
                    "Bad ranks: attr.ndim=%s repuls.ndim=%s; expected both 3",
                    getattr(attr_matrices, "ndim", None),
                    getattr(repuls_matrices, "ndim", None),
                )
                raise ValueError(
                    f"Expected (T,N,N) arrays; got {getattr(attr_matrices, 'shape', None)} "
                    f"and {getattr(repuls_matrices, 'shape', None)}"
                )
            if attr_matrices.shape != repuls_matrices.shape:
                _logger.error("Shape mismatch: attr=%s repuls=%s",
                              attr_matrices.shape, repuls_matrices.shape)
                raise RuntimeError(
                    f"ATTR/REPULS shapes must match exactly; got {attr_matrices.shape} vs {repuls_matrices.shape}"
                )
            T, N1, N2 = attr_matrices.shape
        elif attr_matrices is not None:
            if attr_matrices.ndim != 3:
                raise ValueError(f"Expected (T,N,N); got {attr_matrices.shape}")
            T, N1, N2 = attr_matrices.shape
        elif repuls_matrices is not None:
            if repuls_matrices.ndim != 3:
                raise ValueError(f"Expected (T,N,N); got {repuls_matrices.shape}")
            T, N1, N2 = repuls_matrices.shape
        else:
            _logger.error("No transition matrices detected in %s", RIN_path)
            raise ValueError("No transition matrices detected.")

        if N1 != N2:
            _logger.error("Non-square matrices along last two dims: (%s, %s)", N1, N2)
            raise RuntimeError(
                f"Transition matrices must be square along last two dims; got ({N1}, {N2})"
            )

        _logger.info("Transition stack OK: T=%d, N=%d", T, N1)

        # SHARED MEMORY ELEMENTS (read-only default views; fancy indexing via .array)
        self.attr_matrices: walker_util.SharedNDArray | None = (
            walker_util.SharedNDArray.create(
                shape=attr_matrices.shape,
                dtype=attr_matrices.dtype,
                from_array=attr_matrices,
            ) if attr_matrices is not None else None
        )
        self.repuls_matrices: walker_util.SharedNDArray | None = (
            walker_util.SharedNDArray.create(
                shape=repuls_matrices.shape,
                dtype=repuls_matrices.dtype,
                from_array=repuls_matrices,
            ) if repuls_matrices is not None else None
        )

        self._attr_owner_pid   = os.getpid() if self.attr_matrices  is not None else None
        self._repuls_owner_pid = os.getpid() if self.repuls_matrices is not None else None

        _logger.debug(
            "SharedNDArray created | attr name=%r; repuls name=%r",
            getattr(self.attr_matrices, "name", None),
            getattr(self.repuls_matrices, "name", None),
        )

        # AUXILIARY NETWORK INFORMATION
        self.time_stamp_count = T
        self.node_count       = N1

        # NETWORK ELEMENT
        self.nodes       = np.arange(0, self.node_count, 1, np.intp)
        self.time_stamps = np.arange(0, self.time_stamp_count, 1, np.intp)
        _logger.debug(
            "Index arrays built: nodes=%d, time_stamps=%d",
            self.nodes.size, self.time_stamps.size
        )

        # INTERNAL
        self._memory_cleaned_up: bool = False
        self._seed = np.random.randint(0, 2**32 - 1) if seed is None else int(seed)
        self.rng = np.random.default_rng(self._seed)
        _logger.info("RNG initialized (master seed=%d)", self._seed)

    # explicit resource cleanup
    def close(self) -> None:
        """Release shared-memory resources used by this Walker.

        This method:
        - Closes local handles to the shared-memory backed arrays
        (`self.attr_matrices`, `self.repuls_matrices`) in **the current process**.
        - If the current process is the **creator** of a segment (its PID matches
        `_attr_owner_pid` / `_repuls_owner_pid`), it also **unlinks** that segment
        so the OS can reclaim it once all handles are closed.

        Behavior & guarantees
        ---------------------
        - **Idempotent:** safe to call multiple times; subsequent calls are no-ops.
        - **Multi-process aware:** non-creator processes only close their handles;
        creators close **and** unlink. This prevents `resource_tracker` “leaked
        shared_memory” warnings when using `ProcessPoolExecutor`/spawn.
        - **Best-effort unlink:** `FileNotFoundError` during unlink (already unlinked
        elsewhere) is swallowed.
        - Invoked automatically by the context manager (`__exit__`) and destructor
        (`__del__`), but it's fine to call explicitly.

        After calling `close()`, any operation that relies on the shared arrays may
        fail; treat the instance as finalized.

        Returns:
            None
        """
        if self._memory_cleaned_up:
            _logger.debug("close(): already cleaned up; returning")
            return
        _logger.debug("Closing Walker resources (pid=%s)", os.getpid())
        try:
            if self.attr_matrices is not None:
                self.attr_matrices.close()
            if self.repuls_matrices is not None:
                self.repuls_matrices.close()
            _logger.debug("SharedNDArray handles closed")

            # Unlink in whichever process actually CREATED the segment(s)
            try:
                if self.attr_matrices is not None and getattr(self, "_attr_owner_pid", None) == os.getpid():
                    self.attr_matrices.unlink()
            except FileNotFoundError:
                _logger.debug("attr SharedMemory already unlinked elsewhere")

            try:
                if self.repuls_matrices is not None and getattr(self, "_repuls_owner_pid", None) == os.getpid():
                    self.repuls_matrices.unlink()
            except FileNotFoundError:
                _logger.debug("repuls SharedMemory already unlinked elsewhere")

        finally:
            self._memory_cleaned_up = True
            _logger.debug("Cleanup complete")

    def __enter__(self):
        """Enter context manager scope."""
        _logger.debug("__enter__")
        return self

    def __exit__(self, exc_type, exc, tb):
        """Exit context manager scope and perform cleanup."""
        _logger.debug("__exit__(exc_type=%s)", getattr(exc_type, "__name__", exc_type))
        self.close()

    def __del__(self):
        """Best-effort destructor cleanup (exceptions suppressed)."""
        try:
            if not getattr(self, "_memory_cleaned_up", True):
                _logger.debug("__del__: best-effort close")
            self.close()
        except Exception as e:
            _logger.debug("__del__ suppressed exception: %r", e)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #                               PRIVATE
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        
    def _matrices_of_interaction_type(
        self, interaction_type: Literal["attr", "repuls"]
    ) -> walker_util.SharedNDArray:
        """Return the shared array wrapper for the requested interaction type.

        Args:
            interaction_type: Either ``"attr"`` or ``"repuls"``.

        Returns:
            SharedNDArray wrapper exposing the ``(T, N, N)`` stack.

        Raises:
            ValueError: If the channel is missing or the type is invalid.
        """
        _logger.debug("_matrices_of_interaction_type(%s)", interaction_type)
        if interaction_type == "attr":
            if self.attr_matrices is None:
                raise ValueError(
                    "Attractive transition matrices are not present in the RIN archive."
                )
            return self.attr_matrices
        if interaction_type == "repuls":
            if self.repuls_matrices is None:
                raise ValueError(
                    "Repulsive transition matrices are not present in the RIN archive."
                )
            return self.repuls_matrices
        _logger.error("interaction_type invalid: %r", interaction_type)
        raise ValueError("`interaction_type` must be 'attr' or 'repuls'.")

    def _extract_prob_vector(
        self,
        node: int,
        time_stamp: int,
        interaction_type: Literal["attr", "repuls"],
    ) -> np.ndarray:
        """Copy the transition row for ``node`` at ``time_stamp``.

        Returns:
            ``(N,)`` float array with transition probabilities/weights.
        """
        _logger.debug("_extract_prob_vector(node=%d, t=%d, type=%s)",
                      node, time_stamp, interaction_type)
        matrix = self._matrices_of_interaction_type(interaction_type)[time_stamp]
        vec = matrix[node, :].copy()  # detach from shared buffer
        _logger.debug("prob vector extracted: shape=%s dtype=%s", vec.shape, vec.dtype)
        return vec

    def _step_node(
        self,
        node: int,
        interaction_type: Literal["attr", "repuls"],
        time_stamp: int = 0,
        avoid: np.typing.ArrayLike | None = None,
    ) -> tuple[int, np.ndarray | None]:
        """Sample next node given current node and optional avoidance set.

        Returns:
            ``(next_node, updated_avoid)`` where ``updated_avoid`` is ``None`` if
            avoidance is disabled.
        """
        _logger.debug(
            "_step_node(node=%d, t=%d, type=%s, avoid_len=%s)",
            node, time_stamp, interaction_type,
            None if avoid is None else np.asarray(avoid).size,
        )
        prob_dist = self._extract_prob_vector(node, time_stamp, interaction_type)

        if avoid is None:
            mass = float(np.sum(prob_dist))
            _logger.debug("_step_node: no-avoid branch; mass=%.6f", mass)
            if mass <= 0.0:
                _logger.error("_step_node: zero probability mass without avoidance")
                raise RuntimeError("No valid node transitions: zero probability mass.")
            probs = walker_util.l1_norm(prob_dist)
            return int(self.rng.choice(self.nodes, p=probs)), None

        to_avoid = np.asarray(avoid, dtype=np.intp)
        keep = np.setdiff1d(self.nodes, to_avoid, assume_unique=False)
        _logger.debug("_step_node: keep.size=%d (after removing %d avoids)",
                      keep.size, to_avoid.size)
        if keep.size == 0:
            _logger.error("_step_node: empty candidate set after avoidance")
            raise RuntimeError("No available node transitions (avoiding all nodes).")

        probs = walker_util.l1_norm(prob_dist[keep])
        mass = float(probs.sum())
        _logger.debug("_step_node: normalized mass=%.6f", mass)
        if mass <= 0.0:
            _logger.error("_step_node: zero probability mass after masking/normalization")
            raise RuntimeError("No valid node transitions: probability mass is zero.")

        next_node = int(self.rng.choice(keep, p=probs))
        _logger.debug("_step_node: chosen next_node=%d", next_node)
        to_avoid = np.append(to_avoid, next_node).astype(np.intp, copy=False)

        return next_node, to_avoid

    def _step_time(
        self,
        time_stamp: int,
        interaction_type: Literal["attr", "repuls"],
        stickiness: float,
        on_no_options: Literal["raise", "loop"],
        avoid: np.typing.ArrayLike | None,
    ) -> tuple[int, np.ndarray | None]:
        """Sample next time stamp given stickiness and similarity.

        Raises:
            ValueError: If ``stickiness`` not in ``[0,1]`` or ``on_no_options`` invalid.
            RuntimeError: If no candidates or zero probability mass.
        """
        _logger.debug(
            "_step_time(t=%d, type=%s, stickiness=%.3f, on_no_options=%s, avoid_len=%s)",
            time_stamp, interaction_type, stickiness, on_no_options,
            None if avoid is None else np.asarray(avoid).size,
        )
        if not (0.0 <= stickiness <= 1.0):
            _logger.error("stickiness out of range: %r", stickiness)
            raise ValueError("stickiness must be in [0,1]")
        
        to_avoid = np.array([], dtype=np.intp) if avoid is None else np.asarray(avoid, dtype=np.intp)

        # With probability `stickiness`, remain at the same time stamp
        r = float(self.rng.random())
        _logger.debug("_step_time: rand=%.6f vs stickiness=%.6f", r, float(stickiness))
        if r < float(stickiness):
            _logger.debug("_step_time: sticking at t=%d", time_stamp)
            return int(time_stamp), to_avoid

        # Exclude current time since we chose not to stick
        to_avoid = np.unique(np.append(to_avoid, time_stamp).astype(np.intp, copy=False))
        keep = np.setdiff1d(self.time_stamps, to_avoid, assume_unique=True)
        _logger.debug("_step_time: keep.size=%d (to_avoid.size=%d)", keep.size, to_avoid.size)

        matrices = self._matrices_of_interaction_type(interaction_type)
        current_matrix = matrices[time_stamp]  # axis-0 basic indexing returns a view

        if keep.size == 0:
            if on_no_options == "raise":
                _logger.error("_step_time: no available timestamps (avoid=%s)", np.unique(to_avoid))
                raise RuntimeError(f"No available time stamps (avoid={np.unique(to_avoid)})")
            if on_no_options == "loop":
                _logger.warning("_step_time: looping over all except current (t=%d)", time_stamp)
                to_avoid = np.array([time_stamp], dtype=np.intp)
                keep = self.time_stamps[self.time_stamps != time_stamp]
                if keep.size == 0:
                    _logger.error("_step_time: loop mode impossible (T==1)")
                    raise RuntimeError("No alternative time stamps available for loop mode (T==1).")
                matrices_stack = matrices.array[keep]  # fancy indexing on ndarray, not wrapper
            else:
                _logger.error("_step_time: invalid on_no_options=%r", on_no_options)
                raise ValueError("on_no_options must be 'raise' or 'loop'")
        else:
            matrices_stack = matrices.array[keep]  # fancy indexing on ndarray, not wrapper

        sims = walker_util.apply_on_axis0(matrices_stack, walker_util.cosine_similarity(current_matrix))
        probs = walker_util.l1_norm(sims)
        mass = float(probs.sum())
        _logger.debug("_step_time: candidates=%d, mass=%.6f", keep.size, mass)
        if mass <= 0.0:
            _logger.error(
                "_step_time: zero probability mass (t=%d, type=%s, candidates=%d)",
                time_stamp, interaction_type, keep.size
            )
            raise RuntimeError(
                "No valid time stamps to sample: probability mass is zero after masking/normalization."
            )

        next_time_stamp = int(self.rng.choice(keep, p=probs))
        _logger.debug("_step_time: chosen next_time_stamp=%d", next_time_stamp)
        return next_time_stamp, to_avoid
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #                                PUBLIC
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

    def walk(self,
             start_node: int | None,
             start_time_stamp: int | None,
             length: int,
             interaction_type: Literal["attr", "repuls"],
             self_avoid: bool,
             time_aware: bool = False,
             stickiness: float | None = None,
             on_no_options: Literal["raise", "loop"] | None = None) -> np.ndarray:
        """Generate one walk path.

        Indexing contract:
            Public API is **1-based** for both nodes and time stamps to match
            residue numbering in biomolecular contexts. Internally everything is
            0-based. The returned path is 1-based.

        Args:
            start_node: 1-based start node; if ``None``, sampled uniformly.
            start_time_stamp: 1-based start time; if ``None``, sampled uniformly.
            length: Number of transition **steps** to simulate.
            interaction_type: ``"attr"`` or ``"repuls"``.
            self_avoid: If ``True``, path will not revisit nodes within the same walk.
            time_aware: If ``True``, advance time with :meth:`_step_time` each step.
            stickiness: Required when ``time_aware=True``; probability of staying
                at the current time.
            on_no_options: Required when ``time_aware=True``; behavior when no
                time candidates are available (``"raise"`` or ``"loop"``).

        Returns:
            ``(length + 1,)`` array of dtype ``intp`` with **1-based** node indices.

        Raises:
            ValueError: Bad start indices (after 1-based→0-based) or missing
                time-aware parameters.
            RuntimeError: Propagated from step routines when no valid choices exist.
        """
        _logger.debug(
            "walk(start_node=%r, start_time_stamp=%r, length=%d, type=%s, self_avoid=%s, time_aware=%s)",
            start_node, start_time_stamp, length, interaction_type, self_avoid, time_aware
        )

        # 1-based external API preserved, so validate ranges after conversion
        if start_node is not None:
            node = int(start_node) - 1
            if not (0 <= node < self.node_count):
                _logger.error("start_node out of range after 1-based conversion: %r", start_node)
                raise ValueError(f"start_node out of range after 1-based conversion: {start_node}")
        else:
            node = int(self.rng.choice(self.nodes))

        if start_time_stamp is not None:
            time_stamp = int(start_time_stamp) - 1
            if not (0 <= time_stamp < self.time_stamp_count):
                _logger.error("start_time_stamp out of range after 1-based conversion: %r", start_time_stamp)
                raise ValueError(f"start_time_stamp out of range after 1-based conversion: {start_time_stamp}")
        else:
            time_stamp = int(self.rng.choice(self.time_stamps))

        _logger.debug("walk: initial node=%d, t=%d", node, time_stamp)

        nodes_to_avoid: np.ndarray | None = np.array([node], dtype=np.intp) if self_avoid else None
        time_stamps_to_avoid: np.ndarray | None = None

        pth = np.empty(length + 1, dtype=np.intp)
        pth[0] = node

        if time_aware and (stickiness is None or on_no_options is None):
            _logger.error("time_aware=True but stickiness/on_no_options missing")
            raise ValueError("time_aware=True requires both `stickiness` and `on_no_options`.")

        for step in range(1, length + 1):
            if self_avoid:
                node, nodes_to_avoid = self._step_node(node, interaction_type, time_stamp, nodes_to_avoid)
            else:
                node, _ = self._step_node(node, interaction_type, time_stamp, avoid=None)
            pth[step] = node

            if time_aware:
                time_stamp, time_stamps_to_avoid = self._step_time(
                    time_stamp, interaction_type, stickiness, on_no_options, time_stamps_to_avoid
                )
        
        pth += 1  # ensure 1-based indexing in the output

        _logger.debug("walk: finished path of len=%d", pth.size)
        return pth

    # deterministic per-batch worker: (start_nodes_batch, seedseq/int) -> stack of walks
    def _walk_batch_with_seed(self, work_item, num_walks_from_each: int, *args, **kwargs):
        """Worker: seed RNG and generate a batch of walks for a set of start nodes.

        Args:
            work_item: Tuple ``(start_nodes, seed_obj)`` where ``start_nodes`` is
                an iterable of **0-based** node indices and ``seed_obj`` is an
                ``np.random.SeedSequence`` or an ``int``.
            num_walks_from_each: Number of walks to generate per start node. If 0,
                returns an empty array with the correct width.

        Returns:
            ``(M, L+1)`` array of dtype ``uint16`` with 1-based node indices,
            where ``M = len(start_nodes) * num_walks_from_each`` and ``L`` is
            ``kwargs["length"]`` (defaults to 0 if absent). If no walks are
            generated, returns ``(0, L+1)``.
        """
        start_nodes, seed_obj = work_item
        _logger.debug(
            "_walk_batch_with_seed: batch_size=%d, walks_each=%d",
            np.asarray(start_nodes).size, int(num_walks_from_each)
        )
        self.rng = np.random.default_rng(seed_obj)  # SeedSequence or int OK
        start_nodes = np.asarray(start_nodes, dtype=np.intp)
        out = []
        for snode in start_nodes:
            for _ in range(int(num_walks_from_each)):
                out.append(self.walk(int(snode)+1, *args, **kwargs))  # 1-based API

        if not out:
            L = int(kwargs.get("length", 0))
            return np.empty((0, L + 1), dtype=np.uint16)

        arr = np.stack(out, axis=0).astype(np.uint16, copy=False)
        _logger.debug("_walk_batch_with_seed: produced walks shape=%s dtype=%s", arr.shape, arr.dtype)
        return arr

    def _walks_per_time(self,
                        work_items,
                        processor,
                        num_walks_from_each: int,
                        *,
                        walk_length: int,
                        interaction_type: Literal["attr", "repuls"],
                        self_avoid: bool,
                        time_aware: bool,
                        stickiness: float | None,
                        on_no_options: Literal["raise", "loop"] | None) -> np.ndarray:
        """
        Generate walks separately for each start time stamp and stack as (T, M, L+1).

        - If time_aware=False: each layer t contains walks constrained to time t.
        - If time_aware=True: each layer t contains walks that *start* at t and
          may traverse time via _step_time during the walk.
        """
        per_time: list[np.ndarray] = []
        for t in range(self.time_stamp_count):
            chunks = processor(
                work_items,
                self._walk_batch_with_seed,
                int(num_walks_from_each),
                start_time_stamp=int(t + 1),
                length=walk_length,
                interaction_type=interaction_type,
                self_avoid=self_avoid,
                time_aware=bool(time_aware),
                stickiness=stickiness,
                on_no_options=on_no_options,
            )
            if chunks:
                all_walks_2d = np.concatenate(chunks, axis=0).astype(np.uint16, copy=False)
            else:
                all_walks_2d = np.empty((0, walk_length + 1), dtype=np.uint16)
            per_time.append(all_walks_2d)
        arr_3d = (np.stack(per_time, axis=0)
                  if per_time else np.empty((self.time_stamp_count, 0, walk_length + 1), dtype=np.uint16))
        _logger.info("_walks_per_time: produced (T,M,L+1)=%s for type=%s, self_avoid=%s, time_aware=%s",
                     arr_3d.shape, interaction_type, self_avoid, time_aware)
        return arr_3d

    def sample_walks(self,
                     # walks
                     walk_length: int,
                     walks_per_node: int,
                     saw_frac: float = 0.0,
                     include_attractive: bool = True,
                     include_repulsive: bool = False,
                     # time aware params
                     time_aware: bool = False,
                     stickiness: float | None = None,
                     on_no_options: Literal["raise", "loop"] | None = None,
                     # output
                     output_path: str | Path | None = None,
                     *,
                     # computation
                     in_parallel: bool,
                     # storage
                     compression_level: int = 3,
                     num_walk_matrices_in_compressed_blocks: int | None = None
                     ) -> str:
        """Generate and persist random walks for **all** nodes.

        For each node, produces ``walks_per_node`` paths split between RW and
        SAW according to ``saw_frac``. Optionally enables time-aware stepping.

        Output layout:
            Walk arrays are written as **3-D** with shape ``(T, M, L+1)``, where:
                - ``T`` is the number of time stamps,
                - ``M`` is the total number of walks produced per layer (sum over requested nodes),
                - ``L+1`` is the path length including the start node.
            If ``time_aware=False``, each layer t contains walks constrained to time t.
            If ``time_aware=True``, each layer t contains walks that **start** at time t but may evolve in time.

        Results are chunk-written to a new compressed archive.

        Args:
            walk_length: Number of steps per walk (path length is ``walk_length+1``).
            walks_per_node: Total walks per node (integer).
            saw_frac: Fraction in ``[0,1]`` of per-node walks that are SAWs
                (remainder are RWs).
            include_attractive: If ``True``, generate walks on the attractive channel.
            include_repulsive: If ``True``, generate walks on the repulsive channel.
            time_aware: If ``True``, enable time evolution via :meth:`_step_time`.
            stickiness: Required when ``time_aware=True``; probability of staying
                at the current time step.
            on_no_options: Required when ``time_aware=True``; when no alternative
                time stamps are available, either ``"raise"`` or ``"loop"``.
            output_path: Destination (with or without ``.zip``). Defaults to
                ``WALKS_<timestamp>.zip`` in the current working directory.
            in_parallel: Use :class:`ProcessPoolExecutor` to parallelize over
                node batches (requires main-process guard).
            compression_level: Compression level for the output archive.
            num_walk_matrices_in_compressed_blocks: Max number of walk matrices
                per compressed chunk when writing. Defaults to number of batches.

        Returns:
            String path to the written archive.

        Raises:
            ValueError: If ``saw_frac`` is outside ``[0,1]``.
            RuntimeError: If run in parallel without a main-process guard, or when
                no valid transitions are available during stepping.
        """
        _logger.info(
            "sample_walks: L=%d, per_node=%d, saw_frac=%.3f, time_aware=%s, out=%s, "
            "parallel=%s, arrays_per_chunk=%s",
            walk_length, walks_per_node, saw_frac, time_aware, output_path, in_parallel,
            num_walk_matrices_in_compressed_blocks,
        )

        current_time = sawnergy_util.current_time()
        
        output_path = Path((output_path or (Path(os.getcwd()) /
                        f"WALKS_{current_time}"))).with_suffix(".zip")
        _logger.debug("Output archive path: %s", output_path)

        if not (0.0 <= saw_frac <= 1.0):
            _logger.error("saw_frac out of range: %r", saw_frac)
            raise ValueError("saw_frac must be in [0, 1]")

        # Deterministic integer split
        num_SAWs = int(round(walks_per_node * float(saw_frac)))
        num_RWs  = int(walks_per_node) - num_SAWs
        _logger.info("Per-node counts: SAWs=%d, RWs=%d", num_SAWs, num_RWs)

        num_workers = os.cpu_count() or 1
        batch_size_nodes = (num_workers if in_parallel else 1)
        _logger.debug("Workers=%d, batch_size_nodes=%d", num_workers, batch_size_nodes)

        if in_parallel and not sawnergy_util.is_main_process():
            _logger.error("Process-based parallelism requires main-process guard")
            raise RuntimeError(
                "Process-based parallelism requires running under `if __name__ == '__main__':`."
            )

        processor = sawnergy_util.elementwise_processor(
            in_parallel=in_parallel,
            Executor=ProcessPoolExecutor,
            max_workers=num_workers,
            capture_output=True
        )
        _logger.debug("elementwise_processor created (parallel=%s, workers=%d)", in_parallel, num_workers)

        # Pre-build node batches deterministically
        _logger.debug("Building node batches via sawnergy_util.batches_of (batch_size_nodes=%d)", batch_size_nodes)
        node_batches = list(sawnergy_util.batches_of(self.nodes, batch_size=batch_size_nodes))
        _logger.debug("Built %d node batches", len(node_batches))

        # Derive deterministic child seeds from master seed — stable per batch
        master_ss = np.random.SeedSequence(self._seed)
        child_seeds = master_ss.spawn(len(node_batches))
        work_items = list(zip(node_batches, child_seeds))
        _logger.debug("Prepared %d work_items with child seeds", len(work_items))

        num_walk_matrices_in_compressed_blocks = (
            num_walk_matrices_in_compressed_blocks or len(node_batches)
        )
        _logger.info("arrays_per_chunk resolved to: %d", num_walk_matrices_in_compressed_blocks)

        attractive_RWs_name  = "ATTRACTIVE_RWs"
        attractive_SAWs_name = "ATTRACTIVE_SAWs"
        repulsive_RWs_name   = "REPULSIVE_RWs"
        repulsive_SAWs_name  = "REPULSIVE_SAWs"

        with sawnergy_util.ArrayStorage.compress_and_cleanup(output_path, compression_level) as storage:
            if include_attractive:
                # --- ATTR RWs ---
                _logger.info("Generating ATTR RWs ...")

                attr_RWs_3d = self._walks_per_time(
                    work_items, processor, num_RWs,
                    walk_length=walk_length,
                    interaction_type="attr",
                    self_avoid=False,
                    time_aware=time_aware,
                    stickiness=stickiness,
                    on_no_options=on_no_options,
                )
                _logger.info("ATTR RWs (per-time): shape=%s", attr_RWs_3d.shape)
                storage.write(
                    attr_RWs_3d,
                    to_block_named=attractive_RWs_name,
                    arrays_per_chunk=num_walk_matrices_in_compressed_blocks
                )

                # --- ATTR SAWs ---
                _logger.info("Generating ATTR SAWs ...")
                attr_SAWs_3d = self._walks_per_time(
                    work_items, processor, num_SAWs,
                    walk_length=walk_length,
                    interaction_type="attr",
                    self_avoid=True,
                    time_aware=time_aware,
                    stickiness=stickiness,
                    on_no_options=on_no_options,
                )
                _logger.info("ATTR SAWs (per-time): shape=%s", attr_SAWs_3d.shape)
                storage.write(
                    attr_SAWs_3d,
                    to_block_named=attractive_SAWs_name,
                    arrays_per_chunk=num_walk_matrices_in_compressed_blocks
                )
            
            if include_repulsive:
                # --- REPULS RWs ---
                _logger.info("Generating REPULS RWs ...")
                repuls_RWs_3d = self._walks_per_time(
                    work_items, processor, num_RWs,
                    walk_length=walk_length,
                    interaction_type="repuls",
                    self_avoid=False,
                    time_aware=time_aware,
                    stickiness=stickiness,
                    on_no_options=on_no_options,
                )
                _logger.info("REPULS RWs (per-time): shape=%s", repuls_RWs_3d.shape)
                storage.write(
                    repuls_RWs_3d,
                    to_block_named=repulsive_RWs_name,
                    arrays_per_chunk=num_walk_matrices_in_compressed_blocks
                )

                # --- REPULS SAWs ---
                _logger.info("Generating REPULS SAWs ...")
                repuls_SAWs_3d = self._walks_per_time(
                    work_items, processor, num_SAWs,
                    walk_length=walk_length,
                    interaction_type="repuls",
                    self_avoid=True,
                    time_aware=time_aware,
                    stickiness=stickiness,
                    on_no_options=on_no_options,
                )
                _logger.info("REPULS SAWs (per-time): shape=%s", repuls_SAWs_3d.shape)
                storage.write(
                    repuls_SAWs_3d,
                    to_block_named=repulsive_SAWs_name,
                    arrays_per_chunk=num_walk_matrices_in_compressed_blocks
                )

            # useful metadata
            storage.add_attr("time_created", current_time)
            storage.add_attr("seed", int(self._seed))
            storage.add_attr("rng_scheme", "SeedSequence.spawn_per_batch_v1")
            storage.add_attr("num_workers", int(num_workers))
            storage.add_attr("in_parallel", bool(in_parallel))
            storage.add_attr("batch_size_nodes", int(batch_size_nodes))
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            storage.add_attr("num_RWs",  num_RWs)
            storage.add_attr("num_SAWs", num_SAWs)
            storage.add_attr("node_count", self.node_count)
            storage.add_attr("walk_length", walk_length)
            storage.add_attr("walks_per_node", walks_per_node)
            storage.add_attr("time_stamp_count", self.time_stamp_count)
            # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
            storage.add_attr("attractive_RWs_name", attractive_RWs_name if include_attractive else None)
            storage.add_attr("repulsive_RWs_name", repulsive_RWs_name if include_repulsive else None)
            storage.add_attr("attractive_SAWs_name", attractive_SAWs_name if include_attractive else None)
            storage.add_attr("repulsive_SAWs_name", repulsive_SAWs_name if include_repulsive else None)
            storage.add_attr("walks_layout", "time_leading_3d")  # (T, M, L+1) for all modes

            _logger.info("Wrote metadata")

        _logger.info("sample_walks complete -> %s", str(output_path))
        return str(output_path)


__all__ = [
    "Walker"
]

if __name__ == "__main__":
    pass
