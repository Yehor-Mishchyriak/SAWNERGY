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

    def __init__(self,
                 RIN_path: str | Path,
                 *,
                 attr_data_name: str = "ATTRACTIVE_transitions",
                 repuls_data_name: str = "REPULSIVE_transitions",
                 seed: int | None = None) -> None:
        _logger.info("Initializing Walker from %s (attr=%s, repuls=%s)", RIN_path, attr_data_name, repuls_data_name)

        # load numpy arrays from read-only storage
        with sawnergy_util.ArrayStorage(RIN_path, mode="r") as storage:
            attr_matrices  : np.ndarray = storage.read(attr_data_name, slice(None))
            repuls_matrices: np.ndarray = storage.read(repuls_data_name, slice(None))

        _logger.debug("Loaded arrays: attr shape=%s dtype=%s; repuls shape=%s dtype=%s",
                      getattr(attr_matrices, "shape", None), getattr(attr_matrices, "dtype", None),
                      getattr(repuls_matrices, "shape", None), getattr(repuls_matrices, "dtype", None))

        # shape & consistency checks (expect (T, N, N))
        if attr_matrices.ndim != 3 or repuls_matrices.ndim != 3:
            _logger.error("Bad ranks: attr.ndim=%s repuls.ndim=%s; expected both 3", attr_matrices.ndim, repuls_matrices.ndim)
            raise ValueError(f"Expected (T,N,N) arrays; got {attr_matrices.shape} and {repuls_matrices.shape}")
        if attr_matrices.shape != repuls_matrices.shape:
            _logger.error("Shape mismatch: attr=%s repuls=%s", attr_matrices.shape, repuls_matrices.shape)
            raise RuntimeError(f"ATTR/REPULS shapes must match exactly; got {attr_matrices.shape} vs {repuls_matrices.shape}")
        T, N1, N2 = attr_matrices.shape
        if N1 != N2:
            _logger.error("Non-square matrices along last two dims: (%s, %s)", N1, N2)
            raise RuntimeError(f"Transition matrices must be square along last two dims; got ({N1}, {N2})")
        _logger.info("Transition stack OK: T=%d, N=%d", T, N1)

        # SHARED MEMORY ELEMENTS (read-only default views; fancy indexing via .array)
        self.attr_matrices   = walker_util.SharedNDArray.create(
            shape=attr_matrices.shape,
            dtype=attr_matrices.dtype,
            from_array=attr_matrices
        )
        self.repuls_matrices = walker_util.SharedNDArray.create(
            shape=repuls_matrices.shape,
            dtype=repuls_matrices.dtype,
            from_array=repuls_matrices
        )
        _logger.debug("SharedNDArray created: attr name=%r; repuls name=%r",
                      getattr(self.attr_matrices, "name", None), getattr(self.repuls_matrices, "name", None))

        # AUXILIARY NETWORK INFORMATION
        self.time_stamp_count = T
        self.node_count       = N1

        # NETWORK ELEMENT
        self.nodes       = np.arange(0, self.node_count, 1, np.intp)
        self.time_stamps = np.arange(0, self.time_stamp_count, 1, np.intp)
        _logger.debug("Index arrays built: nodes=%d, time_stamps=%d", self.nodes.size, self.time_stamps.size)

        # INTERNAL
        self._memory_cleaned_up: bool = False
        self._seed = np.random.randint(0, 2**32 - 1) if seed is None else int(seed)
        self.rng = np.random.default_rng(self._seed)
        _logger.info("RNG initialized (master seed=%d)", self._seed)

    # explicit resource cleanup
    def close(self, *, unlink: bool = True) -> None:
        if self._memory_cleaned_up:
            _logger.debug("close(): already cleaned up; unlink=%s", unlink)
            return
        _logger.info("Closing Walker (unlink=%s)", unlink)
        try:
            self.attr_matrices.close()
            self.repuls_matrices.close()
            _logger.debug("SharedNDArray handles closed")
            if unlink:
                _logger.debug("Unlinking shared memory segments")
                self.attr_matrices.unlink()
                self.repuls_matrices.unlink()
        finally:
            self._memory_cleaned_up = True
            _logger.info("Cleanup complete")

    def __enter__(self):
        _logger.debug("__enter__")
        return self

    def __exit__(self, exc_type, exc, tb):
        _logger.debug("__exit__(exc_type=%s)", getattr(exc_type, "__name__", exc_type))
        self.close()

    def __del__(self):
        try:
            if not getattr(self, "_memory_cleaned_up", True):
                _logger.debug("__del__: best-effort close")
            self.close()
        except Exception as e:
            _logger.debug("__del__ suppressed exception: %r", e)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #                               PRIVATE
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        
    def _matrices_of_interaction_type(self, interaction_type: Literal["attr", "repuls"]):
        _logger.debug("_matrices_of_interaction_type(%s)", interaction_type)
        if interaction_type == "attr":
            return self.attr_matrices
        elif interaction_type == "repuls":
            return self.repuls_matrices
        else:
            _logger.error("interaction_type invalid: %r", interaction_type)
            raise ValueError(f"`interaction_type` is expected to be `attr` or `repuls`. Instead, given: {interaction_type}")

    def _extract_prob_vector(self,
                             node: int,
                             time_stamp: int,
                             interaction_type: Literal["attr", "repuls"]):
        _logger.debug("_extract_prob_vector(node=%d, t=%d, type=%s)", node, time_stamp, interaction_type)
        matrix = self._matrices_of_interaction_type(interaction_type)[time_stamp]
        vec = matrix[node, :].copy()  # detach from shared buffer to avoid mutation
        _logger.debug("prob vector extracted: shape=%s dtype=%s", vec.shape, vec.dtype)
        return vec

    def _step_node(self,
                   node: int,
                   interaction_type: Literal["attr", "repuls"],
                   time_stamp: int = 0,
                   avoid: np.typing.ArrayLike | None = None
                   ) -> tuple[int, np.ndarray | None]:
        _logger.debug("_step_node(node=%d, t=%d, type=%s, avoid_len=%s)",
                      node, time_stamp, interaction_type,
                      None if avoid is None else np.asarray(avoid).size)
        prob_dist = self._extract_prob_vector(node, time_stamp, interaction_type)

        if avoid is None:
            _logger.debug("_step_node: no-avoid branch; sampling from all nodes (raw probs sum=%.6f)",
                          float(np.sum(prob_dist)))
            return int(self.rng.choice(self.nodes, p=prob_dist)), None

        to_avoid = np.asarray(avoid, dtype=np.intp)
        keep = np.setdiff1d(self.nodes, to_avoid, assume_unique=False)
        _logger.debug("_step_node: keep.size=%d (after removing %d avoids)", keep.size, to_avoid.size)
        if keep.size == 0:
            _logger.error("_step_node: empty candidate set after avoidance")
            raise RuntimeError("No available node transitions (avoiding all the nodes).")

        probs = walker_util.l1_norm(prob_dist[keep])
        _logger.debug("_step_node: normalized mass=%.6f", float(probs.sum()))
        if probs.sum() <= 0.0:
            _logger.error("_step_node: zero probability mass after masking/normalization")
            raise RuntimeError("No valid node transitions: probability mass is zero after masking/normalization.")

        next_node = int(self.rng.choice(keep, p=probs))
        _logger.debug("_step_node: chosen next_node=%d", next_node)
        to_avoid = np.append(to_avoid, next_node).astype(np.intp, copy=False)

        return next_node, to_avoid

    def _step_time(self,
                   time_stamp: int,
                   interaction_type: Literal["attr", "repuls"],
                   stickiness: float,
                   on_no_options: Literal["raise", "loop"],
                   avoid: np.typing.ArrayLike | None) -> tuple[int, np.ndarray | None]:
        _logger.debug("_step_time(t=%d, type=%s, stickiness=%.3f, on_no_options=%s, avoid_len=%s)",
                      time_stamp, interaction_type, stickiness, on_no_options,
                      None if avoid is None else np.asarray(avoid).size)
        if not (0.0 <= stickiness <= 1.0):
            _logger.error("stickiness out of range: %r", stickiness)
            raise ValueError("stickiness must be in [0,1]")
        
        to_avoid = np.array([], dtype=np.intp) if avoid is None else np.asarray(avoid, dtype=np.intp)

        # with probability = stickiness, remain at the same time stamp
        r = float(self.rng.random())
        _logger.debug("_step_time: rand=%.6f vs stickiness=%.6f", r, float(stickiness))
        if r < float(stickiness):
            _logger.debug("_step_time: sticking at t=%d", time_stamp)
            return int(time_stamp), to_avoid

        # exclude current time since we chose not to stick
        to_avoid = np.unique(np.append(to_avoid, time_stamp).astype(np.intp, copy=False)) # if intp already -- no new buffer
        keep = np.setdiff1d(self.time_stamps, to_avoid, assume_unique=True)
        _logger.debug("_step_time: keep.size=%d (to_avoid.size=%d)", keep.size, to_avoid.size)

        matrices = self._matrices_of_interaction_type(interaction_type)
        current_matrix = matrices[time_stamp]  # axis-0 basic indexing is ok on the sham wrapper

        if keep.size == 0:
            if on_no_options == "raise":
                _logger.error("_step_time: no available timestamps (avoid=%s)", np.unique(to_avoid))
                raise RuntimeError(f"No available time stamps (avoid={np.unique(to_avoid)})")
            elif on_no_options == "loop":
                _logger.warning("_step_time: looping over all except current (t=%d)", time_stamp)
                # avoid current; consider all other timestamps
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
            matrices_stack = matrices.array[keep]      # fancy indexing on ndarray, not wrapper

        sims = walker_util.apply_on_axis0(matrices_stack, walker_util.cosine_similarity(current_matrix))
        probs = walker_util.l1_norm(sims)
        mass = float(probs.sum())
        _logger.debug("_step_time: candidates=%d, mass=%.6f", keep.size, mass)
        if mass <= 0.0:
            _logger.error("_step_time: zero probability mass (t=%d, type=%s, candidates=%d)",
                          time_stamp, interaction_type, keep.size)
            raise RuntimeError(
                f"No valid time stamps to sample: probability mass is zero after masking/normalization. "
                f"time_stamp={time_stamp}, interaction_type={interaction_type}, candidates={len(keep)}."
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
        _logger.debug("walk(start_node=%r, start_time_stamp=%r, length=%d, type=%s, self_avoid=%s, time_aware=%s)",
                      start_node, start_time_stamp, length, interaction_type, self_avoid, time_aware)

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

        pth = np.array([node], dtype=np.intp)

        if time_aware and (stickiness is None or on_no_options is None):
            _logger.error("time_aware=True but stickiness/on_no_options missing")
            raise ValueError("time_aware=True requires both `stickiness` and `on_no_options` to be provided.")

        for _ in range(length):
            if self_avoid:
                node, nodes_to_avoid = self._step_node(node, interaction_type, time_stamp, nodes_to_avoid)
            else:
                node, _ = self._step_node(node, interaction_type, time_stamp, avoid=None)
            pth = np.append(pth, node).astype(np.intp, copy=False)

            if time_aware:
                time_stamp, time_stamps_to_avoid = self._step_time(
                    time_stamp, interaction_type, stickiness, on_no_options, time_stamps_to_avoid
                )

        _logger.debug("walk: finished path of len=%d", pth.size)
        return pth

    # deterministic per-batch worker: (start_nodes_batch, seedseq/int) -> stack of walks
    def _walk_batch_with_seed(self, work_item, num_walks_from_each: int, *args, **kwargs):
        start_nodes, seed_obj = work_item
        _logger.debug("_walk_batch_with_seed: batch_size=%d, walks_each=%d", np.asarray(start_nodes).size, int(num_walks_from_each))
        self.rng = np.random.default_rng(seed_obj)  # SeedSequence or int OK
        start_nodes = np.asarray(start_nodes, dtype=np.intp)
        out = []
        for snode in start_nodes:
            for _ in range(int(num_walks_from_each)):
                out.append(self.walk(int(snode), *args, **kwargs))
        arr = np.stack(out, axis=0).astype(np.uint16, copy=False)
        _logger.debug("_walk_batch_with_seed: produced walks shape=%s dtype=%s", arr.shape, arr.dtype)
        return arr

    def sample_walks(self,
                     # walks
                     walk_length: int,
                     walks_per_node: int,
                     saw_frac: float,
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
                     attractive_dataset_name: str = "ATTRACTIVE",
                     repulsive_dataset_name: str = "REPULSIVE",
                     RW_suffix: str = "_RWs",
                     SAW_suffix: str = "_SAWs",
                     compression_level: int = 3
                     ) -> str:
        _logger.info("sample_walks: L=%d, per_node=%d, saw_frac=%.3f, time_aware=%s, out=%s, parallel=%s",
                     walk_length, walks_per_node, saw_frac, time_aware, output_path, in_parallel)

        if output_path is None:
            _logger.error("output_path is None")
            raise ValueError("output_path must be provided (path to .zip or directory).")

        if not (0.0 <= saw_frac <= 1.0):
            _logger.error("saw_frac out of range: %r", saw_frac)
            raise ValueError("saw_frac must be in [0, 1]")

        # deterministic integer split
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

        with sawnergy_util.ArrayStorage.compress_and_cleanup(output_path, compression_level) as storage:
            # --- ATTR RWs ---
            _logger.info("Generating ATTR RWs ...")
            chunks = processor(
                work_items,
                self._walk_batch_with_seed,
                num_RWs,
                start_time_stamp=None,
                length=walk_length,
                interaction_type="attr",
                self_avoid=False,
                time_aware=time_aware,
                stickiness=stickiness,
                on_no_options=on_no_options,
            )
            if chunks:
                all_walks = np.concatenate(chunks, axis=0).astype(np.uint16, copy=False)
                _logger.info("ATTR RWs: concatenated shape=%s", all_walks.shape)
                storage.write(list(all_walks), to_block_named=attractive_dataset_name + RW_suffix)

            # --- ATTR SAWs ---
            _logger.info("Generating ATTR SAWs ...")
            chunks = processor(
                work_items,
                self._walk_batch_with_seed,
                num_SAWs,
                start_time_stamp=None,
                length=walk_length,
                interaction_type="attr",
                self_avoid=True,
                time_aware=time_aware,
                stickiness=stickiness,
                on_no_options=on_no_options,
            )
            if chunks:
                all_walks = np.concatenate(chunks, axis=0).astype(np.uint16, copy=False)
                _logger.info("ATTR SAWs: concatenated shape=%s", all_walks.shape)
                storage.write(list(all_walks), to_block_named=attractive_dataset_name + SAW_suffix)

            # --- REPULS RWs ---
            _logger.info("Generating REPULS RWs ...")
            chunks = processor(
                work_items,
                self._walk_batch_with_seed,
                num_RWs,
                start_time_stamp=None,
                length=walk_length,
                interaction_type="repuls",
                self_avoid=False,
                time_aware=time_aware,
                stickiness=stickiness,
                on_no_options=on_no_options,
            )
            if chunks:
                all_walks = np.concatenate(chunks, axis=0).astype(np.uint16, copy=False)
                _logger.info("REPULS RWs: concatenated shape=%s", all_walks.shape)
                storage.write(list(all_walks), to_block_named=repulsive_dataset_name + RW_suffix)

            # --- REPULS SAWs ---
            _logger.info("Generating REPULS SAWs ...")
            chunks = processor(
                work_items,
                self._walk_batch_with_seed,
                num_SAWs,
                start_time_stamp=None,
                length=walk_length,
                interaction_type="repuls",
                self_avoid=True,
                time_aware=time_aware,
                stickiness=stickiness,
                on_no_options=on_no_options,
            )
            if chunks:
                all_walks = np.concatenate(chunks, axis=0).astype(np.uint16, copy=False)
                _logger.info("REPULS SAWs: concatenated shape=%s", all_walks.shape)
                storage.write(list(all_walks), to_block_named=repulsive_dataset_name + SAW_suffix)

            # useful metadata for reproducibility
            storage.add_attr("seed", int(self._seed))
            storage.add_attr("rng_scheme", "SeedSequence.spawn_per_batch_v1")
            storage.add_attr("num_workers", int(num_workers))
            storage.add_attr("in_parallel", bool(in_parallel))
            storage.add_attr("batch_size_nodes", int(batch_size_nodes))
            _logger.info("Wrote metadata: seed=%d, workers=%d, parallel=%s, batch=%d",
                         int(self._seed), int(num_workers), bool(in_parallel), int(batch_size_nodes))

        _logger.info("sample_walks complete -> %s", str(output_path))
        return str(output_path)


if __name__ == "__main__":
    pass
