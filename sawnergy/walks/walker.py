# third-pary
import numpy as np
# built-in
from pathlib import Path
from typing import Literal
from concurrent.futures import ProcessPoolExecutor
import os
# local
from . import walker_util
from .. import sawnergy_util


class Walker:

    def __init__(self,
                 RIN_path: str | Path,
                 *,
                 attr_data_name: str = "ATTRACTIVE_transitions",
                 repuls_data_name: str = "REPULSIVE_transitions",
                 seed: int | None = None) -> None:

        # load numpy arrays from read-only storage
        with sawnergy_util.ArrayStorage(RIN_path, mode="r") as storage:
            attr_matrices  : np.ndarray = storage.read(attr_data_name, slice(None))
            repuls_matrices: np.ndarray = storage.read(repuls_data_name, slice(None))

        # shape & consistency checks (expect (T, N, N))
        if attr_matrices.ndim != 3 or repuls_matrices.ndim != 3:
            raise ValueError(f"Expected (T,N,N) arrays; got {attr_matrices.shape} and {repuls_matrices.shape}")
        if attr_matrices.shape != repuls_matrices.shape:
            raise RuntimeError(f"ATTR/REPULS shapes must match exactly; got {attr_matrices.shape} vs {repuls_matrices.shape}")
        T, N1, N2 = attr_matrices.shape
        if N1 != N2:
            raise RuntimeError(f"Transition matrices must be square along last two dims; got ({N1}, {N2})")

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

        # AUXILIARY NETWORK INFORMATION
        self.time_stamp_count = T
        self.node_count       = N1

        # NETWORK ELEMENT
        self.nodes       = np.arange(0, self.node_count, 1, np.intp)
        self.time_stamps = np.arange(0, self.time_stamp_count, 1, np.intp)

        # INTERNAL
        self._memory_cleaned_up: bool = False
        self._seed = np.random.randint(0, 2**32 - 1) if seed is None else int(seed)
        self.rng = np.random.default_rng(self._seed)

    # explicit resource cleanup
    def close(self, *, unlink: bool = True) -> None:
        if self._memory_cleaned_up:
            return
        try:
            self.attr_matrices.close()
            self.repuls_matrices.close()
            if unlink:
                self.attr_matrices.unlink()
                self.repuls_matrices.unlink()
        finally:
            self._memory_cleaned_up = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #                               PRIVATE
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        
    def _matrices_of_interaction_type(self, interaction_type: Literal["attr", "repuls"]):
        if interaction_type == "attr":
            return self.attr_matrices
        elif interaction_type == "repuls":
            return self.repuls_matrices
        else:
            raise ValueError(f"`interaction_type` is expected to be `attr` or `repuls`. Instead, given: {interaction_type}")

    def _extract_prob_vector(self,
                             node: int,
                             time_stamp: int,
                             interaction_type: Literal["attr", "repuls"]):
        matrix = self._matrices_of_interaction_type(interaction_type)[time_stamp]
        return matrix[node, :].copy()  # detach from shared buffer to avoid mutation

    def _step_node(self,
                   node: int,
                   interaction_type: Literal["attr", "repuls"],
                   time_stamp: int = 0,
                   avoid: np.typing.ArrayLike | None = None
                   ) -> tuple[int, np.ndarray | None]:
        prob_dist = self._extract_prob_vector(node, time_stamp, interaction_type)

        if avoid is None:
            return int(self.rng.choice(self.nodes, p=probs)), None

        to_avoid = np.asarray(avoid, dtype=np.intp)
        keep = np.setdiff1d(self.nodes, to_avoid, assume_unique=False)
        if keep.size == 0:
            raise RuntimeError("No available node transitions (avoiding all the nodes).")

        probs = walker_util.l1_norm(prob_dist[keep])
        if probs.sum() <= 0.0:
            raise RuntimeError("No valid node transitions: probability mass is zero after masking/normalization.")

        next_node = int(self.rng.choice(keep, p=probs))
        to_avoid = np.append(to_avoid, next_node).astype(np.intp, copy=False)

        return next_node, to_avoid

    def _step_time(self,
                   time_stamp: int,
                   interaction_type: Literal["attr", "repuls"],
                   stickiness: float,
                   on_no_options: Literal["raise", "loop"],
                   avoid: np.typing.ArrayLike | None) -> tuple[int, np.ndarray | None]:
        if not (0.0 <= stickiness <= 1.0):
            raise ValueError("stickiness must be in [0,1]")
        
        to_avoid = np.array([], dtype=np.intp) if avoid is None else np.asarray(avoid, dtype=np.intp)

        # with probability = stickiness, remain at the same time stamp
        if self.rng.random() < float(stickiness):
            return int(time_stamp), to_avoid

        # exclude current time since we chose not to stick
        to_avoid = np.unique(np.append(to_avoid, time_stamp).astype(np.intp, copy=False)) # if intp already -- no new buffer
        keep = np.setdiff1d(self.time_stamps, to_avoid, assume_unique=True)

        matrices = self._matrices_of_interaction_type(interaction_type)
        current_matrix = matrices[time_stamp]  # axis-0 basic indexing is ok on the sham wrapper

        if keep.size == 0:
            if on_no_options == "raise":
                raise RuntimeError(f"No available time stamps (avoid={np.unique(to_avoid)})")
            elif on_no_options == "loop":
                # avoid current; consider all other timestamps
                to_avoid = np.array([time_stamp], dtype=np.intp)
                keep = self.time_stamps[self.time_stamps != time_stamp]
                matrices_stack = matrices.array[keep]  # fancy indexing on ndarray, not wrapper
            else:
                raise ValueError("on_no_options must be 'raise' or 'loop'")
        else:
            matrices_stack = matrices.array[keep]      # fancy indexing on ndarray, not wrapper

        sims = walker_util.apply_on_axis0(matrices_stack, walker_util.cosine_similarity(current_matrix))
        probs = walker_util.l1_norm(sims)
        if probs.sum() <= 0.0:
            raise RuntimeError(
                f"No valid time stamps to sample: probability mass is zero after masking/normalization. "
                f"time_stamp={time_stamp}, interaction_type={interaction_type}, candidates={len(keep)}."
            )

        next_time_stamp = int(self.rng.choice(keep, p=probs))
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

        # 1-based external API preserved, so validate ranges after conversion
        if start_node is not None:
            node = int(start_node) - 1
            if not (0 <= node < self.node_count):
                raise ValueError(f"start_node out of range after 1-based conversion: {start_node}")
        else:
            node = int(self.rng.choice(self.nodes))

        if start_time_stamp is not None:
            time_stamp = int(start_time_stamp) - 1
            if not (0 <= time_stamp < self.time_stamp_count):
                raise ValueError(f"start_time_stamp out of range after 1-based conversion: {start_time_stamp}")
        else:
            time_stamp = int(self.rng.choice(self.time_stamps))

        nodes_to_avoid: np.ndarray | None = np.array([node], dtype=np.intp) if self_avoid else None
        time_stamps_to_avoid: np.ndarray | None = None

        pth = np.array([node], dtype=np.intp)

        if time_aware and (stickiness is None or on_no_options is None):
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

        return pth

    # deterministic per-batch worker: (start_nodes_batch, seedseq/int) -> stack of walks
    def _walk_batch_with_seed(self, work_item, num_walks_from_each: int, *args, **kwargs):
        start_nodes, seed_obj = work_item
        self.rng = np.random.default_rng(seed_obj)  # SeedSequence or int OK
        start_nodes = np.asarray(start_nodes, dtype=np.intp)
        out = []
        for snode in start_nodes:
            for _ in range(int(num_walks_from_each)):
                out.append(self.walk(int(snode), *args, **kwargs))
        return np.stack(out, axis=0).astype(np.uint16, copy=False)

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

        if output_path is None:
            raise ValueError("output_path must be provided (path to .zip or directory).")

        if not (0.0 <= saw_frac <= 1.0):
            raise ValueError("saw_frac must be in [0, 1]")

        # deterministic integer split
        num_SAWs = int(round(walks_per_node * float(saw_frac)))
        num_RWs  = int(walks_per_node) - num_SAWs

        num_workers = os.cpu_count() or 1
        batch_size_nodes = (num_workers if in_parallel else 1)

        if in_parallel and not sawnergy_util.is_main_process():
            raise RuntimeError(
                "Process-based parallelism requires running under `if __name__ == '__main__':`."
            )

        processor = sawnergy_util.elementwise_processor(
            in_parallel=in_parallel,
            Executor=ProcessPoolExecutor,
            max_workers=num_workers,
            capture_output=True
        )

        # Pre-build node batches deterministically
        node_batches = list(sawnergy_util.batches_of(self.nodes, batch_size=batch_size_nodes))
        # Derive deterministic child seeds from master seed — stable per batch
        master_ss = np.random.SeedSequence(self._seed)
        child_seeds = master_ss.spawn(len(node_batches))
        work_items = list(zip(node_batches, child_seeds))

        with sawnergy_util.ArrayStorage.compress_and_cleanup(output_path, compression_level) as storage:
            # --- ATTR RWs ---
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
                storage.write(list(all_walks), to_block_named=attractive_dataset_name + RW_suffix)

            # --- ATTR SAWs ---
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
                storage.write(list(all_walks), to_block_named=attractive_dataset_name + SAW_suffix)

            # --- REPULS RWs ---
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
                storage.write(list(all_walks), to_block_named=repulsive_dataset_name + RW_suffix)

            # --- REPULS SAWs ---
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
                storage.write(list(all_walks), to_block_named=repulsive_dataset_name + SAW_suffix)

            # useful metadata for reproducibility
            storage.add_attr("seed", int(self._seed))
            storage.add_attr("rng_scheme", "SeedSequence.spawn_per_batch_v1")
            storage.add_attr("num_workers", int(num_workers))
            storage.add_attr("in_parallel", bool(in_parallel))
            storage.add_attr("batch_size_nodes", int(batch_size_nodes))

        return str(output_path)


if __name__ == "__main__":
    pass
