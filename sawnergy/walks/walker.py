# third-pary
import numpy as np
# built-in
from pathlib import Path
from typing import Literal
# local
from . import walker_util
from .. import sawnergy_util


class Walker:

    def __init__(self,
                RIN_path: str | Path,
                *,
                attr_data_name: str = "ATTRACTIVE_transitions",
                repuls_data_name: str = "REPULSIVE_transitions") -> None:

        with sawnergy_util.ArrayStorage(RIN_path, mode="r") as storage:
            attr_matrices  : np.ndarray = storage.read(attr_data_name, slice(None))
            repuls_matrices: np.ndarray = storage.read(repuls_data_name, slice(None))

        # NETWORK ELEMENT
        self.attr_matrices   = walker_util.SharedNDArray.create(
            shape=attr_matrices.shape,
            dtype=attr_matrices.dtype,
            from_array=attr_matrices
        )

        # NETWORK ELEMENT
        self.repuls_matrices = walker_util.SharedNDArray.create(
            shape=repuls_matrices.shape,
            dtype=repuls_matrices.dtype,
            from_array=repuls_matrices
        )

        # AUXILIARY NETWORK INFORMATION
        self.time_stamp_count = self.attr_matrices.shape[0]
        self.node_count       = self.attr_matrices.shape[1]

        # NETWORK ELEMENT
        self.nodes       = np.arange(0, self.node_count, 1, np.intp)
        self.time_stamps = np.arange(0, self.time_stamp_count, 1, np.intp)

        # VALIDATION
        if self.attr_matrices.shape[0] != self.repuls_matrices.shape[0]:
            raise RuntimeError("attr/repuls time dimension mismatch")
        if self.attr_matrices.shape[1] != self.repuls_matrices.shape[1]:
            raise RuntimeError("attr/repuls node dimension mismatch")

        # INTERNAL
        self._memory_cleaned_up: bool = False

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
        return matrix[node, :].copy()

    def _step_node(self,
                  node: int,
                  interaction_type: Literal["attr", "repuls"],
                  time_stamp: int = 0,  # 0 is default because having 1 avged over the traj matrix is common
                  avoid: np.typing.ArrayLike | None = None
                  ) -> tuple[int, np.ndarray | None]:
        prob_dist = self._extract_prob_vector(node, time_stamp, interaction_type)

        if avoid is None:
            keep = self.nodes
            probs = walker_util.l1_norm(prob_dist)
            if probs.sum() <= 0.0:
                raise RuntimeError("No valid node transitions: probability mass is zero.")
            next_node = int(np.random.choice(keep, p=probs))
            # no self-avoid tracking when avoid=None
            return next_node, None

        # self-avoid path
        to_avoid = np.asarray(avoid, dtype=np.intp)
        # candidates = all nodes except those avoided
        keep = np.setdiff1d(self.nodes, to_avoid, assume_unique=False)
        if keep.size == 0:
            raise RuntimeError("No available node transitions (avoiding all the nodes).")

        probs = walker_util.l1_norm(prob_dist[keep])
        if probs.sum() <= 0.0:
            raise RuntimeError("No valid node transitions: probability mass is zero after masking/normalization.")

        next_node = int(np.random.choice(keep, p=probs))
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

        # since .random() is uniform, `stickiness`% of the time it will fall below `stickiness`
        if np.random.random() < float(stickiness):
            return int(time_stamp), to_avoid

        # exclude current time since we chose not to stick
        to_avoid = np.unique(np.append(to_avoid, time_stamp).astype(np.intp, copy=False))
        keep = np.setdiff1d(self.time_stamps, to_avoid, assume_unique=True)

        matrices = self._matrices_of_interaction_type(interaction_type)
        current_matrix = matrices[time_stamp]

        if keep.size == 0:
            if on_no_options == "raise":
                raise RuntimeError(f"No available time stamps (avoid={np.unique(to_avoid)})")
            elif on_no_options == "loop":
                to_avoid = np.array([time_stamp], dtype=np.intp) # avoid current still
                keep = self.time_stamps
                matrices_stack = matrices.array  # using ndarray, not SharedNDArray
            else:
                raise ValueError("on_no_options must be 'raise' or 'loop'")
        else:
            matrices_stack = matrices.array[keep]  # fancy indexing on ndarray

        sims = walker_util.apply_on_axis0(matrices_stack, walker_util.cosine_similarity(current_matrix))
        probs = walker_util.l1_norm(sims)
        if probs.sum() <= 0.0:
            raise RuntimeError(
                f"No valid time stamps to sample: probability mass is zero after masking/normalization. "
                f"time_stamp={time_stamp}, interaction_type={interaction_type}, candidates={len(keep)}. "
                "Likely causes: all candidate matrices have zero norm, all similarities evaluated to 0, or all candidates were masked."
            )

        next_time_stamp = int(np.random.choice(keep, p=probs))
        return next_time_stamp, to_avoid  # unchanged because we already added current to avoid above
    
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

        node = int(start_node)-1 if start_node is not None else int(np.random.choice(self.nodes))
        time_stamp = int(start_time_stamp)-1 if start_time_stamp is not None else int(np.random.choice(self.time_stamps))

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


if __name__ == "__main__":
    pass
