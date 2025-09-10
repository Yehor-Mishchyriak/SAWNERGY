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

    def _step_node(self, # RECODE
                  node: int,
                  time_stamp: int,
                  interaction_type: Literal["attr", "repuls"],
                  avoid: np.typing.ArrayLike) -> int:
        # make it return the update to_avouid array
        prob_dist = self._extract_prob_vector(node, time_stamp, interaction_type)
        if avoid is not None:
            prob_dist[np.asarray(avoid, dtype=np.intp)] = 0.0
            if prob_dist.sum() <= 0.0:
                raise RuntimeError("No available node transitions (avoiding all the nodes).")
            prob_dist = walker_util.l1_norm(prob_dist)
        return int(np.random.choice(self.nodes, p=prob_dist))

    def _step_time(self, # RECODE
                time_stamp: int,
                interaction_type: Literal["attr", "repuls"],
                avoid: np.typing.ArrayLike,
                *,
                stickiness: float,
                on_no_options: Literal["raise", "loop"]) -> tuple[int, np.ndarray]:
        if not (0.0 <= stickiness <= 1.0):
            raise ValueError("stickiness must be in [0,1]")
        
        avoid_arr = np.asarray(avoid, dtype=np.intp)

        # stickiness:
        # since .random() is uniform, `stickiness`% of the time it will fall below `stickiness`
        if np.random.random() < float(stickiness):
            return int(time_stamp), avoid_arr

        # exclude current time since we chose not to stick
        to_avoid = np.append(avoid_arr, time_stamp).astype(np.intp, copy=False)
        keep = np.setdiff1d(self.time_stamps, np.unique(to_avoid))

        matrices = self._matrices_of_interaction_type(interaction_type)
        current_matrix = matrices[time_stamp]

        if keep.size == 0:
            if on_no_options == "raise":
                raise RuntimeError(f"No available time stamps (avoid={np.unique(avoid_arr)})")
            elif on_no_options == "loop":
                # reset avoidance and sample from all time stamps
                to_avoid = np.array([], dtype=np.intp)
                keep = self.time_stamps
                matrices_stack = matrices.array # using ndarray, not SharedNDArray
            else:
                raise ValueError("on_no_options must be 'raise' or 'loop'")
        else:
            matrices_stack = matrices.array[keep] # fancy indexing on ndarray

        sims = walker_util.apply_on_axis0(matrices_stack, walker_util.cosine_similarity(current_matrix))
        probs = walker_util.l1_norm(sims)
        if probs.sum() <= 0.0:
            raise RuntimeError(
                f"No valid time stamps to sample: probability mass is zero after masking/normalization. "
                f"time_stamp={time_stamp}, interaction_type={interaction_type}, candidates={len(keep)}. "
                "Likely causes: all candidate matrices have zero norm, all similarities evaluated to 0, or all candidates were masked."
            )

        return int(np.random.choice(keep, p=probs)), to_avoid
    
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #                                PUBLIC
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

    def walk(start: int, length: int, self_avoid: bool, time_aware: bool, stickiness: float | None = None):
        pth = np.zeros(shape=(length,), dtype=np.uint16)
        for _ in range(length):
            pass
            # step node and step time and store updated to_avoid arrays for further use
            # pass them regardless of whether time_aware or self_avoid,
            # in those cases they must be empty though, so make sure
            # this behaviour is consistent across _step_node and _step_time
            



if __name__ == "__main__":
    pass
