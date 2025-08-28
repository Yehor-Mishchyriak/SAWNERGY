from __future__ import annotations

# third-pary
import numpy as np
# built-in
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import re
import math
import tempfile
# local
from . import rin_util
from .. import sawnergy_util

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__) 

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class RINBuilder:
    """
    Class responsible for building Residue Interaction Networks
    from MD simulations by individual frames or batches of frames
    (averaging interaction energies across each batch).
    """

    def __init__(self, cpptraj_path: Path | str | None = None):
        if isinstance(cpptraj_path, str):
            cpptraj_path = Path(cpptraj_path)

        self.cpptraj = rin_util.locate_cpptraj(explicit=cpptraj_path, verify=True)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #  CPPTRAJ HELPERS
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        
    _elec_vdw_pattern = re.compile(r"""
        ^\s*\[printdata\s+PW\[EMAP\]\s+square2d\s+noheader\]\s*\r?\n
        (.*?)   
        ^\s*\[printdata\s+PW\[VMAP\]\s+square2d\s+noheader\]\s*\r?\n
        (.*?)                                               
        (?=^\s*TIME:|\Z)
    """, re.MULTILINE | re.DOTALL | re.VERBOSE)

    def _get_number_frames(self,
                        topology_file: str,
                        trajectory_file: str,
                        *,
                        subprocess_env: dict | None = None) -> int:
        out = rin_util.run_cpptraj(self.cpptraj,
                                argv=["-p", topology_file, "-y", trajectory_file, "-tl"],
                                env=subprocess_env).replace("Frames: ", "")
        try:
            return int(out)
        except ValueError:
            raise RuntimeError(f"Could not retrieve the number of frames from '{trajectory_file}' trajectory")
    
    def _get_atomic_composition_of_molecule(self,
                                        topology_file: str,
                                        trajectory_file: str,
                                        molecule_id: int,
                                        *,
                                        subprocess_env: dict | None = None) -> dict:
        tmp_file: Path = sawnergy_util.temporary_file(prefix="mol_comp", suffix=".dat")
        try:
            molecule_compositions_script = (self._load_data_from(topology_file, trajectory_file, 1, 1) + \
                                       self._extract_molecule_compositions()) > str(tmp_file)
            script = molecule_compositions_script.render()
            rin_util.run_cpptraj(self.cpptraj, script=script, env=subprocess_env)
            hierarchy = rin_util.CpptrajMaskParser.hierarchize_molecular_composition(tmp_file)
            return hierarchy[molecule_id]
        finally:
            try:
                tmp_file.unlink()
            except OSError:
                pass

    def _calc_avg_atomic_interactions_in_frames(self,
                                        frame_range: tuple[int, int],
                                        topology_file: str,
                                        trajectory_file: str,
                                        molecule_id: int,
                                        *,
                                        subprocess_env: dict | None = None) -> np.ndarray:
        start_frame, end_frame = frame_range
        interaction_calc_script = (
            self._load_data_from(topology_file, trajectory_file, start_frame, end_frame)
            + self._calc_nonbonded_energies_in_molecule(molecule_id)
        ) > rin_util.PAIRWISE_STDOUT
        script = interaction_calc_script.render()
        output = rin_util.run_cpptraj(self.cpptraj, script=script, env=subprocess_env)

        m = self._elec_vdw_pattern.search(output)
        if not m:
            raise ValueError("Could not find EMAP/VMAP blocks in cpptraj output.")
        emap_txt, vmap_txt = m.group(1), m.group(2)

        # Robust to wrapped lines: read all numbers, ignore line structure
        emap_flat = np.fromstring(emap_txt, dtype=np.float32, sep=' ')
        vmap_flat = np.fromstring(vmap_txt, dtype=np.float32, sep=' ')

        if emap_flat.size != vmap_flat.size:
            raise ValueError(f"EMAP and VMAP sizes differ: {emap_flat.size} vs {vmap_flat.size}")

        n = int(round(math.sqrt(emap_flat.size)))
        if n * n != emap_flat.size:
            raise ValueError(f"Block is not square: {emap_flat.size} values (cannot reshape to nxn).")

        elec_matrix = emap_flat.reshape(n, n)
        vdw_matrix  = vmap_flat.reshape(n, n)

        interaction_matrix = (elec_matrix + vdw_matrix).astype(np.float32)
        return interaction_matrix

    # ---------------------------------------------------------------------------------------------- #
    #                                       CPPTRAJ COMMANDS
    # ---------------------------------------------------------------------------------------------- #

    @staticmethod
    def _load_data_from(topology_file: str,
                       trajectory_file: str,
                       start_frame: int,
                       end_frame: int) -> rin_util.CpptrajScript:
        return rin_util.CpptrajScript((f"parm {topology_file}",
                                       f"trajin {trajectory_file} {start_frame} {end_frame}",
                                       "noprogress silenceactions"))

    @staticmethod
    def _calc_nonbonded_energies_in_molecule(molecule_id: int) -> rin_util.CpptrajScript:
        return rin_util.CpptrajScript.from_cmd(f"pairwise PW ^{molecule_id} cuteelec 0.0 cutevdw 0.0")
    
    @staticmethod
    def _extract_molecule_compositions() -> rin_util.CpptrajScript:
        return rin_util.CpptrajScript.from_cmd(f"mask :*")

    # ---------------------------------------------------------------------------------------------- #
    #                                          POST-CPPTRAJ
    # ---------------------------------------------------------------------------------------------- #

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #  CONVERSION OF ATOMIC LEVEL INTERACTIONS INTO RESIDUE LEVEL
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    def _compute_residue_membership_matrix(res_to_atoms: dict[int, set[int]],
                                        *,
                                        dtype=np.float32) -> np.ndarray:
        pairs = [(a, r) for r, atoms in res_to_atoms.items() for a in atoms]
        if not pairs:
            return np.zeros((0, 0), dtype=dtype)

        # aligned index arrays
        rows = np.fromiter((a for a, _ in pairs), dtype=np.int64) - 1 # -1 because Python is 0-based
        cols = np.fromiter((r for _, r in pairs), dtype=np.int64) - 1

        n_res = int(cols.max()) + 1
        n_atoms = int(rows.max()) + 1

        # vectorized membership assignment
        P = np.zeros((n_atoms, n_res), dtype=dtype)
        P[rows, cols] = 1.0
        return P

    def _convert_atomic_to_residue_interactions(atomic_matrix: np.ndarray,
                                            membership_matrix: np.ndarray) -> np.ndarray:
        return (membership_matrix.T @ atomic_matrix @ membership_matrix).astype(dtype=np.float32)

    # ---------------------------------------------------------------------------------------------- #
    #                                           PUBLIC API
    # ---------------------------------------------------------------------------------------------- #

    def build_rin(self,
                 topology_file: str,
                 trajectory_file: str,
                 molecule_of_interest: int,
                 frame_range: tuple[int, int] | None = None,
                 frame_batch_size: int = -1,
                 *,
                 in_parallel: bool = False,
                 max_workers: int = 2) -> str:
        
        # --------------------- MD META DATA ----------------------
        total_frames = self._get_number_frames(topology_file, trajectory_file)
        molecule_composition = self._get_atomic_composition_of_molecule(topology_file, trajectory_file, molecule_of_interest)

        # ----------- AUXILIARY VARIABLES' PREPARATION ------------
        if frame_batch_size <= 0:
            frame_batch_size = total_frames
        
        if frame_range is None:
            start_frame, end_frame = 0, total_frames
        else:
            start_frame, end_frame = frame_range

        # ------ INTERACTION DATA EXTRACTION AND PROCESSING -------
        frames = sawnergy_util.batches_of(range(start_frame, end_frame+1), batch_size=frame_batch_size, out_as=tuple, inclusive_end=True)

        # initialize the frame processor
        frame_processor = sawnergy_util.elementwise_processor(in_parallel=in_parallel,
                                                      Executor=ThreadPoolExecutor,
                                                      max_workers=max_workers,
                                                      capture_output=True)
        # initialize the matrix processor
        matrix_processor = sawnergy_util.elementwise_processor(in_parallel=False, capture_output=True)

        # limit cpptraj parallelism
        subprocess_env = sawnergy_util.create_updated_subprocess_env(
            OMP_NUM_THREADS=1,
            MKL_NUM_THREADS=1,
            OPENBLAS_NUM_THREADS=1,
            MKL_DYNAMIC=False
        ) if in_parallel else None

        # create a membership matrix for atoms in residues
        membership_matrix = self._compute_residue_membership_matrix(molecule_composition)

        for frame_batch in sawnergy_util.batches_of(frames, batch_size=max_workers):
            atomic_matrices = frame_processor(frame_batch,
                                    self._calc_avg_atomic_interactions_in_frames,
                                    topology_file,
                                    trajectory_file,
                                    molecule_of_interest,
                                    subprocess_env=subprocess_env)
            residue_matrices = matrix_processor(atomic_matrices, self._convert_atomic_to_residue_interactions, membership_matrix)

            


        # ---- INTERACTION DATA POST-PROCESSING ----
        ...

        # ----- INTERACTION DATA NORMALIZATION -----
        ...

        # -------- INTERACTION DATA STORAGE --------
        

__all__ = [
    "RINBuilder"
]
