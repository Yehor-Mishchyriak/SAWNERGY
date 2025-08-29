from __future__ import annotations

# third-pary
import numpy as np
import threadpoolctl
# built-in
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor
import re
import math
import os
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

    # ---------------------------------------------------------------------------------------------- #
    #                                         CPPTRAJ HELPERS
    # ---------------------------------------------------------------------------------------------- #
        
    _elec_vdw_pattern = re.compile(r"""
        ^\s*\[printdata\s+PW\[EMAP\]\s+square2d\s+noheader\]\s*\r?\n
        ([0-9.eE+\-\s]+?)
        ^\s*\[printdata\s+PW\[VMAP\]\s+square2d\s+noheader\]\s*\r?\n
        ([0-9.eE+\-\s]+?)
        (?=^\s*\[|^\s*TIME:|\Z)
    """, re.MULTILINE | re.DOTALL | re.VERBOSE)

    _com_block_pattern = lambda _, N: re.compile(rf"""
        ^[^\n]*\bCOMZ{N}\b[^\n]*\n
        ([0-9.eE+\-\s]+?)
        (?=^\s*\[quit\]\s*$)
    """, re.MULTILINE | re.DOTALL | re.VERBOSE)

    _com_row_pattern = re.compile(r'^\s*\d+\s+(.+?)\s*$', re.MULTILINE)

    def _get_number_frames(self,
                        topology_file: str,
                        trajectory_file: str,
                        *,
                        subprocess_env: dict | None = None,
                        timeout: float | None = None) -> int:
        out = rin_util.run_cpptraj(self.cpptraj,
                                argv=["-p", topology_file, "-y", trajectory_file, "-tl"],
                                env=subprocess_env,
                                timeout=timeout).replace("Frames: ", "")
        try:
            return int(out)
        except ValueError:
            raise RuntimeError(f"Could not retrieve the number of frames from '{trajectory_file}' trajectory")
    
    def _get_atomic_composition_of_molecule(self,
                                        topology_file: str,
                                        trajectory_file: str,
                                        molecule_id: int,
                                        *,
                                        subprocess_env: dict | None = None,
                                        timeout: float | None = None) -> dict:
        tmp_file: Path = sawnergy_util.temporary_file(prefix="mol_comp", suffix=".dat")
        try:
            molecule_compositions_script = (self._load_data_from(topology_file, trajectory_file, 1, 1) + \
                                       self._extract_molecule_compositions()) > str(tmp_file)
            script = molecule_compositions_script.render()
            rin_util.run_cpptraj(self.cpptraj, script=script, env=subprocess_env, timeout=timeout)
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
                                        subprocess_env: dict | None = None,
                                        timeout: float | None = None) -> np.ndarray:
        start_frame, end_frame = frame_range
        interaction_calc_script = (
            self._load_data_from(topology_file, trajectory_file, start_frame, end_frame)
            + self._calc_nonbonded_energies_in_molecule(molecule_id)
        ) > rin_util.PAIRWISE_STDOUT
        script = interaction_calc_script.render()
        output = rin_util.run_cpptraj(self.cpptraj, script=script, env=subprocess_env, timeout=timeout)

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

    def _get_residue_COMs_per_frame(
        self,
        frame_range: tuple[int, int],
        topology_file: str,
        trajectory_file: str,
        molecule_id: int,
        number_residues: int,
        *,
        subprocess_env: dict | None = None,
        timeout: float | None = None,
    ) -> np.ndarray:
        start_frame, end_frame = frame_range
        if end_frame < start_frame:
            raise ValueError(f"Bad frame_range {frame_range}: end < start")
        number_frames = end_frame - start_frame + 1

        # build and run script
        COM_script = (
            self._load_data_from(topology_file, trajectory_file, start_frame, end_frame)
            + self._compute_residue_COMs_in_molecule(molecule_id)
        ) > rin_util.COM_STDOUT(molecule_id)
        output = rin_util.run_cpptraj(self.cpptraj, script=COM_script.render(),
                                    env=subprocess_env, timeout=timeout)

        # extract the block and the per-frame coord lines
        m = self._com_block_pattern(number_residues).search(output)
        if not m:
            raise RuntimeError("Could not find COM print block in cpptraj output.")
        block = m.group(1)
        lines = self._com_row_pattern.findall(block) # list[str], coords only (no frame #)

        # sanity checks
        if len(lines) != number_frames:
            raise RuntimeError(f"Expected {number_frames} frame rows, got {len(lines)}.")

        # parse, validate, reshape
        rows = [np.fromstring(line, dtype=np.float32, sep=' ') for line in lines]
        bad = [i for i, arr in enumerate(rows) if arr.size != number_residues * 3]
        if bad:
            raise RuntimeError(
                f"Row(s) {bad[:5]} have wrong length; expected {number_residues*3} floats."
            )

        coords = np.stack(rows, axis=0)  # (n_frames, n_res*3)
        coords = coords.reshape(number_frames, number_residues, 3)  # (n_frames, n_res, 3); 3 is the X, Y, Z
        return coords

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

    @staticmethod
    def _compute_residue_COMs_in_molecule(molecule_id: int):
        return rin_util.CpptrajScript((
            "autoimage",
            "unwrap byres",
            f"for residues R inmask ^{molecule_id}  i=1;i++",
            "vector COM$i center $R",
            "done"
        ))

    # ---------------------------------------------------------------------------------------------- #
    #                                          POST-CPPTRAJ
    # ---------------------------------------------------------------------------------------------- #

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #  CONVERSION OF ATOMIC LEVEL INTERACTIONS INTO RESIDUE LEVEL
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    @staticmethod
    def _compute_residue_membership_matrix(
        res_to_atoms: dict[int, set[int]],
        *,
        dtype=np.float32
    ) -> np.ndarray:
        """
        Build an (n_atoms x n_residues) membership matrix P where P[a_idx, r_idx] = 1
        iff atom with original ID 'a' belongs to residue with original ID 'r'.
        Both residues and atoms are renumbered to contiguous 0..N-1 indices.

        Parameters
        ----------
        res_to_atoms : dict[int, set[int]]
            Mapping from (possibly non-1-based, non-contiguous) residue IDs
            to a set of (possibly non-1-based, non-contiguous) atom IDs.
        dtype : np.dtype
            Output dtype.

        Returns
        -------
        P : np.ndarray, shape (n_atoms, n_residues)
            Binary membership matrix.
        """
        if not res_to_atoms:
            return np.zeros((0, 0), dtype=dtype)

        # ----- Build contiguous indices for residues (columns) -----
        # Use numeric sort so indices are stable and predictable.
        res_ids = sorted(res_to_atoms.keys())
        res_to_col = {r: i for i, r in enumerate(res_ids)}
        n_res = len(res_ids)

        # ----- Build contiguous indices for atoms (rows) -----
        # Union all atom IDs, then sort numerically.
        atom_ids_set = set()
        for r in res_ids:
            atom_ids_set.update(res_to_atoms[r])
        atom_ids = sorted(atom_ids_set)
        atom_to_row = {a: i for i, a in enumerate(atom_ids)}
        n_atoms = len(atom_ids)

        # ----- Fill membership matrix -----
        P = np.zeros((n_atoms, n_res), dtype=dtype)
        for r in res_ids:
            c = res_to_col[r]
            for a in res_to_atoms[r]:
                P[atom_to_row[a], c] = 1.0

        return P

    @staticmethod
    def _convert_atomic_to_residue_interactions(atomic_matrix: np.ndarray,
                                                membership_matrix: np.ndarray) -> np.ndarray:
        thread_count = os.cpu_count() or 1 
        with threadpoolctl.threadpool_limits(limits=thread_count):
            result = (membership_matrix.T @ atomic_matrix @ membership_matrix).astype(dtype=np.float32)
        return result

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #  POST-PROCESSING OF RESIDUE LEVEL INTERACTIONS
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

    def _split_into_attractive_repulsive(self, residue_matrix: np.ndarray) -> np.ndarray:
        attr = np.where(residue_matrix <= 0, -residue_matrix, 0.0).astype(np.float32, copy=False)
        rep  = np.where(residue_matrix >  0, residue_matrix, 0.0).astype(np.float32, copy=False)
        return np.stack([attr, rep], axis=0) # (2, N, N)

    def _prune_low_energies(self, two_channel_residue_matrix: np.ndarray, q: float) -> np.ndarray:
        if not (0.0 < q <= 1.0):
            raise ValueError(f"Invalid 'q' value. Expected a value between 0 and 1; received: {q}")
        A = two_channel_residue_matrix[0]
        R = two_channel_residue_matrix[1]
        Ath = np.quantile(A, q, axis=1, keepdims=True)
        Rth = np.quantile(R, q, axis=1, keepdims=True)
        two_channel_residue_matrix[0] = np.where(A < Ath, 0.0, A)
        two_channel_residue_matrix[1] = np.where(R < Rth, 0.0, R)
        return two_channel_residue_matrix

    def _remove_self_interactions(self, two_channel_residue_matrix: np.ndarray) -> np.ndarray:
        np.fill_diagonal(two_channel_residue_matrix[0], 0.0); np.fill_diagonal(two_channel_residue_matrix[1], 0.0)
        return two_channel_residue_matrix
   
    def _symmetrize(self, two_channel_residue_matrix: np.ndarray) -> np.ndarray:
        A = two_channel_residue_matrix[0]
        R = two_channel_residue_matrix[1]
        two_channel_residue_matrix[0] = (A + A.T) * 0.5
        two_channel_residue_matrix[1] = (R + R.T) * 0.5
        return two_channel_residue_matrix

    def _L1_normalize(self, two_channel_residue_matrix: np.ndarray) -> np.ndarray:
        A = two_channel_residue_matrix[0]
        R = two_channel_residue_matrix[1]
        eps = 1e-12
        Asum = A.sum(axis=1, keepdims=True)
        Rsum = R.sum(axis=1, keepdims=True)
        two_channel_residue_matrix[0] = np.divide(A, np.clip(Asum, eps, None),
                                                out=np.zeros_like(A), where=Asum > 0)
        two_channel_residue_matrix[1] = np.divide(R, np.clip(Rsum, eps, None),
                                                out=np.zeros_like(R), where=Rsum > 0)
        return two_channel_residue_matrix

    def _store_interactions_array(self,
               arr: np.ndarray,
               storage: sawnergy_util.ArrayStorage,
               arrays_per_chunk: int,
               attractive_interactions_dataset_name: str,
               repulsive_interactions_dataset_name: str) -> None:
        storage.write(these_arrays=[arr[0]],
                        to_block_named=attractive_interactions_dataset_name,
                        arrays_per_chunk=arrays_per_chunk)
        storage.write(these_arrays=[arr[1]],
                        to_block_named=repulsive_interactions_dataset_name,
                        arrays_per_chunk=arrays_per_chunk)

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
                 max_workers: int = 2,
                 output_path: str | Path | None = None,
                 num_matrices_in_compressed_blocks: int = 10,
                 prune_low_energies_frac: float = 0.1,
                 cpptraj_run_time_limit: float | None = None,
                 COM_dataset_name="COM",
                 attractive_interactions_dataset_name="ATTRACTIVE",
                 repulsive_interactions_dataset_name="REPULSIVE") -> str:

        # ------------------------- MD META DATA --------------------------
        total_frames = self._get_number_frames(topology_file, trajectory_file, timeout=cpptraj_run_time_limit)
        molecule_composition = self._get_atomic_composition_of_molecule(topology_file, trajectory_file,
                                                                    molecule_of_interest, timeout=cpptraj_run_time_limit)
        number_residues = len(molecule_composition)

        # ----------- AUXILIARY VARIABLES' / TOOLS PREPARATION ------------
        number_processors = os.cpu_count() or 1
        output_path = Path((output_path or (Path(os.getcwd()) / f"RIN_{sawnergy_util.current_time()}"))).with_suffix(".zip")
  
        if frame_batch_size <= 0:
            frame_batch_size = total_frames
        
        if frame_range is None:
            start_frame, end_frame = 1, total_frames
        else:
            start_frame, end_frame = frame_range

        frames = sawnergy_util.batches_of(range(start_frame, end_frame+1), batch_size=frame_batch_size, out_as=tuple, inclusive_end=True)

        # initialize the frame processor
        frame_processor = sawnergy_util.elementwise_processor(in_parallel=in_parallel,
                                                      Executor=ThreadPoolExecutor,
                                                      max_workers=max_workers,
                                                      capture_output=True)
        # initialize the matrix processor
        matrix_processor = sawnergy_util.elementwise_processor(in_parallel=False, capture_output=False)

        # limit cpptraj parallelism if running pairwise commands in parallel in Python
        non_bonded_energies_subprocess_env = \
            sawnergy_util.create_updated_subprocess_env(
                OMP_NUM_THREADS=1,
                MKL_NUM_THREADS=1,
                OPENBLAS_NUM_THREADS=1,
                MKL_DYNAMIC=False
            ) if in_parallel else None

        # allow cpptraj parallelism for COM coordinates calculation
        COM_subprocess_env = \
            sawnergy_util.create_updated_subprocess_env(
                OMP_NUM_THREADS=number_processors,
                MKL_NUM_THREADS=number_processors,
                OPENBLAS_NUM_THREADS=number_processors,
                MKL_DYNAMIC=True
            ) if in_parallel else None


        # create a membership matrix for atoms in residues
        membership_matrix = self._compute_residue_membership_matrix(molecule_composition)

        # ---------- INTERACTION DATA EXTRACTION AND PROCESSING -----------
        with sawnergy_util.ArrayStorage.compress_and_cleanup(output_path) as storage:
            for frame_batch in sawnergy_util.batches_of(frames, batch_size=max_workers):
                atomic_matrices = frame_processor(frame_batch,
                                        self._calc_avg_atomic_interactions_in_frames,
                                        topology_file,
                                        trajectory_file,
                                        molecule_of_interest,
                                        subprocess_env=non_bonded_energies_subprocess_env,
                                        timeout=cpptraj_run_time_limit)
  
                matrix_processor(
                    atomic_matrices,
                    sawnergy_util.compose_steps({
                        self._convert_atomic_to_residue_interactions:{"membership_matrix": membership_matrix},
                        self._split_into_attractive_repulsive:{},
                        self._prune_low_energies:{"q": prune_low_energies_frac},
                        self._remove_self_interactions:{},
                        self._symmetrize:{},
                        self._L1_normalize:{},
                        self._store_interactions_array:{
                            "storage": storage,
                            "arrays_per_chunk": num_matrices_in_compressed_blocks,
                            "attractive_interactions_dataset_name": attractive_interactions_dataset_name,
                            "repulsive_interactions_dataset_name": repulsive_interactions_dataset_name}
                    })
                )

            COMs = self._get_residue_COMs_per_frame(
                frame_range=(start_frame, end_frame),
                topology_file=topology_file,
                trajectory_file=trajectory_file,
                molecule_id=molecule_of_interest,
                number_residues=number_residues,
                subprocess_env=COM_subprocess_env,
                timeout=cpptraj_run_time_limit
            )

            storage.write([COMs], to_block_named=COM_dataset_name, arrays_per_chunk=1)

        return str(output_path)


__all__ = [
    "RINBuilder"
]

if __name__ == "__main__":
    pass
