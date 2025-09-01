from __future__ import annotations

# third-pary
import numpy as np
import threadpoolctl
# built-in
from pathlib import Path
import logging
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
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
    """Builds Residue Interaction Networks (RINs) from MD trajectories.

    This class orchestrates running cpptraj to:
      * compute per-frame, per-residue centers of mass (COMs),
      * compute pairwise atomic non-bonded energies (electrostatics + van der Waals),
      * project atomic interactions to residue-level interactions,
      * post-process residue matrices (split, prune, symmetrize, normalize),
      * package outputs into a compressed archive.

    Args:
        cpptraj_path (Path | str | None): Optional explicit path to the `cpptraj`
            executable. If not provided, an attempt is made to locate it via
            `rin_util.locate_cpptraj`.

    Attributes:
        cpptraj (Path): Resolved path to the `cpptraj` executable.
    """

    def __init__(self, cpptraj_path: Path | str | None = None):
        _logger.debug("Initializing RINBuilder with cpptraj_path=%s", cpptraj_path)
        if isinstance(cpptraj_path, str):
            cpptraj_path = Path(cpptraj_path)

        self.cpptraj = rin_util.locate_cpptraj(explicit=cpptraj_path, verify=True)
        _logger.info("Using cpptraj at %s", self.cpptraj)

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
        """Return total number of frames in a trajectory.

        Args:
            topology_file (str): Path to topology (parm/prmtop) file.
            trajectory_file (str): Path to trajectory file readable by cpptraj.
            subprocess_env (dict | None): Optional environment overrides for the
                cpptraj subprocess (e.g., thread settings).
            timeout (float | None): Optional time limit (seconds) for the cpptraj call.

        Returns:
            int: Total number of frames.

        Raises:
            RuntimeError: If cpptraj output cannot be parsed into an integer.
        """
        _logger.debug("Requesting number of frames (topology=%s, trajectory=%s, timeout=%s)",
                      topology_file, trajectory_file, timeout)
        raw_out = rin_util.run_cpptraj(self.cpptraj,
                                argv=["-p", topology_file, "-y", trajectory_file, "-tl"],
                                env=subprocess_env,
                                timeout=timeout)
        _logger.debug("cpptraj -tl raw output: %r", raw_out)
        out = raw_out.replace("Frames: ", "")
        try:
            frames = int(out)
            _logger.info("Detected %d frames in trajectory %s", frames, trajectory_file)
            return frames
        except ValueError:
            _logger.exception("Failed parsing frame count from cpptraj output: %r", out)
            raise RuntimeError(f"Could not retrieve the number of frames from '{trajectory_file}' trajectory")
    
    def _get_atomic_composition_of_molecule(self,
                                        topology_file: str,
                                        trajectory_file: str,
                                        molecule_id: int,
                                        *,
                                        subprocess_env: dict | None = None,
                                        timeout: float | None = None) -> dict:
        """Extract per-residue atomic composition for a molecule.

        Runs a small cpptraj script that prints residue/atom membership and parses
        it into a dictionary mapping residue IDs to sets of atom IDs.

        Args:
            topology_file (str): Path to topology (parm/prmtop) file.
            trajectory_file (str): Path to trajectory file.
            molecule_id (int): Molecule selector/ID used by cpptraj (e.g., `^1`).
            subprocess_env (dict | None): Optional environment overrides for cpptraj.
            timeout (float | None): Optional time limit (seconds).

        Returns:
            dict: Mapping ``{residue_id: set(atom_ids)}``.

        Raises:
            KeyError: If the requested `molecule_id` is not present in parsed output.
        """
        _logger.debug("Extracting atomic composition (molecule_id=%s)", molecule_id)
        tmp_file: Path = sawnergy_util.temporary_file(prefix="mol_comp", suffix=".dat")
        _logger.debug("Temporary composition file: %s", tmp_file)
        try:
            molecule_compositions_script = (self._load_data_from(topology_file, trajectory_file, 1, 1) + \
                                       self._extract_molecule_compositions()) > str(tmp_file)
            script = molecule_compositions_script.render()
            _logger.debug("Running composition cpptraj script (len=%d chars)", len(script))
            rin_util.run_cpptraj(self.cpptraj, script=script, env=subprocess_env, timeout=timeout)
            hierarchy = rin_util.CpptrajMaskParser.hierarchize_molecular_composition(tmp_file)
            if molecule_id not in hierarchy:
                _logger.error("Molecule ID %s not found in composition hierarchy (available keys: %s)",
                              molecule_id, list(hierarchy.keys())[:10])
            comp = hierarchy[molecule_id]
            _logger.info("Retrieved composition for molecule %s (residues=%d)", molecule_id, len(comp))
            return comp
        finally:
            try:
                tmp_file.unlink()
                _logger.debug("Cleaned up temp file %s", tmp_file)
            except OSError:
                _logger.warning("Failed to remove temp file %s", tmp_file, exc_info=True)

    def _calc_avg_atomic_interactions_in_frames(self,
                                        frame_range: tuple[int, int],
                                        topology_file: str,
                                        trajectory_file: str,
                                        molecule_id: int,
                                        *,
                                        subprocess_env: dict | None = None,
                                        timeout: float | None = None) -> np.ndarray:
        """Compute average atomic interaction matrix over a frame range.

        Uses cpptraj `pairwise` to compute electrostatic (EMAP) and van der Waals
        (VMAP) atomic interaction matrices, sums them, and returns the result.

        Args:
            frame_range (tuple[int, int]): Inclusive (start_frame, end_frame).
            topology_file (str): Path to topology file.
            trajectory_file (str): Path to trajectory file.
            molecule_id (int): Molecule selector/ID for restricting computation.
            subprocess_env (dict | None): Optional environment for cpptraj.
            timeout (float | None): Optional time limit (seconds).

        Returns:
            np.ndarray: 2D array (n_atoms, n_atoms) of summed interactions.

        Raises:
            ValueError: If EMAP/VMAP blocks are not found, sizes mismatch, or the
                block cannot be reshaped into a square matrix.
        """
        start_frame, end_frame = frame_range
        _logger.debug("Calculating avg atomic interactions (frames=%s..%s, molecule_id=%s)",
                      start_frame, end_frame, molecule_id)
        interaction_calc_script = (
            self._load_data_from(topology_file, trajectory_file, start_frame, end_frame)
            + self._calc_nonbonded_energies_in_molecule(molecule_id)
        ) > rin_util.PAIRWISE_STDOUT
        script = interaction_calc_script.render()
        _logger.debug("Running pairwise cpptraj script (len=%d chars)", len(script))
        output = rin_util.run_cpptraj(self.cpptraj, script=script, env=subprocess_env, timeout=timeout)
        _logger.debug("cpptraj pairwise output length: %d", len(output))

        m = self._elec_vdw_pattern.search(output)
        if not m:
            _logger.error("EMAP/VMAP blocks not found in cpptraj output.")
            raise ValueError("Could not find EMAP/VMAP blocks in cpptraj output.")
        emap_txt, vmap_txt = m.group(1), m.group(2)

        # Robust to wrapped lines: read all numbers, ignore line structure
        emap_flat = np.fromstring(emap_txt, dtype=np.float32, sep=' ')
        vmap_flat = np.fromstring(vmap_txt, dtype=np.float32, sep=' ')
        _logger.debug("Parsed EMAP=%d values, VMAP=%d values", emap_flat.size, vmap_flat.size)

        if emap_flat.size != vmap_flat.size:
            _logger.error("Size mismatch EMAP(%d) vs VMAP(%d)", emap_flat.size, vmap_flat.size)
            raise ValueError(f"EMAP and VMAP sizes differ: {emap_flat.size} vs {vmap_flat.size}")

        n = int(round(math.sqrt(emap_flat.size)))
        if n * n != emap_flat.size:
            _logger.error("Non-square block: %d values (cannot form nxn)", emap_flat.size)
            raise ValueError(f"Block is not square: {emap_flat.size} values (cannot reshape to nxn).")

        elec_matrix = emap_flat.reshape(n, n)
        vdw_matrix  = vmap_flat.reshape(n, n)
        _logger.debug("Reshaped EMAP/VMAP to (%d, %d)", n, n)

        interaction_matrix = (elec_matrix + vdw_matrix).astype(np.float32)
        _logger.info("Computed interaction matrix shape: %s", interaction_matrix.shape)
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
    ) -> list[np.ndarray]:
        """Compute per-residue COM coordinates for each frame.

        Runs a cpptraj loop to compute ``vector COM<i>`` per residue and parses the
        printed data.

        Args:
            frame_range (tuple[int, int]): Inclusive ``(start_frame, end_frame)`` (1-based).
            topology_file (str): Path to topology file.
            trajectory_file (str): Path to trajectory file.
            molecule_id (int): Molecule selector/ID used by cpptraj for residue iteration.
            number_residues (int): Expected residue count (used for validation).
            subprocess_env (dict | None): Optional environment overrides for cpptraj.
            timeout (float | None): Optional time limit (seconds) for cpptraj.

        Returns:
            list[np.ndarray]: A list of length ``n_frames`` where each element is a
            ``(n_residues, 3)`` array of ``float32`` giving XYZ COM coordinates for that
            frame. Element ``0`` corresponds to ``start_frame``, element ``-1`` to
            ``end_frame``.

        Raises:
            ValueError: If ``frame_range`` is invalid (end < start).
            RuntimeError: If the COM print block is missing/malformed or row sizes mismatch.
        """
        start_frame, end_frame = frame_range
        _logger.debug("Getting COMs per frame (frames=%s..%s, residues=%d, molecule_id=%s)",
                    start_frame, end_frame, number_residues, molecule_id)
        if end_frame < start_frame:
            _logger.error("Bad frame_range %s: end < start", frame_range)
            raise ValueError(f"Bad frame_range {frame_range}: end < start")
        number_frames = end_frame - start_frame + 1

        # build and run cpptraj script
        COM_script = (
            self._load_data_from(topology_file, trajectory_file, start_frame, end_frame)
            + self._compute_residue_COMs_in_molecule(molecule_id)
        ) > rin_util.COM_STDOUT(molecule_id)
        script_rendered = COM_script.render()
        _logger.debug("Running COM cpptraj script (len=%d chars)", len(script_rendered))
        output = rin_util.run_cpptraj(self.cpptraj, script=script_rendered,
                                    env=subprocess_env, timeout=timeout)
        _logger.debug("cpptraj COM output length: %d", len(output))

        # extract COM block and per-frame rows
        m = self._com_block_pattern(number_residues).search(output)
        if not m:
            _logger.error("COM print block not found in cpptraj output (expected COMZ%d header).",
                        number_residues)
            raise RuntimeError("Could not find COM print block in cpptraj output.")
        block = m.group(1)
        lines = self._com_row_pattern.findall(block)  # list[str], coords only (no frame #)
        _logger.debug("Extracted %d COM rows (expected %d)", len(lines), number_frames)

        if len(lines) != number_frames:
            _logger.error("Frame row count mismatch: expected %d, got %d",
                        number_frames, len(lines))
            raise RuntimeError(f"Expected {number_frames} frame rows, got {len(lines)}.")

        # parse and reshape to (n_residues, 3) per frame
        rows = [np.fromstring(line, dtype=np.float32, sep=' ') for line in lines]
        bad = [i for i, arr in enumerate(rows) if arr.size != number_residues * 3]
        if bad:
            _logger.error("Row(s) with wrong length detected (showing first few): %s", bad[:5])
            raise RuntimeError(
                f"Row(s) {bad[:5]} have wrong length; expected {number_residues*3} floats."
            )

        coords: list[np.ndarray] = [row.reshape(number_residues, 3) for row in rows]
        _logger.info("Built %d COM arrays of shape %s (one per frame)",
                    len(coords), coords[0].shape)
        return coords

    # ---------------------------------------------------------------------------------------------- #
    #                                       CPPTRAJ COMMANDS
    # ---------------------------------------------------------------------------------------------- #

    @staticmethod
    def _load_data_from(topology_file: str,
                       trajectory_file: str,
                       start_frame: int,
                       end_frame: int) -> rin_util.CpptrajScript:
        """Create a cpptraj script that loads topology/trajectory and selects frames.

        Args:
            topology_file (str): Path to topology file.
            trajectory_file (str): Path to trajectory file.
            start_frame (int): First frame (1-based inclusive).
            end_frame (int): Last frame (1-based inclusive).

        Returns:
            rin_util.CpptrajScript: Composable script object.
        """
        _logger.debug("Preparing data load (parm=%s, trajin=%s %s %s)",
                      topology_file, trajectory_file, start_frame, end_frame)
        return rin_util.CpptrajScript((f"parm {topology_file}",
                                       f"trajin {trajectory_file} {start_frame} {end_frame}",
                                       "noprogress silenceactions"))

    @staticmethod
    def _calc_nonbonded_energies_in_molecule(molecule_id: int) -> rin_util.CpptrajScript:
        """Create a cpptraj command to compute pairwise non-bonded energies.

        Args:
            molecule_id (int): Molecule selector/ID for pairwise computation.

        Returns:
            rin_util.CpptrajScript: Script with `pairwise PW` command.
        """
        _logger.debug("Preparing pairwise command for molecule_id=%s", molecule_id)
        return rin_util.CpptrajScript.from_cmd(f"pairwise PW ^{molecule_id} cuteelec 0.0 cutevdw 0.0")
    
    @staticmethod
    def _extract_molecule_compositions() -> rin_util.CpptrajScript:
        """Create a cpptraj command that emits residue/atom masks.

        Returns:
            rin_util.CpptrajScript: Script with `mask :*` command.
        """
        _logger.debug("Preparing mask extraction command")
        return rin_util.CpptrajScript.from_cmd(f"mask :*")

    @staticmethod
    def _compute_residue_COMs_in_molecule(molecule_id: int):
        """Create a cpptraj loop to compute per-residue COM vectors.

        Args:
            molecule_id (int): Molecule selector/ID whose residues are iterated.

        Returns:
            rin_util.CpptrajScript: Script that defines COM vectors (COM1, COM2, ...).
        """
        _logger.debug("Preparing COM vectors loop for molecule_id=%s", molecule_id)
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
        """Build a binary (n_atoms x n_residues) membership matrix.

        Contiguously re-indexes arbitrary atom/residue IDs to 0..N-1 and sets
        ``P[row(atom), col(residue)] = 1`` when the atom belongs to the residue.

        Args:
            res_to_atoms (dict[int, set[int]]): Mapping of residue IDs to sets of atom IDs.
            dtype (np.dtype): Output dtype.

        Returns:
            np.ndarray: Membership matrix of shape (n_atoms, n_residues).
        """
        if not res_to_atoms:
            _logger.info("Empty residue->atoms mapping; returning (0,0) matrix.")
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

        _logger.debug("Membership dims: atoms=%d, residues=%d", n_atoms, n_res)

        # ----- Fill membership matrix -----
        P = np.zeros((n_atoms, n_res), dtype=dtype)
        for r in res_ids:
            c = res_to_col[r]
            for a in res_to_atoms[r]:
                P[atom_to_row[a], c] = 1.0

        _logger.info("Built membership matrix with shape %s and density %.6f",
                     P.shape, float(P.sum()) / (P.size if P.size else 1.0))
        return P

    @staticmethod
    def _convert_atomic_to_residue_interactions(atomic_matrix: np.ndarray,
                                                membership_matrix: np.ndarray) -> np.ndarray:
        """Project atomic interaction matrix to residue space.

        Computes ``R = Pᵀ @ A @ P`` where `A` is atomic (n_atoms x n_atoms) and
        `P` is membership (n_atoms x n_residues).

        Args:
            atomic_matrix (np.ndarray): Atomic interaction matrix (n_atoms, n_atoms).
            membership_matrix (np.ndarray): Membership matrix (n_atoms, n_residues).

        Returns:
            np.ndarray: Residue interaction matrix (n_residues, n_residues).
        """
        _logger.debug("Converting atomic->residue: atomic_matrix=%s, membership=%s",
                      atomic_matrix.shape, membership_matrix.shape)
        thread_count = os.cpu_count() or 1 
        with threadpoolctl.threadpool_limits(limits=thread_count):
            result = (membership_matrix.T @ atomic_matrix @ membership_matrix).astype(dtype=np.float32)
        _logger.info("Residue interaction matrix shape: %s", result.shape)
        return result

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #  POST-PROCESSING OF RESIDUE LEVEL INTERACTIONS
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

    def _split_into_attractive_repulsive(self, residue_matrix: np.ndarray) -> np.ndarray:
        """Split residue interactions into attractive and repulsive channels.

        Negative values go to the attractive channel (as positive magnitudes),
        positive values go to the repulsive channel.

        Args:
            residue_matrix (np.ndarray): Residue interaction matrix (N, N).

        Returns:
            np.ndarray: Array of shape (2, N, N): [attractive, repulsive].
        """
        _logger.debug("Splitting matrix into attractive/repulsive channels; input shape=%s",
                      residue_matrix.shape)
        attr = np.where(residue_matrix <= 0, -residue_matrix, 0.0).astype(np.float32, copy=False)
        rep  = np.where(residue_matrix >  0, residue_matrix, 0.0).astype(np.float32, copy=False)
        out = np.stack([attr, rep], axis=0) # (2, N, N)
        _logger.info("Two-channel matrix shape: %s", out.shape)
        return out

    def _prune_low_energies(self, two_channel_residue_matrix: np.ndarray, q: float) -> np.ndarray:
        """Zero out values below a per-row quantile threshold.

        Applies independently to attractive and repulsive channels.

        Args:
            two_channel_residue_matrix (np.ndarray): Array (2, N, N).
            q (float): Quantile in (0, 1] used as threshold.

        Returns:
            np.ndarray: Pruned two-channel matrix.

        Raises:
            ValueError: If `q` is not in (0, 1].
        """
        _logger.debug("Pruning low energies with q=%s on matrix shape=%s", q, two_channel_residue_matrix.shape)
        if not (0.0 < q <= 1.0):
            _logger.error("Invalid pruning quantile q=%s", q)
            raise ValueError(f"Invalid 'q' value. Expected a value between 0 and 1; received: {q}")
        A = two_channel_residue_matrix[0]
        R = two_channel_residue_matrix[1]
        Ath = np.quantile(A, q, axis=1, keepdims=True)
        Rth = np.quantile(R, q, axis=1, keepdims=True)
        two_channel_residue_matrix[0] = np.where(A < Ath, 0.0, A)
        two_channel_residue_matrix[1] = np.where(R < Rth, 0.0, R)
        _logger.info("Pruning done at q=%s", q)
        return two_channel_residue_matrix

    def _remove_self_interactions(self, two_channel_residue_matrix: np.ndarray) -> np.ndarray:
        """Zero the diagonal in both channels.

        Args:
            two_channel_residue_matrix (np.ndarray): Array (2, N, N).

        Returns:
            np.ndarray: Same array with zeroed diagonals.
        """
        _logger.debug("Zeroing self-interactions on shape=%s", two_channel_residue_matrix.shape)
        np.fill_diagonal(two_channel_residue_matrix[0], 0.0); np.fill_diagonal(two_channel_residue_matrix[1], 0.0)
        return two_channel_residue_matrix
   
    def _symmetrize(self, two_channel_residue_matrix: np.ndarray) -> np.ndarray:
        """Symmetrize both channels via (M + Mᵀ)/2.

        Args:
            two_channel_residue_matrix (np.ndarray): Array (2, N, N).

        Returns:
            np.ndarray: Symmetrized two-channel matrix.
        """
        _logger.debug("Symmetrizing two-channel matrix shape=%s", two_channel_residue_matrix.shape)
        A = two_channel_residue_matrix[0]
        R = two_channel_residue_matrix[1]
        two_channel_residue_matrix[0] = (A + A.T) * 0.5
        two_channel_residue_matrix[1] = (R + R.T) * 0.5
        _logger.info("Symmetrization complete")
        return two_channel_residue_matrix

    def _L1_normalize(self, two_channel_residue_matrix: np.ndarray) -> np.ndarray:
        """Row-wise L1-normalization of both channels.

        Each row is divided by its sum; zero rows remain zero.

        Args:
            two_channel_residue_matrix (np.ndarray): Array (2, N, N).

        Returns:
            np.ndarray: L1-normalized two-channel matrix.
        
        Note:
            Row-wise normalization breaks symmetry.
            This is because the numbers you get after normalization
            are relative interaction strengths. Meaning, despite
            two residues i and j having equal interaction energy
            they both have different interaction strengths with
            other neighbors, that is, altough (i,j) = (j,i),
            (i,k) != (j,k), so when you normalize you get
            different distributions, so (i,j) != (j,i) now.
        """
        _logger.debug("L1-normalizing two-channel matrix shape=%s", two_channel_residue_matrix.shape)
        A = two_channel_residue_matrix[0]
        R = two_channel_residue_matrix[1]
        eps = 1e-12
        Asum = A.sum(axis=1, keepdims=True)
        Rsum = R.sum(axis=1, keepdims=True)
        two_channel_residue_matrix[0] = np.divide(A, np.clip(Asum, eps, None),
                                                out=np.zeros_like(A), where=Asum > 0)
        two_channel_residue_matrix[1] = np.divide(R, np.clip(Rsum, eps, None),
                                                out=np.zeros_like(R), where=Rsum > 0)
        _logger.info("L1 normalization complete (zero-row counts: A=%d, R=%d)",
                     int((Asum <= eps).sum()), int((Rsum <= eps).sum()))
        return two_channel_residue_matrix

    def _store_two_channel_array(
        self,
        arr: np.ndarray,
        storage: sawnergy_util.ArrayStorage,
        arrays_per_chunk: int,
        attractive_dataset_name: str,
        repulsive_dataset_name: str,
        return_back: bool = True
    ) -> np.ndarray | None:
        """Persist a two-channel residue interaction array into storage.

        Writes the attractive channel to ``attractive_dataset_name`` and the
        repulsive channel to ``repulsive_dataset_name`` as append-only rows
        (axis 0) using the provided ``arrays_per_chunk`` for Zarr chunking.

        Parameters
        ----------
        arr : np.ndarray
            Two-channel array with shape ``(2, N, N)``:
            ``arr[0]`` = attractive magnitudes (|negative energies|),
            ``arr[1]`` = repulsive magnitudes (positive energies).
        storage : sawnergy_util.ArrayStorage
            Open ArrayStorage to write into (must be writable).
        arrays_per_chunk : int
            Chunk length along axis 0 for compression/IO efficiency.
        attractive_dataset_name : str
            Target dataset name for the attractive channel.
        repulsive_dataset_name : str
            Target dataset name for the repulsive channel.
        return_back : bool, default True
            If True, return ``arr`` so this function can be composed in a
            pipeline; otherwise return ``None``.

        Returns
        -------
        np.ndarray | None
            ``arr`` if ``return_back`` is True; otherwise ``None``.

        Notes
        -----
        This function does not modify ``arr`` in place.
        """
        if arr.ndim != 3 or arr.shape[0] != 2:
            _logger.warning(
                "Expected two-channel array with shape (2, N, N); got %s", getattr(arr, "shape", None)
            )

        _logger.debug(
            "Storing two-channel array: shape=%s, chunksize=%s, datasets=(%s, %s)",
            arr.shape, arrays_per_chunk, attractive_dataset_name, repulsive_dataset_name
        )

        storage.write(
            these_arrays=[arr[0]],
            to_block_named=attractive_dataset_name,
            arrays_per_chunk=arrays_per_chunk
        )
        storage.write(
            these_arrays=[arr[1]],
            to_block_named=repulsive_dataset_name,
            arrays_per_chunk=arrays_per_chunk
        )

        _logger.info(
            "Stored attractive/repulsive arrays to '%s' / '%s'",
            attractive_dataset_name, repulsive_dataset_name
        )

        if return_back:
            return arr
        return None

    # ---------------------------------------------------------------------------------------------- #
    #                                           PUBLIC API
    # ---------------------------------------------------------------------------------------------- #

    def build_rin(
        self,
        topology_file: str,
        trajectory_file: str,
        molecule_of_interest: int,
        frame_range: tuple[int, int] | None = None,
        frame_batch_size: int = -1,
        prune_low_energies_frac: float = 0.3,
        output_path: str | Path | None = None,
        keep_absolute_energies: bool = True,
        *,
        absolute_energies_suffix: str = "_energies",
        transition_probabilities_suffix: str = "_transitions",
        COM_dataset_name: str = "COM",
        attractive_dataset_name: str = "ATTRACTIVE",
        repulsive_dataset_name: str = "REPULSIVE",
        in_parallel: bool = False,
        max_workers: int | None = None,
        num_matrices_in_compressed_blocks: int = 10,
        compression_level: int = 3,
        cpptraj_run_time_limit: float | None = None
    ) -> str:
        """
        Build a Residue Interaction Network (RIN) archive from an MD trajectory.

        Pipeline (per requested frames):
        1) Discover MD metadata (total frames, residues of target molecule).
        2) For each frame batch:
        a) Run cpptraj pairwise on atoms → EMAP + VMAP → sum = atomic matrix.
        b) Project atomic→residue using membership matrix (Pᵀ @ A @ P).
        c) Post-process residue matrix into two channels:
            - split into attractive/repulsive,
            - prune by per-row quantile,
            - remove self-interactions,
            - symmetrize.
        d) Optionally store **absolute energies** (two blocks).
        e) Row-wise L1 normalize (produces transition probabilities) and store.
        3) Compute per-residue COM coordinates for each frame and store.
        4) Zip the store to a read-only ``.zip`` (Zarr v3) and return its path.

        Parameters
        ----------
        topology_file : str
            Path to the topology (parm/prmtop).
        trajectory_file : str
            Path to the trajectory readable by cpptraj.
        molecule_of_interest : int
            Molecule selector/ID used by cpptraj (e.g., 1 for ``^1``).
        frame_range : tuple[int, int] | None, optional
            1-based inclusive (start, end). If None, uses the full trajectory.
        frame_batch_size : int, optional
            Number of frames per batch for pairwise calculations. If <= 0, uses
            all frames (single batch).
        prune_low_energies_frac : float, optional
            Per-row quantile ``q`` in (0, 1] for pruning small values.
        output_path : str | Path | None, optional
            Destination path (with or without ``.zip``). Defaults to
            ``RIN_<YYYY-MM-DD_HHMMSS>.zip`` in the CWD.
        keep_absolute_energies : bool, optional
            If True, stores the pre-normalized attractive/repulsive matrices
            under ``<ATTRACTIVE|REPULSIVE><absolute_energies_suffix>``.
        absolute_energies_suffix : str, optional
            Suffix for absolute-energy datasets (default: ``"_energies"``).
        transition_probabilities_suffix : str, optional
            Suffix for normalized datasets (default: ``"_transitions"``).
        COM_dataset_name : str, optional
            Dataset name for COM coordinates (default: ``"COM"``).
        attractive_dataset_name : str, optional
            Base dataset name for attractive channel (default: ``"ATTRACTIVE"``).
        repulsive_dataset_name : str, optional
            Base dataset name for repulsive channel (default: ``"REPULSIVE"``).
        in_parallel : bool, optional
            If True, parallelizes per-frame work (with care to avoid oversubscription).
        max_workers : int | None, optional
            Maximum workers for thread/process pools. Defaults to ``os.cpu_count()``.
        num_matrices_in_compressed_blocks : int, optional
            Number of matrices per chunk along axis 0 in the Zarr arrays.
        compression_level : int, optional
            Blosc Zstd compression level for the final ZipStore (0-9).
        cpptraj_run_time_limit : float | None, optional
            Timeout (seconds) applied to cpptraj subprocesses.

        Returns
        -------
        str
            Path to the created ``.zip`` archive (Zarr v3).

        Notes
        -----
        - Row-wise L1 normalization produces *directed* transition probabilities
        (rows sum to 1) and breaks symmetry by design.
        - Energies are split into two channels: attractive (-E → +|E|) and repulsive (+E).
        - Logging at INFO shows milestones; DEBUG adds deeper diagnostics.
        """
        _logger.info(
            "Building RIN (mol=%s, traj=%s, frame_range=%s, frame_batch_size=%s, "
            "keep_abs=%s, parallel=%s, max_workers=%s, comp_level=%s)",
            molecule_of_interest, trajectory_file, frame_range, frame_batch_size,
            keep_absolute_energies, in_parallel, max_workers, compression_level
        )

        if in_parallel and not sawnergy_util.is_main_process():
            raise RuntimeError("Parallel execution is not allowed"
                            "If the caller process is not main."
                            "Solution: call 'build_rin' inside"
                            "`if __name__ == '__main__':\n...`\nblock.")

        # ----------------------------------- MD META DATA -------------------------------------
        total_frames = self._get_number_frames(
            topology_file,
            trajectory_file,
            timeout=cpptraj_run_time_limit
        )
        molecule_composition = self._get_atomic_composition_of_molecule(
            topology_file,
            trajectory_file,
            molecule_of_interest,
            timeout=cpptraj_run_time_limit
        )
        number_residues = len(molecule_composition)
        _logger.info("MD metadata: total_frames=%d, residues=%d", total_frames, number_residues)
        # --------------------------------------------------------------------------------------

        # --------------------- AUXILIARY VARIABLES' / TOOLS PREPARATION -----------------------
        max_workers = max_workers or (os.cpu_count() or 1)
        output_path = Path((output_path or (Path(os.getcwd()) /
                        f"RIN_{sawnergy_util.current_time()}"))).with_suffix(".zip")
        _logger.debug("Output archive: %s", output_path)

        # -=- FRAMES OF THE MD SIMULATION -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        if frame_batch_size <= 0:
            frame_batch_size = total_frames
        if frame_range is None:
            start_frame, end_frame = 1, total_frames
        else:
            start_frame, end_frame = frame_range
        frames = (
            (s, min(s + frame_batch_size - 1, end_frame))
            for s in range(start_frame, end_frame + 1, max(1, frame_batch_size))
        )
        _logger.debug("Frame batch size: %s", frame_batch_size)
        _logger.debug("Processing frames [%d..%d]", start_frame, end_frame)
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

        # -=- DATA PROCESSORS -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-
        frame_processor = sawnergy_util.elementwise_processor(
            in_parallel=in_parallel,
            Executor=ThreadPoolExecutor,
            max_workers=max_workers,
            capture_output=True
        )
        primary_matrix_processor = sawnergy_util.elementwise_processor(
            in_parallel=False,  # <- BLAS handles parallelism
            capture_output=True
        )
        secondary_matrix_processor = sawnergy_util.elementwise_processor(
            in_parallel=in_parallel,
            Executor=ProcessPoolExecutor,
            max_workers=max_workers,
            capture_output=False
        )
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=--=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=

        # -=- ADJUST CPPTRAJ PARALLELISM -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
        non_bonded_energies_subprocess_env = (
            sawnergy_util.create_updated_subprocess_env(
                OMP_NUM_THREADS=1,  # PREVENTING OVERSUBSCRIPTION
                MKL_NUM_THREADS=1,
                OPENBLAS_NUM_THREADS=1,
                MKL_DYNAMIC=False
            ) if in_parallel else None
        )

        COM_subprocess_env = (
            sawnergy_util.create_updated_subprocess_env(
                OMP_NUM_THREADS=max_workers,
                MKL_NUM_THREADS=max_workers,
                OPENBLAS_NUM_THREADS=max_workers,
                MKL_DYNAMIC=True
            ) if in_parallel else None
        )
        _logger.debug("Subprocess env set (pairwise, COM) for parallel=%s", in_parallel)
        # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

        # create a membership matrix for atoms in residues
        membership_matrix = self._compute_residue_membership_matrix(molecule_composition)
        _logger.info(
            "Membership matrix ready: shape=%s, nnz=%d",
            membership_matrix.shape, int(membership_matrix.sum())
        )
        # --------------------------------------------------------------------------------------

        # -------------------- INTERACTION DATA EXTRACTION AND PROCESSING ----------------------
        with sawnergy_util.ArrayStorage.compress_and_cleanup(output_path, compression_level) as storage:
            _logger.debug("Opened storage at %s", output_path)

            # Build secondary computation pipeline
            secondary_computation_steps = (
                (self._split_into_attractive_repulsive, None),
                (self._prune_low_energies, {"q": prune_low_energies_frac}),
                (self._remove_self_interactions, None),
                (self._symmetrize, None),
            )

            if keep_absolute_energies:
                secondary_computation_steps += (
                    (
                        self._store_two_channel_array,
                        {
                            "storage": storage,
                            "arrays_per_chunk": num_matrices_in_compressed_blocks,
                            "attractive_dataset_name": attractive_dataset_name + absolute_energies_suffix,
                            "repulsive_dataset_name": repulsive_dataset_name + absolute_energies_suffix,
                            "return_back": True,
                        },
                    ),
                )

            secondary_computation_steps += (
                (self._L1_normalize, None),
                (
                    self._store_two_channel_array,
                    {
                        "storage": storage,
                        "arrays_per_chunk": num_matrices_in_compressed_blocks,
                        "attractive_dataset_name": attractive_dataset_name + transition_probabilities_suffix,
                        "repulsive_dataset_name": repulsive_dataset_name + transition_probabilities_suffix,
                        "return_back": False,
                    },
                ),
            )

            for frame_batch in sawnergy_util.batches_of(
                frames, batch_size=max_workers if in_parallel else 1
            ):
                _logger.debug("Submitting frame batch: %s", frame_batch)

                atomic_matrices = frame_processor(
                    frame_batch,
                    self._calc_avg_atomic_interactions_in_frames,
                    topology_file,
                    trajectory_file,
                    molecule_of_interest,
                    subprocess_env=non_bonded_energies_subprocess_env,
                    timeout=cpptraj_run_time_limit,
                )

                interaction_matrices = primary_matrix_processor(
                    atomic_matrices,
                    self._convert_atomic_to_residue_interactions,
                    membership_matrix=membership_matrix,
                )

                secondary_matrix_processor(
                    interaction_matrices,
                    sawnergy_util.compose_steps(*secondary_computation_steps),
                )

                _logger.debug("Finished processing and storing a batch of matrices")

            _logger.debug(
                "Getting COMs per frame (frames=%d..%d, residues=%d, molecule_id=%s)",
                start_frame, end_frame, number_residues, molecule_of_interest
            )
            COMs = self._get_residue_COMs_per_frame(
                frame_range=(start_frame, end_frame),
                topology_file=topology_file,
                trajectory_file=trajectory_file,
                molecule_id=molecule_of_interest,
                number_residues=number_residues,
                subprocess_env=COM_subprocess_env,
                timeout=cpptraj_run_time_limit,
            )
            _logger.debug(
                "Writing COMs with %d frames to storage (chunk=%d)",
                len(COMs), num_matrices_in_compressed_blocks
            )
            storage.write(
                COMs,
                to_block_named=COM_dataset_name,
                arrays_per_chunk=num_matrices_in_compressed_blocks,
            )
        # --------------------------------------------------------------------------------------

        _logger.info("RIN build complete -> %s", output_path)
        return str(output_path)


__all__ = [
    "RINBuilder"
]

if __name__ == "__main__":
    pass
