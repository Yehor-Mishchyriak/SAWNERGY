from pathlib import Path
import logging 
from . import util as rin_util
from .. import util as sawnergy_util

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

    def __init__(self, cpptraj_path: Path | str | None = None) -> None:
        if isinstance(cpptraj_path, str):
            cpptraj_path = Path(cpptraj_path)

        self.cpptraj = rin_util.locate_cpptraj(explicit=cpptraj_path, verify=True)

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #  CPPTRAJ HELPERS
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        
    def _get_number_frames(self, topology_file: str, trajectory_file: str) -> int:
        out = rin_util.run_cpptraj(self.cpptraj, argv=["-p", topology_file, "-y", trajectory_file, "-tl"]).replace("Frames: ", "")
        try:
            return int(out)
        except ValueError:
            raise RuntimeError(f"Could not retrieve the number of frames from '{trajectory_file}' trajectory")
    
    def _get_atomic_composition_of_molecule(self, topology_file: str, trajectory_file: str, molecule_id: int) -> dict:
        tmp_file: Path = sawnergy_util.temporary_file(prefix="mol_comp", suffix=".dat")
        try:
            molecule_compositions_script = self._load_data_from(topology_file, trajectory_file, 1, 1) + \
                                       self._extract_molecule_compositions() | str(tmp_file)
            script = molecule_compositions_script.render()
            rin_util.run_cpptraj(self.cpptraj, script=script)
            hierarchy = rin_util.CpptrajMaskParser.hierarchize_molecular_composition(tmp_file)
            return hierarchy[molecule_id]
        finally:
            try:
                tmp_file.unlink()
            except OSError:
                pass
    # ---------------------------------------------------------------------------------------------- #
    #                                       CPPTRAJ COMMANDS
    # ---------------------------------------------------------------------------------------------- #

    @staticmethod
    def _load_data_from(topology_file: str,
                       trajectory_file: str,
                       start_frame: int,
                       end_frame: int) -> rin_util.CpptrajScript:
        return rin_util.CpptrajScript((f"parm {topology_file}", f"trajin {trajectory_file} {start_frame} {end_frame}"))

    @staticmethod
    def _calc_nonbonded_energies_in_molecule(molecule_id: int) -> rin_util.CpptrajScript:
        return rin_util.CpptrajScript.from_cmd(f"pairwise ^{molecule_id} cuteelec 0.0 cutevdw 0.0")
    
    @staticmethod
    def _extract_molecule_compositions() -> rin_util.CpptrajScript:
        return rin_util.CpptrajScript.from_cmd(f"mask :*")

    # ---------------------------------------------------------------------------------------------- #
    #                                          POST-CPPTRAJ
    # ---------------------------------------------------------------------------------------------- #

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #  CONVERSION OF ATOMIC LEVEL INTERACTIONS INTO RESIDUE LEVEL
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

    # ---------------------------------------------------------------------------------------------- #
    #                                           PUBLIC API
    # ---------------------------------------------------------------------------------------------- #

    def build_rin(self,
                 topology_file: str,
                 trajectory_file: str,
                 molecule_of_interest: int,
                 frame_range: tuple[int, int] | None = None,
                 frame_batch_size: int = -1) -> str:
        
        # ---------- MD META DATA ----------
        total_frames = self._get_number_frames(topology_file, trajectory_file)
        molecule_composition = self._get_atomic_composition_of_molecule(topology_file, trajectory_file, molecule_of_interest)

        # -------- INTERACTION DATA --------
        # note: in 'frame_range+1' the 1 is added for end inclusion
        frames = sawnergy_util.batches_of(range(1, frame_range+1), batch_size=frame_batch_size, out_as=tuple, inclusive_end=True)


__all__ = [
    "RINBuilder"
]
