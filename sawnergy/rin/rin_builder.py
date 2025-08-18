from pathlib import Path
import logging 
from typing import Any
from . import util

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

        self.cpptraj = util.locate_cpptraj(explicit=cpptraj_path, verify=True)

    # ----------------------------------------------------------------------------------------------
    #                                       CPPTRAJ COMMANDS
    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def _load_data_from(topology_file: str,
                       trajectory_file: str,
                       start_frame: int | None = None,
                       end_frame: int | None = None) -> util.CpptrajScript:
        if (start_frame is not None) and (end_frame is not None):
            return util.CpptrajScript((f"parm {topology_file}", f"trajin {trajectory_file} {start_frame} {end_frame}"))    
        return util.CpptrajScript((f"parm {topology_file}", f"trajin {trajectory_file}"))

    @staticmethod
    def _calc_pairwise_nonbonded_energies(start_res: int,
                                        end_res: int,
                                        interaction_type: str = None) -> util.CpptrajScript:
        # nonbonded pairwise energies are the interaction strengths (AT~AT).
        # H-bonds (and everything else: salt bridges, hydrophobics via LJ, etc.) are implicitly captured.
        return util.CpptrajScript.from_cmd(f"pairwise :{start_res}-{end_res} cuteelec 0.0 cutevdw 0.0")
    
    @staticmethod
    def _map_atoms_to_residues(start_res: int, end_res: int):
        return util.CpptrajScript.from_cmd(f"atominfo :{start_res}-{end_res}")

    # ----------------------------------------------------------------------------------------------
    #                                          POST-CPPTRAJ
    # ----------------------------------------------------------------------------------------------

    """HERE GOES CONVERSION FROM RAW CPPTRAJ .DATs TO AGGREGATED NORMALIZED INTERACTION MATRICES"""
    """WILL LEVERAGE CYTHON AND THREADING"""

    # ----------------------------------------------------------------------------------------------
    #                                           PUBLIC API
    # ----------------------------------------------------------------------------------------------

    def build_rin(self,
                 topology_file: str,
                 trajectory_file: str,
                 frame_range: tuple[int, int] | None = None,
                 batch_size: int = 1) -> str:
        # script = (self._load_data_from(topology_file, trajectory_file, 1, 2) + self._calc_pairwise_nonbonded_energies(1, 5)) >> "example1.dat"
        script = (self._load_data_from(topology_file, trajectory_file, 1, 2) + self._map_atoms_to_residues(1, 5)) | "mapping.dat"
        print(script.render())
        util.run_cpptraj(self.cpptraj, script.render())


__all__ = [
    "RINBuilder"
]
