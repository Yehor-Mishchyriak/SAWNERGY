from pathlib import Path
import logging 
from typing import Any, Iterable
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

    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
    #  CPPTRAJ HELPERS
    # -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
        
    def _get_number_frames(self, topology_file: str, trajectory_file: str) -> int:
        out = util.run_cpptraj(self.cpptraj, argv=["-p", topology_file, "-y", trajectory_file, "-tl"]).replace("Frames: ", "")
        try:
            return int(out)
        except ValueError:
            raise RuntimeError(f"Could not retrieve the number of frames from '{trajectory_file}' trajectory")
    
    @staticmethod
    def _selection_mask(atom_ids: Iterable[int]) -> str:
        return f"@{','.join(map(str, atom_ids))}"

    @staticmethod
    def _hierarchize_molecular_composition(mol_compositions_file: str) -> dict:
        pass

    # ---------------------------------------------------------------------------------------------- #
    #                                       CPPTRAJ COMMANDS
    # ---------------------------------------------------------------------------------------------- #

    @staticmethod
    def _load_data_from(topology_file: str,
                       trajectory_file: str,
                       start_frame: int,
                       end_frame: int) -> util.CpptrajScript:
        return util.CpptrajScript((f"parm {topology_file}", f"trajin {trajectory_file} {start_frame} {end_frame}"))

    @staticmethod
    def _calc_pairwise_nonbonded_energies(atom_ids: Iterable[int]) -> util.CpptrajScript:
        return util.CpptrajScript.from_cmd(f"pairwise {RINBuilder._selection_mask(atom_ids)} cuteelec 0.0 cutevdw 0.0")
    
    @staticmethod
    def _extract_molecule_compositions() -> util.CpptrajScript:
        return util.CpptrajScript.from_cmd(f"mask :*")

    # ---------------------------------------------------------------------------------------------- #
    #                                          POST-CPPTRAJ
    # ---------------------------------------------------------------------------------------------- #

    # ---------------------------------------------------------------------------------------------- #
    #                                           PUBLIC API
    # ---------------------------------------------------------------------------------------------- #

    def build_rin(self,
                 topology_file: str,
                 trajectory_file: str,
                 frame_range: tuple[int, int] | None = None,
                 batch_size: int = 1) -> str:
        # script = (self._load_data_from(topology_file, trajectory_file, 1, 2) + self._calc_pairwise_nonbonded_energies(1, 5)) >> "example1.dat"
        script = (self._load_data_from(topology_file, trajectory_file, 1, 1) + self._map_atoms_to_residues()) | "mapping.dat"
        util.run_cpptraj(self.cpptraj, script.render())


__all__ = [
    "RINBuilder"
]
