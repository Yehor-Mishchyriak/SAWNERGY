from pathlib import Path
from typing import Any
from . import util

# *----------------------------------------------------*
#                         GLOBALS
# *----------------------------------------------------*


class RINBuilder:
    """
    Callable class responsible for building Residue Interaction Networks
    from an MD simulation by individual frames or batches of frames
    (averaging across each batch).
    """

    def __init__(self, cpptraj_path: Path | None = None) -> None:
        if isinstance(cpptraj_path, str):
            cpptraj_path = Path(cpptraj_path)

        self.cpptraj = util.locate_cpptraj(explicit=cpptraj_path, verify=True)

    # ----------------------------------------------------------------------------------------------
    #                                       CPPTRAJ COMMANDS
    # ----------------------------------------------------------------------------------------------

    @staticmethod
    def load_data_from(topology_file: str, trajectory_file: str, start_frame: int, end_frame: int):
        return (f"parm {topology_file}\n"
                f"trajin {trajectory_file} {start_frame} {end_frame}")

    @staticmethod
    def calc_pairwise_nonbonded_energies(start_res_id: int, end_res_id: int, interaction_type: str):
        """
        'interaction_type' can either be EELEC or EVDW
        """
        return f"pairwise {interaction_type} :{start_res_id}-{end_res_id} :{start_res_id}-{end_res_id}"
    
    @staticmethod
    def save_avg_to(pth: Path):
        return f"avgout {pth.resolve()}"
    
    @staticmethod
    def cpptraj_script_from(*commands: str):
        return "\n".join(commands + ("run",))

    def __call__(self,
                 trajectory_file: Path,
                 topology_file: Path,
                 frame_range: tuple[int, int] | None = None,
                 batch_size: int = 1) -> Path:
        pass


__all__ = [
    "RINBuilder"
]
