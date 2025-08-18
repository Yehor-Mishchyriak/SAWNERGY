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

    def __call__(self,
                 trajectory_file: Path,
                 topology_file: Path,
                 frame_range: tuple[int, int] | None = None,
                 batch_size: int = 1) -> Path:
        pass
