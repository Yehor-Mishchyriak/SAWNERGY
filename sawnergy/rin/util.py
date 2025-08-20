from __future__ import annotations

import logging
from dataclasses import dataclass, field
import os, shutil, subprocess
from pathlib import Path

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class CpptrajNotFound(RuntimeError):
    def __init__(self, candidates: list[Path]) -> None:
        msg = (
            "Could not locate a working `cpptraj` executable.\n"
            f"Tried the following locations:\n" +
            "\n".join(f"  - {p}" for p in candidates) +
            "\nEnsure that AmberTools is installed and `cpptraj` is on your PATH, "
            "or set the CPPTRAJ environment variable to its location."
        )
        super().__init__(msg)

@dataclass(frozen=True)
class CpptrajScript:
    commands: tuple[str] = field(default_factory=tuple)

    @classmethod
    def from_cmd(cls, cmd: str) -> CpptrajScript:
        return cls((cmd,))

    def __add__(self, other: str | CpptrajScript) -> CpptrajScript:
        if isinstance(other, str):
            return CpptrajScript(self.commands + (other,))
        elif isinstance(other, CpptrajScript):
            return CpptrajScript(self.commands + other.commands)
        else:
            return NotImplemented

    def __or__(self, file_name: str) -> CpptrajScript: # |
        save_to = (self.commands[-1] + f" out {file_name}",)
        return CpptrajScript(self.commands[:-1] + save_to)

    def __ge__(self, file_name: str) -> CpptrajScript: # >=
        save_to = (self.commands[-1] + f" avgout {file_name}",)
        return CpptrajScript(self.commands[:-1] + save_to)
    
    def __rshift__(self, file_name: str) -> CpptrajScript: # >>
        save_to = (self.commands[-1] + f" emapout elec_{file_name} vmapout vdw_{file_name}",)
        return CpptrajScript(self.commands[:-1] + save_to)

    def __gt__(self, file_name: str) -> CpptrajScript: # >
        save_to = (self.commands[-1] + f" eout {file_name}",)
        return CpptrajScript(self.commands[:-1] + save_to)

    def render(self) -> str:
        return "\n".join(self.commands + ("run", "quit", ""))

# *----------------------------------------------------*
#                       FUNCTIONS
# *----------------------------------------------------*

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
#  WRAPPERS AND HELPERS FOR THE CPPTRAJ EXECUTABLE
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

def locate_cpptraj(explicit: Path | None = None, verify: bool = True) -> str:
    """Locate a working `cpptraj` executable.

    This function attempts to resolve the path to the `cpptraj` binary used in
    AmberTools. It searches for `cpptraj` in the following order:

    1. An explicitly provided path.
    2. The `CPPTRAJ` environment variable.
    3. System PATH (via `shutil.which`).
    4. The `AMBERHOME/bin` directory.
    5. The `CONDA_PREFIX/bin` directory.

    Each candidate path is checked for existence and executability. If
    `verify=True`, the function also probes the binary with the `-h` flag
    to ensure it responds correctly.

    Args:
        explicit (Path | None): An explicit path to the `cpptraj` executable.
            If provided, this is the first candidate tested.
        verify (bool): If True, run `cpptraj -h` to confirm functionality
            of the executable. If False, only existence and executability
            are checked. Defaults to True.

    Returns:
        str: The absolute path to a verified `cpptraj` executable.

    Raises:
        CpptrajNotFound: If no functional `cpptraj` instance can be located.
        subprocess.TimeoutExpired: If the `cpptraj -h` verification command
            exceeds the timeout limit.
    """

    _logger.info("Attempting to locate a `cpptraj` executable")

    if explicit is not None: _logger.info(f"An explicit path was provided: {explicit.resolve()}")
    else: _logger.info("No explicit path was provided")

    candidates = []
    if explicit: candidates.append(Path(explicit))

    if os.getenv("CPPTRAJ"): candidates.append(Path(os.getenv("CPPTRAJ")))

    for name in ("cpptraj", "cpptraj.exe"):
        exe = shutil.which(name)
        if exe: candidates.append(Path(exe))

    if os.getenv("AMBERHOME"): candidates.append(Path(os.getenv("AMBERHOME")) / "bin" / "cpptraj")
    if os.getenv("CONDA_PREFIX"): candidates.append(Path(os.getenv("CONDA_PREFIX")) / "bin" / "cpptraj")

    _logger.info(f"Checking the following paths for cpptraj presence: {candidates}")
    for p in candidates:
        if p and p.exists() and os.access(p, os.X_OK):
            _logger.info(f"Found a `cpptraj` instance at {p}")

            if not verify:
                _logger.info(f"No verification was prompted. Returning the path {p}")
                return str(p.resolve())

            _logger.info("Attempting to verify that it works")
            try:
                # cpptraj -h prints a help message
                proc = subprocess.run([str(p), "-h"], capture_output=True, text=True, timeout=15) # 15 sec timeout
            except subprocess.TimeoutExpired:
                _logger.warning(f"The instance at {p} hung during verification (timeout). Skipping.")
                continue

            if proc.returncode in (0, 1):
                _logger.info(f"The instance is functional. Returning the path {p}")
                return str(p.resolve())
            else:
                _logger.warning(f"The instance is not functional: {proc.stderr}")
    
    _logger.error(f"No functional `cpptraj` instance was found")
    raise CpptrajNotFound(candidates)

def run_cpptraj(cpptraj: str,
                script: str | None = None,
                argv: list[str] | None = None,
                timeout: int = 30):
    args = [cpptraj] + (argv or [])
    try:
        proc = subprocess.run(
            args,
            input=script,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout
        )
        return proc.stdout
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        _logger.error(f"Cpptraj execution failed (code {e.returncode}). Stderr:\n{stderr}")
        raise
    except Exception as e:
        _logger.error(f"Unexpected error while running cpptraj: {e}")
        raise

# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #
#  CPPTRAJ OUTPUT PARSERS
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-= #

def com_parser(line: str) -> str:
    frame, x, y, z, _, _, _= line.split()
    return f"{frame},{x},{y},{z}\n"


if __name__ == "__main__":
    pass
