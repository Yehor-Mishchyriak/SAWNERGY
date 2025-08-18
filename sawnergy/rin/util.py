import logging
import os, shutil, subprocess
from pathlib import Path


_logger = logging.getLogger(__name__)


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


def run_cpptraj(cpptraj: Path, script: str):
    """Run a cpptraj script with the specified executable.

    This function executes the `cpptraj` binary with the given script
    content passed via standard input. It enforces error checking, so
    non-zero return codes from `cpptraj` raise an exception.

    Args:
        cpptraj (Path): Path to the `cpptraj` executable.
        script (str): The cpptraj script to execute, passed as input.

    Raises:
        subprocess.SubprocessError: If the subprocess fails due to a 
            `cpptraj`-related error (e.g., non-zero exit status).
        Exception: If any other unexpected error occurs during execution.

    Returns:
        None
    """
    try:
        subprocess.run([str(cpptraj)], input=script, text=True, check=True)
    except subprocess.SubprocessError as e:
        _logger.error(f"Cpptraj execution failed: {e}")
        raise # raises the last caught exception
    except Exception as e:
        _logger.error(f"Unexpected error while running cpptraj: {e}")
        raise
        

if __name__ == "__main__":
    pass
