from __future__ import annotations

import logging
from dataclasses import dataclass, field
import os, shutil, subprocess
from pathlib import Path
from ..util import read_lines
import re

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
    """Immutable container for building cpptraj scripts from composable commands.

    This class supports operator-based composition to make script authoring concise.
    Commands are kept in order and `render()` produces a full cpptraj input, ending
    with `run` and `quit`.

    Attributes:
        commands: Ordered tuple of cpptraj command strings.
    """
    commands: tuple[str] = field(default_factory=tuple)

    @classmethod
    def from_cmd(cls, cmd: str) -> CpptrajScript:
        """Create a script from a single cpptraj command.

        Args:
            cmd: A single cpptraj command line (without trailing newline).

        Returns:
            CpptrajScript: A new script containing exactly this command.
        """
        return cls((cmd,))

    def __add__(self, other: str | CpptrajScript) -> CpptrajScript:
        """Append a command or concatenate another script.

        Using `+` with a string appends that command as the next line.
        Using `+` with another `CpptrajScript` concatenates their commands.

        Args:
            other: A command string to append, or another script to concatenate.

        Returns:
            CpptrajScript: A new script with commands combined.

        Raises:
            NotImplementedError: Returned implicitly as `NotImplemented` if `other`
                is of an unsupported type (lets Python try reversed operation).
        """
        if isinstance(other, str):
            return CpptrajScript(self.commands + (other,))
        elif isinstance(other, CpptrajScript):
            return CpptrajScript(self.commands + other.commands)
        else:
            return NotImplemented

    def __or__(self, file_name: str) -> CpptrajScript: # |
        """Add an `out <file>` redirection to the last command via `|`.

        This is syntactic sugar for appending `out <file>` to the most recent command.

        Args:
            file_name: Output file name to use with `out`.

        Returns:
            CpptrajScript: A new script with the modified last command.
        """
        save_to = (self.commands[-1] + f" out {file_name}",)
        return CpptrajScript(self.commands[:-1] + save_to)

    def __ge__(self, file_name: str) -> CpptrajScript: # >=
        """Add an `avgout <file>` redirection to the last command via `>=`.

        Args:
            file_name: Output file name to use with `avgout`.

        Returns:
            CpptrajScript: A new script with the modified last command.
        """
        save_to = (self.commands[-1] + f" avgout {file_name}",)
        return CpptrajScript(self.commands[:-1] + save_to)
    
    def __rshift__(self, file_name: str) -> CpptrajScript: # >>
        """Add `emapout elec_<file> vmapout vdw_<file>` to the last command via `>>`.

        This is a convenience for energy/van der Waals map outputs commonly used in cpptraj.

        Args:
            file_name: Base file name used to derive both `elec_...` and `vdw_...` outputs.

        Returns:
            CpptrajScript: A new script with the modified last command.
        """
        save_to = (self.commands[-1] + f" emapout elec_{file_name} vmapout vdw_{file_name}",)
        return CpptrajScript(self.commands[:-1] + save_to)

    def __gt__(self, file_name: str) -> CpptrajScript: # >
        """Add an `eout <file>` redirection to the last command via `>`.

        Args:
            file_name: Output file name to use with `eout`.

        Returns:
            CpptrajScript: A new script with the modified last command.
        """
        save_to = (self.commands[-1] + f" eout {file_name}",)
        return CpptrajScript(self.commands[:-1] + save_to)

    def render(self) -> str:
        """Render the script into a cpptraj input string.

        The output includes all commands in order, followed by `run` and `quit`,
        and ends with a trailing newline.

        Returns:
            str: The complete cpptraj input text.
        """
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
    """Run the `cpptraj` executable with an optional inline script.

    This wrapper executes `cpptraj` either with command-line arguments only
    or with an input script provided via stdin. Standard output is returned on
    success; any non-zero exit code raises `subprocess.CalledProcessError`.

    Args:
        cpptraj: Path to the `cpptraj` executable.
        script: Complete cpptraj input (string) to feed via stdin. If None,
            no stdin is provided.
        argv: Additional command-line arguments to pass to `cpptraj`
            (e.g., topology/trajectory flags).
        timeout: Maximum wall time (seconds) allowed for the process.

    Returns:
        str: Captured standard output from the `cpptraj` process.

    Raises:
        subprocess.CalledProcessError: If `cpptraj` exits with a non-zero code.
        subprocess.TimeoutExpired: If execution exceeds `timeout`.
        Exception: For unexpected I/O or process invocation errors.
    """
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

# *----------------------------------------------------*
#                 CPPTRAJ OUTPUT PARSERS
# *----------------------------------------------------*

# MD ITEMS (atoms, residues, molecules)
class CpptrajMaskParser:
    """Namespace container for cpptraj mask table parsing helpers."""
    __slots__ = ()  # no instances allowed

    # ---------- REGEX ----------
    _spaces_pattern = re.compile(r"\s+")
    _items_pattern = re.compile(r"\[(\w+)\]")  # captures fields of interest: [AtNum], [Rnum], [Mnum]

    # --------- HELPERS ---------
    @staticmethod
    def _id2item_map(header: str) -> dict[str, int]:
        """Map column names found in bracket tokens in `header` to their 0-based indices."""
        cols = CpptrajMaskParser._items_pattern.findall(header)
        return {name: i for i, name in enumerate(cols)}

    @staticmethod
    def _collapse_spaces(s: str) -> str:
        """Normalize all whitespace runs to a single space and strip ends."""
        return CpptrajMaskParser._spaces_pattern.sub(" ", s).strip()

    @staticmethod
    def _get_row_items(row: str, header_map: dict[str, int]) -> tuple[str, str, str]:
        """Extract molecule/residue/atom IDs from a data row using the header map."""
        items = CpptrajMaskParser._collapse_spaces(row).split()
        try:
            return (
                int(items[header_map["Mnum"]]),
                int(items[header_map["Rnum"]]),
                int(items[header_map["AtNum"]]),
            )
        except KeyError as ke:
            raise ValueError(f"Required column missing in header: {ke}") from ke
        except IndexError as ie:
            raise ValueError(f"Row has fewer fields than expected: {row!r}") from ie

    # --------- PUBLIC ----------
    @staticmethod
    def hierarchize_molecular_composition(mol_compositions_file: str) -> dict[str, dict[str, set[str]]]:
        """
        Build {molecule_id: {residue_id: {atom_id, ...}, ...}} from a cpptraj mask table.

        Assumes the file's header line contains bracketed column labels (e.g., [AtNum], [Rnum], [Mnum]).
        """
        lines = read_lines(mol_compositions_file, skip_header=False)
        if not lines:
            raise RuntimeError(f"0 lines were read from {mol_compositions_file}")

        header = lines[0]
        header_map = CpptrajMaskParser._id2item_map(header)

        required = {"Mnum", "Rnum", "AtNum"}
        missing = required.difference(header_map)
        if missing:
            raise ValueError(f"Missing required columns in header: {sorted(missing)}")

        hierarchy: dict[str, dict[str, set[str]]] = {}

        for line in lines[1:]:
            if not line.strip():
                continue
            molecule_id, residue_id, atom_id = CpptrajMaskParser._get_row_items(line, header_map)

            residues = hierarchy.setdefault(molecule_id, {})
            atoms = residues.setdefault(residue_id, set())
            atoms.add(atom_id)

        return hierarchy

# CENTER OF THE MASS
def com_parser(line: str) -> str:
    """Parse a cpptraj `center of mass` output line into CSV format.

    The input line is expected to contain seven whitespace-separated fields:
    `frame x y z vx vy vz` (velocity fields ignored here). The function emits
    a CSV string with the first four values: `frame,x,y,z\\n`.

    Args:
        line: A single line from cpptraj's COM output.

    Returns:
        str: A CSV-formatted line containing `frame,x,y,z` and a trailing newline.

    Raises:
        ValueError: If the input line does not contain at least four fields.
    """
    frame, x, y, z, _, _, _= line.split()
    return f"{frame},{x},{y},{z}\n"


if __name__ == "__main__":
    pass
