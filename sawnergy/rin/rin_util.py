from __future__ import annotations

# built-in
import logging
from dataclasses import dataclass, field
import os, shutil, subprocess
from pathlib import Path
import re
# local
from ..sawnergy_util import read_lines

# *----------------------------------------------------*
#                        GLOBALS
# *----------------------------------------------------*

_logger = logging.getLogger(__name__)
PAIRWISE_STDOUT: CpptrajScript

# *----------------------------------------------------*
#                        CLASSES
# *----------------------------------------------------*

class CpptrajNotFound(RuntimeError):
    """Error raised when a functional `cpptraj` executable cannot be found.

    The exception message lists the candidate paths that were tried and gives a
    brief hint on how to make `cpptraj` discoverable (install AmberTools, add
    to PATH, or set the CPPTRAJ environment variable).
    """
    def __init__(self, candidates: list[Path]):
        """Initialize the exception with the candidate paths.

        Args:
            candidates: Ordered list of filesystem paths that were checked for a
                working `cpptraj` executable.
        """
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
    """Immutable builder for composing cpptraj input scripts.

    Instances hold a tuple of command strings. You can:
    - Append a command with `+ "cmd"`.
    - Concatenate two scripts with `+ other_script`.
    - Redirect the last command to a file with the overloaded `>` operator.
    - Render the final script text with `render()`, which ensures a trailing
      newline and injects a `run` command if one is not already present.

    Attributes:
        commands: Ordered tuple of cpptraj command lines (without trailing
            newlines).
    """
    commands: tuple[str] = field(default_factory=tuple)

    @classmethod
    def from_cmd(cls, cmd: str) -> CpptrajScript:
        """Create a script containing a single command.

        Args:
            cmd: A single cpptraj command line (no trailing newline required).

        Returns:
            CpptrajScript: A new script with exactly one command.
        """
        return cls((cmd,))

    def __add__(self, other: str | CpptrajScript) -> CpptrajScript:
        """Concatenate a command or another script.

        Args:
            other: Either a command string to append as a new line, or another
                `CpptrajScript` whose commands will be appended in order.

        Returns:
            CpptrajScript: A new script with `other` appended.

        Raises:
            TypeError: If `other` is not a `str` or `CpptrajScript`.
        """
        if isinstance(other, str):
            return CpptrajScript(self.commands + (other,))
        elif isinstance(other, CpptrajScript):
            return CpptrajScript(self.commands + other.commands)
        else:
            return NotImplemented

    def __gt__(self, file_name: str | CpptrajScript) -> CpptrajScript: # >
        """Overload `>` to add an `out <file>` target to the last command.

        If `file_name` is a string, append `out <file_name>` to the last command.
        If `file_name` is a `CpptrajScript`, treat this as concatenation (same
        effect as `self + file_name`).

        Args:
            file_name: Output filename to attach to the last command, or another
                script to concatenate.

        Returns:
            CpptrajScript: A new script with modified/concatenated commands.
        """
        if isinstance(file_name, CpptrajScript):
            return self + file_name
        else:
            save_to = (self.commands[-1] + f" out {file_name}",)
            return CpptrajScript(self.commands[:-1] + save_to)

    def render(self) -> str:
        """Render the script to text, auto-inserting `run` if missing.

        Returns:
            str: The full script text joined by newlines. If no `run` appears in
            `commands`, a `run` line (plus a trailing blank line) is added.
        """
        commands = self.commands + ("",) if "run" in self.commands else self.commands + ("run", "")
        return "\n".join(commands)

PAIRWISE_STDOUT = CpptrajScript((
                  "run",
                  "printdata PW[EMAP] square2d noheader",
                  "printdata PW[VMAP] square2d noheader"
                ))

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
                timeout: int = 30,
                *,
                env: dict | None = None):
    """Run `cpptraj` and return its standard output.

    If `script` text is provided, it is sent to cpptraj via STDIN. A trailing
    `quit` line is appended automatically if the script does not already end
    with one. Alternatively, you can pass command-line arguments via `argv`
    (e.g., `["-i", "script.in"]`) and leave `script=None`.

    Args:
        cpptraj: Path to the `cpptraj` executable.
        script: Complete cpptraj script to feed on STDIN. If not `None` and not
            already terminated by `quit`, the function appends `quit\\n`.
        argv: Additional command-line arguments to pass to `cpptraj`.
        timeout: Maximum wall time in seconds for the subprocess.
        env: Environment variables for the child process. Values must be strings.

    Returns:
        str: Captured `stdout` produced by `cpptraj`.

    Raises:
        subprocess.CalledProcessError: If cpptraj exits with a non-zero status.
        Exception: For unexpected errors during subprocess execution.
    """
    if script is not None:
        if not script.rstrip().lower().endswith("quit"):
            script = script + "quit\n"

    args = [cpptraj] + (argv or [])
    try:
        _logger.debug(f"Running cpptraj command: {script} with args: {args}")
        proc = subprocess.run(
            args,
            input=script,
            text=True,
            capture_output=True,
            check=True,
            timeout=timeout,
            env=env
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
    def _get_row_items(row: str, header_map: dict[str, int]) -> tuple[int, int, int]:
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
    def hierarchize_molecular_composition(mol_compositions_file: str) -> dict[int, dict[int, set[int]]]:
        """
        Build {molecule_id: {residue_id: {atom_id, ...}, ...}} from a cpptraj mask table.

        Assumes the file's header line contains bracketed column labels (e.g., [AtNum], [Rnum], [Mnum]).

        Args:
            mol_compositions_file: Path to a text file produced by cpptraj that
                lists atoms with bracketed header tokens identifying molecule,
                residue, and atom indices.

        Returns:
            dict[int, dict[int, set[int]]]: Nested mapping from molecule ID to
            residue ID to the set of atom IDs.

        Raises:
            RuntimeError: If the input file is empty.
            ValueError: If required columns are missing or a row is malformed.
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

        hierarchy: dict[int, dict[int, set[int]]] = {}

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
