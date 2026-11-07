from __future__ import annotations

from importlib import import_module
from importlib.metadata import version
import inspect

import pytest


# Fixed V1 baseline after prerequisite fixes and schema validation.
# Review the specification before changing this inventory; never refresh it in tests.
SIGNATURES = """\
sawnergy.rin:RINBuilder (cpptraj_path: 'Path | str | None' = None)
sawnergy.rin:RINBuilder.build_rin (self, topology_file: 'str', trajectory_file: 'str', molecule_of_interest: 'int', frame_range: 'tuple[int, int] | None' = None, frame_batch_size: 'int' = -1, prune_low_energies_frac: 'float' = 0.85, output_path: 'str | Path | None' = None, keep_prenormalized_energies: 'bool' = True, *, include_attractive: 'bool' = True, include_repulsive: 'bool' = True, parallel_cpptraj: 'bool' = False, simul_cpptraj_instances: 'int | None' = None, num_matrices_in_compressed_blocks: 'int' = 10, compression_level: 'int' = 3, cpptraj_run_time_limit: 'float | None' = None) -> 'str'
sawnergy.rin:CpptrajScript (commands: 'tuple[str]' = <factory>) -> None
sawnergy.rin:CpptrajScript.from_cmd (cmd: 'str') -> 'CpptrajScript'
sawnergy.rin:CpptrajScript.__add__ (self, other: 'str | CpptrajScript') -> 'CpptrajScript'
sawnergy.rin:CpptrajScript.__gt__ (self, file_name: 'str | CpptrajScript') -> 'CpptrajScript'
sawnergy.rin:CpptrajScript.render (self) -> 'str'
sawnergy.rin:run_cpptraj (cpptraj: 'str', script: 'str | None' = None, argv: 'list[str] | None' = None, timeout: 'float | None' = None, *, env: 'dict | None' = None)
sawnergy.walks:Walker (RIN_path: 'str | Path', *, seed: 'int | None' = None) -> 'None'
sawnergy.walks:Walker.walk (self, start_node: 'int | None', start_time_stamp: 'int | None', length: 'int', interaction_type: "Literal['attr', 'repuls']", self_avoid: 'bool', time_aware: 'bool' = False, stickiness: 'float | None' = None, on_no_options: "Literal['raise', 'loop'] | None" = None) -> 'np.ndarray'
sawnergy.walks:Walker.sample_walks (self, walk_length: 'int', walks_per_node: 'int', saw_frac: 'float' = 0.0, include_attractive: 'bool' = True, include_repulsive: 'bool' = False, time_aware: 'bool' = False, stickiness: 'float | None' = None, on_no_options: "Literal['raise', 'loop'] | None" = None, output_path: 'str | Path | None' = None, *, in_parallel: 'bool', max_parallel_workers: 'int | None' = None, compression_level: 'int' = 3, num_walk_matrices_in_compressed_blocks: 'int | None' = None) -> 'str'
sawnergy.walks:Walker.close (self) -> 'None'
sawnergy.walks:Walker.__enter__ (self)
sawnergy.walks:Walker.__exit__ (self, exc_type, exc, tb)
sawnergy.walks:l1_norm (X: numpy.ndarray) -> numpy.ndarray
sawnergy.walks:apply_on_axis0 (X: numpy.ndarray, func)
sawnergy.walks:cosine_similarity (A: numpy.ndarray, eps: float = 1e-12)
sawnergy.embedding:Embedder (WALKS_path: 'str | Path', *, seed: 'int | None' = None) -> 'None'
sawnergy.embedding:Embedder.embed_frame (self, frame_id: 'int', RIN_type: "Literal['attr', 'repuls']", using: "Literal['RW', 'SAW', 'merged']", num_epochs: 'int', negative_sampling: 'bool' = False, window_size: 'int' = 2, num_negative_samples: 'int' = 10, batch_size: 'int' = 1024, *, in_weights: 'np.ndarray | None' = None, out_weights: 'np.ndarray | None' = None, lr_step_per_batch: 'bool' = False, shuffle_data: 'bool' = True, dimensionality: 'int' = 128, alpha: 'float' = 0.75, device: 'str | None' = None, model_base: "Literal['torch', 'pureml']" = 'pureml', model_kwargs: 'dict[str, object] | None' = None, kind: "tuple[Literal['in', 'out', 'avg']]" = ('in',), _seed: 'int | None' = None) -> 'list[tuple[np.ndarray, str]]'
sawnergy.embedding:Embedder.embed_all (self, RIN_type: "Literal['attr', 'repuls']", using: "Literal['RW', 'SAW', 'merged']", num_epochs: 'int', negative_sampling: 'bool' = False, window_size: 'int' = 2, num_negative_samples: 'int' = 10, batch_size: 'int' = 1024, *, lr_step_per_batch: 'bool' = False, shuffle_data: 'bool' = True, dimensionality: 'int' = 128, alpha: 'float' = 0.75, device: 'str | None' = None, model_base: "Literal['torch', 'pureml']" = 'pureml', model_kwargs: 'dict[str, object] | None' = None, kind: "Literal['in', 'out', 'avg']" = 'in', output_path: 'str | Path | None' = None, num_matrices_in_compressed_blocks: 'int' = 20, compression_level: 'int' = 3) -> 'str'
sawnergy.embedding:align_frames (this: 'np.ndarray', to_this: 'np.ndarray', *, center: 'bool' = True, add_back_mean: 'bool' = True, allow_reflection: 'bool' = False) -> 'np.ndarray'
sawnergy.embedding:Visualizer (EMB_path: 'str | Path', figsize: 'tuple[int, int]' = (9, 7), default_node_color: 'str' = '#6B7280', depthshade: 'bool' = False, antialiased: 'bool' = False, init_elev: 'float' = 35, init_azim: 'float' = 45, *, show: 'bool' = False, normalize_rows: 'bool' = False) -> 'None'
sawnergy.embedding:Visualizer.build_frame (self, frame_id: 'int', *, node_colors: 'str | np.ndarray | None' = 'rainbow', displayed_nodes: 'Sequence[int] | str | None' = 'ALL', show_node_labels: 'bool' = False, show: 'bool' = False) -> 'None'
sawnergy.embedding:Visualizer.savefig (self, path: 'str | Path', *, dpi: 'int' = 150) -> 'None'
sawnergy.embedding:Visualizer.close (self) -> 'None'
sawnergy.visual:Visualizer (RIN_path: 'str | Path', figsize: 'tuple[int, int]' = (9, 7), node_size: 'int' = 120, edge_width: 'float' = 1.25, default_node_color: 'str' = '#6B7280', depthshade: 'bool' = False, antialiased: 'bool' = False, init_elev: 'float' = 35, init_azim: 'float' = 45, *, show: 'bool' = False) -> 'None'
sawnergy.visual:Visualizer.build_frame (self, frame_id: 'int', displayed_nodes: "np.typing.ArrayLike | Literal['ALL'] | None" = 'ALL', displayed_pairwise_attraction_for_nodes: "np.typing.ArrayLike | Literal['DISPLAYED_NODES'] | None" = 'DISPLAYED_NODES', displayed_pairwise_repulsion_for_nodes: "np.typing.ArrayLike | Literal['DISPLAYED_NODES'] | None" = 'DISPLAYED_NODES', frac_node_interactions_displayed: 'float' = 0.01, global_interactions_frac: 'bool' = True, global_opacity: 'bool' = True, global_color_saturation: 'bool' = True, node_colors: 'str | tuple[tuple[Iterable[int], str]] | None' = None, title: 'str | None' = None, padding: 'float' = 0.1, spread: 'float' = 1.0, show: 'bool' = False, *, show_node_labels: 'bool' = False, node_label_size: 'int' = 6, attractive_edge_cmap: 'str' = 'autumn', repulsive_edge_cmap: 'str' = 'winter')
sawnergy.visual:Visualizer.animate_trajectory (self, start: 'int' = 1, stop: 'int | None' = None, step: 'int' = 1, interval_ms: 'int' = 50, loop: 'bool' = False, **build_kwargs)
sawnergy.rin.rin_util:CpptrajNotFound (candidates: 'list[Path]')
sawnergy.rin.rin_util:locate_cpptraj (explicit: 'Path | None' = None, verify: 'bool' = True) -> 'str'
sawnergy.rin.rin_util:COM_STDOUT (mol_id)
sawnergy.rin.rin_util:CpptrajMaskParser.hierarchize_molecular_composition (mol_compositions_file: 'str') -> 'dict[int, dict[int, set[int]]]'
sawnergy.sawnergy_util:ArrayStorage (pth: 'Path | str', mode: 'str') -> 'None'
sawnergy.sawnergy_util:ArrayStorage.close (self) -> 'None'
sawnergy.sawnergy_util:ArrayStorage.__enter__ (self)
sawnergy.sawnergy_util:ArrayStorage.__exit__ (self, exc_type, exc, tb)
sawnergy.sawnergy_util:ArrayStorage.__repr__ (self) -> 'str'
sawnergy.sawnergy_util:ArrayStorage.write (self, these_arrays: 'Sequence[np.ndarray] | np.ndarray', to_block_named: 'str', *, arrays_per_chunk: 'int | None' = None) -> 'None'
sawnergy.sawnergy_util:ArrayStorage.read (self, from_block_named: 'str', ids: 'int | slice | tuple[int]' = None)
sawnergy.sawnergy_util:ArrayStorage.block_iter (self, from_block_named: 'str', *, step: 'int' = 1) -> 'Iterator'
sawnergy.sawnergy_util:ArrayStorage.list_blocks (self) -> 'list[str]'
sawnergy.sawnergy_util:ArrayStorage.delete_block (self, named: 'str') -> 'None'
sawnergy.sawnergy_util:ArrayStorage.add_attr (self, key: 'str', val: 'Any') -> 'None'
sawnergy.sawnergy_util:ArrayStorage.get_attr (self, key: 'str')
sawnergy.sawnergy_util:ArrayStorage.compress (self, into: 'str | Path | None' = None, *, compression_level: 'int') -> 'str'
sawnergy.sawnergy_util:ArrayStorage.compress_and_cleanup (output_pth: 'str | Path', compression_level: 'int') -> 'Iterator[ArrayStorage]'
sawnergy.sawnergy_util:elementwise_processor (in_parallel: 'bool' = False, Executor: 'type[ThreadPoolExecutor] | type[ProcessPoolExecutor] | None' = None, max_workers: 'int | None' = None, capture_output: 'bool' = True) -> 'Callable[[Iterable[Any], Callable[..., Any], Any], list[Any] | None]'
sawnergy.sawnergy_util:batches_of (iterable: 'Iterable', batch_size: 'int' = -1, *, out_as: 'type' = <class 'list'>, ranges: 'bool' = False, inclusive_end: 'bool' = False)
sawnergy.sawnergy_util:read_lines (file_path: 'str', skip_header: 'bool' = True) -> 'list[str]'
sawnergy.sawnergy_util:temporary_file (prefix: 'str', suffix: 'str') -> 'Path'
sawnergy.sawnergy_util:create_updated_subprocess_env (**var_vals: 'Any') -> 'dict[str, str]'
sawnergy.sawnergy_util:compose_steps (*steps: 'tuple[Callable[..., Any], dict[str, Any] | None]') -> 'Callable[[Any], Any]'
sawnergy.sawnergy_util:current_time () -> 'str'
sawnergy.sawnergy_util:is_main_process () -> 'bool'
sawnergy.sawnergy_util:files_from (dir_path: 'str', pattern: 're.Pattern' = None) -> 'list[str]'
sawnergy.sawnergy_util:chunked_file (file_path: 'str', allowed_memory_percentage_hint: 'float', num_workers: 'int') -> 'Iterable[list[str]]'
sawnergy.sawnergy_util:chunked_dir (dir_path: 'str', allowed_memory_percentage_hint: 'float', num_workers: 'int')
sawnergy.logging_util:configure_logging (logs_dir: pathlib.Path | str, file_level: int = 30, console_level: int = 30, force: bool = False) -> None
"""


def _resolve(path):
    module, member = path.split(":", 1)
    obj = import_module(module)
    for name in member.split("."):
        obj = getattr(obj, name)
    return obj


@pytest.mark.parametrize("path,expected", [line.split(" ", 1) for line in SIGNATURES.splitlines()])
def test_frozen_public_signatures(path, expected):
    assert str(inspect.signature(_resolve(path))) == expected


@pytest.mark.parametrize("public_path,implementation_path", [
    ("sawnergy.rin:RINBuilder", "sawnergy.rin.rin_builder:RINBuilder"),
    ("sawnergy.rin:CpptrajScript", "sawnergy.rin.rin_util:CpptrajScript"),
    ("sawnergy.rin:run_cpptraj", "sawnergy.rin.rin_util:run_cpptraj"),
    ("sawnergy.walks:Walker", "sawnergy.walks.walker:Walker"),
    ("sawnergy.walks:SharedNDArray", "sawnergy.walks.walker_util:SharedNDArray"),
    ("sawnergy.walks:l1_norm", "sawnergy.walks.walker_util:l1_norm"),
    ("sawnergy.walks:apply_on_axis0", "sawnergy.walks.walker_util:apply_on_axis0"),
    ("sawnergy.walks:cosine_similarity", "sawnergy.walks.walker_util:cosine_similarity"),
    ("sawnergy.embedding:Embedder", "sawnergy.embedding.embedder:Embedder"),
    ("sawnergy.embedding:align_frames", "sawnergy.embedding.embedder:align_frames"),
    ("sawnergy.embedding:Visualizer", "sawnergy.embedding.visualizer:Visualizer"),
    ("sawnergy.visual:Visualizer", "sawnergy.visual.visualizer:Visualizer"),
])
def test_public_reexports_preserve_identity(public_path, implementation_path):
    assert _resolve(public_path) is _resolve(implementation_path)


@pytest.mark.parametrize("module,names", [
    ("sawnergy", "sawnergy_util logging_util rin visual walks embedding"),
    ("sawnergy.rin", "RINBuilder run_cpptraj CpptrajScript"),
    ("sawnergy.walks", "Walker SharedNDArray l1_norm apply_on_axis0 cosine_similarity"),
    ("sawnergy.embedding", "Embedder align_frames Visualizer SGNS_PureML SGNS_Torch SG_PureML SG_Torch"),
    ("sawnergy.visual", "Visualizer BLUE GREEN RED YELLOW PURPLE PINK TEAL ORANGE CYAN INDIGO GRAY LIME ROSE SKY SLATE HEAT COLD"),
    ("sawnergy.rin.rin_util", "run_cpptraj CpptrajScript"),
    ("sawnergy.sawnergy_util", "ArrayStorage elementwise_processor files_from chunked_file chunked_dir batches_of compose_steps"),
    ("sawnergy.logging_util", "configure_logging"),
])
def test_existing_exports_remain_declared(module, names):
    # Listing lazy backend names must not require importing optional backends.
    assert set(names.split()).issubset(import_module(module).__all__)


def test_package_version_is_distribution_metadata():
    assert _resolve("sawnergy:__version__") == version("sawnergy")
