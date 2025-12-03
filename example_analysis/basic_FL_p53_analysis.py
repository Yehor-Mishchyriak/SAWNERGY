from pathlib import Path
import logging
import torch  # optional, only when using model_base="torch"

from sawnergy.logging_util import configure_logging
from sawnergy.rin import RINBuilder
from sawnergy.walks import Walker
from sawnergy.embedding import Embedder

configure_logging("./logs", file_level=logging.WARNING, console_level=logging.INFO, force=True)

# 1) Build a Residue Interaction Network archive
rin_path = Path("RIN_demo.zip")
rin_builder = RINBuilder()  # auto-locates cpptraj
rin_builder.build_rin(
    topology_file="system.prmtop",
    trajectory_file="trajectory.nc",
    molecule_of_interest=1,
    frame_range=(1, 100),          # inclusive, 1-based
    frame_batch_size=10,           # processed in 10-frame batches
    prune_low_energies_frac=0.85,  # per-row quantile pruning
    include_attractive=True,
    include_repulsive=False,
    output_path=rin_path,
)

# 2) Sample random / self-avoiding walks
walks_path = Path("WALKS_demo.zip")
with Walker(rin_path, seed=123) as walker:
    walker.sample_walks(
        walk_length=16,
        walks_per_node=100,
        saw_frac=0.25,          # 25% SAWs, 75% RWs
        include_attractive=True,
        include_repulsive=False,
        time_aware=False,
        output_path=walks_path,
        in_parallel=False,      # required kwarg; set True only under a main-guard
    )

# 3) Train per-frame embeddings
embedder = Embedder(walks_path, seed=999)
emb_path = embedder.embed_all(
    RIN_type="attr",              # "attr" or "repuls"
    using="merged",               # "RW" | "SAW" | "merged"
    num_epochs=10,
    negative_sampling=False,      # SG (False) or SGNS (True)
    window_size=4,
    device="cuda" if torch.cuda.is_available() else "cpu",
    model_base="torch",
    kind="in",                    # stored embedding kind
    output_path="EMBEDDINGS_demo.zip",
)
print("Embeddings written to", emb_path)