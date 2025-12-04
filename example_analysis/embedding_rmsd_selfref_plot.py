from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sawnergy import sawnergy_util
from sawnergy.embedding import align_frames

DATA = Path("../sawnergy/untracked/FL/processed_FL_p53_sims/Embeddings")
FILES = [
    ("WT p53 rep 1", DATA / "p53_WT_md1100_str_Rep1_pr97_wl30_ws5_embeddings.zip"),
    ("WT p53 rep 2", DATA / "p53_WT_md1100_str_Rep2_pr97_wl30_ws5_embeddings.zip"),
    ("WT p53 rep 3", DATA / "p53_WT_md1100_str_Rep3_pr97_wl30_ws5_embeddings.zip"),
    ("WT p53 rep 4", DATA / "p53_WT_md1100_str_Rep4_pr97_wl30_ws5_embeddings.zip"),
    ("Y220C p53 rep 1", DATA / "p53_Y220C_md1100_str_Rep1_pr97_wl30_ws5_embeddings.zip"),
    ("Y220C p53 rep 2", DATA / "p53_Y220C_md1100_str_Rep2_pr97_wl30_ws5_embeddings.zip"),
    ("Y220C p53 rep 3", DATA / "p53_Y220C_md1100_str_Rep3_pr97_wl30_ws5_embeddings.zip"),
    ("Y220C p53 rep 4", DATA / "p53_Y220C_md1100_str_Rep4_pr97_wl30_ws5_embeddings.zip")
]

OUT = Path("../sawnergy/example_analysis/embedding_rmsd_plot.png")

def load_emb(path: Path) -> np.ndarray:
    with sawnergy_util.ArrayStorage(path, mode="r") as st:
        emb = st.read("FRAME_EMBEDDINGS")
    return np.asarray(emb, dtype=np.float32)

def rmsd(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.linalg.norm(diff) / np.sqrt(diff.size))

if __name__ == "__main__":
    curves = []
    for label, path in FILES:
        emb = load_emb(path) # (T, N, D)
        ref = emb[0]
        rmsd_curve = []
        for t in range(emb.shape[0]):
            aligned = align_frames(emb[t], ref)
            rmsd_curve.append(rmsd(aligned, ref))
        curves.append((label, np.asarray(rmsd_curve, dtype=float)))

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"WT": "#282cad", "Y": "#fc0330"}
    for label, arr in curves:
        key = "WT" if label.startswith("WT") else "Y"
        ax.plot(range(1, len(arr) + 1), arr, label=label, color=colors[key], alpha=0.9)
    ax.set_title("RMSD curves referenced to the first time stamp within each replicate")
    ax.set_xlabel("Time")
    ax.set_ylabel("RMSD vs own frame 1 (aligned)")
    ax.legend(ncol=2, frameon=True)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT)
    print(f"Wrote {OUT}")
