"""Build the shared session-6 call dataset: per-call cards + embeddings, cached.

Outputs (in /home/andry/UVS/scripts/):
  calls_dataset.npz   cards[N,128,128] uint8, emb_dinov2[N,384], emb_resnet18[N,512],
                      feat[N,F] (z-scored acoustic features), + index arrays
  calls_meta.csv      per-call metadata (mouse, geno, phase, class, times, features)
  ../figures/fig00_card_montage.png   QC montage of example call cards
"""
import sys, pathlib
sys.path.insert(0, "/home/andry/UVS/scripts")
sys.path.insert(0, "/home/andry/tracking_project/NEMBA")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import common as C

OUT_NPZ = C.SCRIPTS / "calls_dataset.npz"
OUT_META = C.SCRIPTS / "calls_meta.csv"

print("loading calls …")
df = C.load_calls()
print(f"  {len(df)} calls  |  by geno:\n{df.groupby(['geno','phase'], observed=True).size().unstack(fill_value=0)}")

print("extracting per-call spectrogram cards …")
cards = C.extract_cards(df)
print(f"  cards {cards.shape} dtype {cards.dtype}")

# --- NEMBA frame embeddings -------------------------------------------------
# frame embedders want RGB [N,H,W,3]; map the grayscale card through magma
import matplotlib.cm as cm
cards_rgb = (cm.magma(cards.astype(np.float32) / 255.0)[..., :3] * 255).astype(np.uint8)

def frame_embed(name):
    try:
        from embedders_backbones import make_frame_embedder
        emb = make_frame_embedder(name)
        E = emb.embed(cards_rgb).astype(np.float32)   # [N,H,W,3] RGB
        print(f"  {name}: embeddings {E.shape}")
        return E
    except Exception as exc:
        print(f"  {name}: FAILED ({type(exc).__name__}: {exc})")
        return None

print("computing NEMBA frame embeddings …")
emb_dino = frame_embed("dinov2_s")
emb_res = frame_embed("resnet18")

# --- acoustic feature matrix (z-scored) ------------------------------------
feat = C.feature_matrix(df)
feat = (feat - feat.mean(0)) / (feat.std(0) + 1e-9)

# --- save -------------------------------------------------------------------
save = dict(cards=cards, feat=feat.astype(np.float32),
            mouse=df["mouse"].to_numpy(), geno=df["geno"].to_numpy(),
            phase=df["phase"].astype(str).to_numpy(),
            top1=df["class_top1"].to_numpy(),
            start=df["start(s)"].to_numpy(dtype=float))
if emb_dino is not None:
    save["emb_dinov2"] = emb_dino
if emb_res is not None:
    save["emb_resnet18"] = emb_res
np.savez_compressed(OUT_NPZ, **save)
df.to_csv(OUT_META, index=False)
print(f"saved {OUT_NPZ}  and  {OUT_META}")

# --- QC montage: a few example cards per mouse ------------------------------
fig, axes = plt.subplots(len(C.FILES), 8, figsize=(12, 1.5 * len(C.FILES)))
for r, e in enumerate(C.FILES):
    idx = np.where(df["mouse"].to_numpy() == e["mouse"])[0]
    pick = idx[np.linspace(0, len(idx) - 1, 8).astype(int)] if len(idx) else []
    for c in range(8):
        ax = axes[r, c]
        ax.axis("off")
        if c < len(pick):
            ax.imshow(cards[pick[c]], origin="upper", cmap="magma", aspect="auto")
            if c == 0:
                ax.set_title(f"{e['mouse']} ({e['geno']})", loc="left",
                             color=C.GENO_COLOR[e["geno"]], fontsize=10, fontweight="bold")
fig.suptitle("Example USV call cards (20–125 kHz, 120 ms) per mouse", fontsize=12)
C.save_fig(fig, "fig00_card_montage.png")
print("done")
