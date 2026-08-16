"""V2 stage 2: bout-contrastive voice embedding, trained from scratch.

v1's autoencoder learned "what calls look like" (reconstruction). Here we train
what ATTRIBUTION needs: an embedding where calls by the same caller are close.
Labels come free from bout structure - consecutive calls with gaps < 0.5 s are
emitted by one animal with near-certainty (bout continuity), so same-bout call
pairs are positives. Negatives are other calls in the batch (mostly other
sessions/animals). Singleton-bout calls fall back to augmented self-pairs
(SimCLR-style). No stim IDs, no pretrained weights.

Outputs: attribution/v2_contrastive.npz (32-d embedding per call)
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A

import torch
import torch.nn as nn
import torch.nn.functional as F

DIM = 32
EPOCHS = 80
BATCH_BOUTS = 96
TEMP = 0.15


class Encoder(nn.Module):
    def __init__(self, dim: int = DIM):
        super().__init__()
        ch = (16, 32, 64, 96)
        layers = []
        c_in = 1
        for c in ch:
            layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1),
                       nn.BatchNorm2d(c), nn.GELU()]
            c_in = c
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(ch[-1] * 16, 128), nn.GELU(),
                                  nn.Linear(128, dim))

    def forward(self, x):
        return F.normalize(self.head(self.trunk(x).flatten(1)), dim=1)


def augment(img: np.ndarray, rng) -> np.ndarray:
    out = np.roll(img, rng.integers(-4, 5), axis=1)
    out = np.roll(out, rng.integers(-2, 3), axis=0)
    out = np.clip(out * rng.uniform(0.8, 1.2), 0, 1.3)
    if rng.random() < 0.3:
        out = out + rng.normal(0, 0.05, out.shape).astype(np.float32)
    return out.astype(np.float32)


def main():
    torch.manual_seed(A.RNG_SEED)
    rng = np.random.default_rng(A.RNG_SEED)
    out = A.ensure_out()

    specs = np.load(out / "call_specs.npz", allow_pickle=False)
    S = specs["specs"].astype(np.float32)
    ids = specs["call_id"]
    bouts = pd.read_csv(out / "v2_bouts.csv")
    bmap = dict(zip(bouts["call_id"], bouts["bout_id"]))
    bout_of = np.array([bmap.get(c, f"solo_{i}") for i, c in enumerate(ids)])

    bout_to_rows: dict[str, list[int]] = {}
    for i, b in enumerate(bout_of):
        bout_to_rows.setdefault(b, []).append(i)
    bout_keys = np.array(list(bout_to_rows))
    print(f"{len(S)} crops, {len(bout_keys)} bouts")

    model = Encoder()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    losses = []
    for ep in range(EPOCHS):
        model.train()
        perm = rng.permutation(len(bout_keys))
        tot, nb = 0.0, 0
        for i0 in range(0, len(perm), BATCH_BOUTS):
            sel = bout_keys[perm[i0:i0 + BATCH_BOUTS]]
            va, vb = [], []
            for b in sel:
                rows = bout_to_rows[b]
                if len(rows) >= 2:
                    r1, r2 = rng.choice(rows, 2, replace=False)
                else:
                    r1 = r2 = rows[0]
                va.append(augment(S[r1], rng))
                vb.append(augment(S[r2], rng))
            xa = torch.from_numpy(np.stack(va)[:, None])
            xb = torch.from_numpy(np.stack(vb)[:, None])
            za = model(xa)
            zb = model(xb)
            logits = za @ zb.T / TEMP
            labels = torch.arange(len(sel))
            loss = 0.5 * (F.cross_entropy(logits, labels)
                          + F.cross_entropy(logits.T, labels))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sched.step()
        losses.append(tot / nb)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep+1:3d}/{EPOCHS}  infoNCE={losses[-1]:.4f}", flush=True)

    model.eval()
    with torch.no_grad():
        Z = np.concatenate([
            model(torch.from_numpy(S[i:i + 256][:, None])).numpy()
            for i in range(0, len(S), 256)])
    np.savez_compressed(out / "v2_contrastive.npz", z=Z.astype(np.float32),
                        call_id=ids, loss_curve=np.array(losses, np.float32))
    torch.save(model.state_dict(), out / "v2_contrastive.pt")
    print(f"embedding {Z.shape} -> v2_contrastive.npz")


if __name__ == "__main__":
    main()
