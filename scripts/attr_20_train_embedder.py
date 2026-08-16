"""Train a small convolutional autoencoder FROM SCRATCH on this dataset's own
64x64 call spectrogram crops (no pretrained weights of any kind) and export a
compact latent embedding per call.

The latent space complements the handcrafted ridge features with holistic
shape information (harmonics, jumps, noise structure) that individual "voice
fingerprints" can live in. Training is CPU-friendly: ~2k crops, ~60 epochs.

Outputs: attribution/call_embeddings.npz (latent + reconstruction loss curve)
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
import attr_common as A

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

LATENT = 24


class CropSet(Dataset):
    def __init__(self, specs: np.ndarray, augment: bool):
        self.x = specs.astype(np.float32)
        self.augment = augment

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        img = self.x[i]
        if self.augment:
            rng = np.random.default_rng()
            # small time/frequency shifts and gain jitter: invariances a voice
            # fingerprint should have (position in crop and loudness are not identity)
            img = np.roll(img, rng.integers(-3, 4), axis=1)
            img = np.roll(img, rng.integers(-2, 3), axis=0)
            img = np.clip(img * rng.uniform(0.85, 1.15), 0, 1.2)
        return torch.from_numpy(img[None].copy())


class ConvAE(nn.Module):
    def __init__(self, latent: int = LATENT):
        super().__init__()
        ch = (16, 32, 64, 96)
        enc = []
        c_in = 1
        for c in ch:
            enc += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.BatchNorm2d(c), nn.GELU()]
            c_in = c
        self.enc = nn.Sequential(*enc)                       # -> (96, 4, 4)
        self.to_latent = nn.Linear(ch[-1] * 4 * 4, latent)
        self.from_latent = nn.Linear(latent, ch[-1] * 4 * 4)
        dec = []
        rev = (96, 64, 32, 16)
        for i, c in enumerate(rev[1:] + (1,)):
            dec += [nn.ConvTranspose2d(rev[i] if i == 0 else rev[i], c, 4, stride=2, padding=1)]
            if c != 1:
                dec += [nn.BatchNorm2d(c), nn.GELU()]
        self.dec = nn.Sequential(*dec)

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.to_latent(h)

    def forward(self, x):
        z = self.encode(x)
        h = self.from_latent(z).view(-1, 96, 4, 4)
        return self.dec(h), z


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-prefix", default="call")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(A.RNG_SEED)
    np.random.seed(A.RNG_SEED)

    out = A.ensure_out()
    d = np.load(out / f"{args.in_prefix}_specs.npz", allow_pickle=False)
    specs = d["specs"].astype(np.float32)
    call_id = d["call_id"]
    print(f"training on {len(specs)} crops, latent={LATENT}")

    train_dl = DataLoader(CropSet(specs, augment=True), batch_size=args.batch,
                          shuffle=True, num_workers=0, drop_last=False)
    model = ConvAE()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.MSELoss()

    torch.set_num_threads(max(1, torch.get_num_threads()))
    losses = []
    for ep in range(args.epochs):
        model.train()
        tot, nb = 0.0, 0
        for xb in train_dl:
            opt.zero_grad()
            rec, _ = model(xb)
            loss = loss_fn(rec, xb)
            loss.backward()
            opt.step()
            tot += float(loss)
            nb += 1
        sched.step()
        losses.append(tot / nb)
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"epoch {ep+1:3d}/{args.epochs}  mse={losses[-1]:.5f}", flush=True)

    model.eval()
    with torch.no_grad():
        zs = []
        for i in range(0, len(specs), 256):
            xb = torch.from_numpy(specs[i:i + 256][:, None])
            zs.append(model.encode(xb).numpy())
        Z = np.concatenate(zs)

    # a handful of reconstructions for the method figure
    with torch.no_grad():
        idx = np.linspace(0, len(specs) - 1, 8).astype(int)
        rec, _ = model(torch.from_numpy(specs[idx][:, None]))
    np.savez_compressed(
        out / "call_embeddings.npz",
        latent=Z.astype(np.float32), call_id=call_id,
        loss_curve=np.array(losses, dtype=np.float32),
        example_idx=idx, example_orig=specs[idx],
        example_rec=rec.numpy()[:, 0],
    )
    torch.save(model.state_dict(), out / "embedder_scratch.pt")
    print(f"embeddings {Z.shape} -> {out / 'call_embeddings.npz'}")


if __name__ == "__main__":
    main()
