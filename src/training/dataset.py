"""In-memory dataset over precomputed ImageNet-1K latents.

The full latent tensor (1.28M x 4 x 32 x 32 fp16 ~= 10 GB) is loaded into
host RAM at construction. Indexing then becomes O(1) without disk I/O,
which keeps the training step GPU-bound. Horizontal flip on latents is
the only augmentation, matching DiT/SiT.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "imagenet_latents"


class LatentShardDataset(Dataset):
    """Concatenates all precomputed shards of a split into RAM."""

    def __init__(
        self,
        data_dir: str | Path = DEFAULT_DATA_DIR,
        split: str = "train",
        hflip_prob: float = 0.5,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.hflip_prob = float(hflip_prob)
        if not (0.0 <= self.hflip_prob <= 1.0):
            raise ValueError(f"hflip_prob must be in [0, 1], got {self.hflip_prob}")

        index_path = self.data_dir / "index.json"
        if not index_path.exists():
            raise FileNotFoundError(f"missing index file: {index_path}")
        index = json.loads(index_path.read_text())

        shards = sorted(
            name for name in index
            if name.startswith(f"{split}_") and name.endswith(".pt")
        )
        if not shards:
            raise FileNotFoundError(
                f"no shards for split={split!r} under {self.data_dir}"
            )

        total_n = sum(int(index[name]["n"]) for name in shards)
        self.latents = torch.empty((total_n, 4, 32, 32), dtype=torch.float16)
        self.labels = torch.empty((total_n,), dtype=torch.int64)

        offset = 0
        for name in shards:
            shard = torch.load(self.data_dir / name, weights_only=True)
            n = shard["latents"].shape[0]
            self.latents[offset:offset + n] = shard["latents"]
            self.labels[offset:offset + n] = shard["labels"].long()
            offset += n
        assert offset == total_n, f"offset {offset} != total {total_n}"

        self._shards = shards
        self._split = split

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.latents[idx]
        if self.hflip_prob > 0.0 and torch.rand(()) < self.hflip_prob:
            z = torch.flip(z, dims=[-1])
        return z, self.labels[idx]

    def __repr__(self) -> str:
        gb = self.latents.numel() * self.latents.element_size() / 1e9
        return (
            f"LatentShardDataset(split={self._split!r}, n={len(self)}, "
            f"shards={len(self._shards)}, latents={gb:.2f} GB, "
            f"hflip_prob={self.hflip_prob})"
        )
