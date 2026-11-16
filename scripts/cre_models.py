"""StressCRE candidate networks (plan step 4). Imported by train_cre.py and by the later scoring steps
(paralog pairs, promoter variants), which reload the saved state_dicts from models/<name>/fold<k>.pt.

All candidates are multi-task: 10 logits (5 stresses x up/down, order TASKS) and take the gene covariates
(plan 3.5, decision 5) through CovNet, joined before the dense layers. For a variant or a paralog comparison the
covariates of a gene are held fixed, so differences come from the sequence only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

STRESSES = ["drought", "salt", "heat", "cold", "alkaline"]
TASKS = [f"{s}-{d}" for s in STRESSES for d in ("up", "down")]
CODE = {"A": 0, "C": 1, "G": 2, "T": 3}          # anything else (N, masked CDS, spacer) -> 4 = all-zero column


def encode(seq):
    """uint8 codes 0-4 for one sequence (A C G T other)."""
    import numpy as np
    lut = np.full(256, 4, np.uint8)
    for b, v in CODE.items():
        lut[ord(b)] = v
        lut[ord(b.lower())] = v
    return lut[np.frombuffer(seq.encode(), np.uint8)]


def onehot(codes):
    """codes [B, L] (uint8 on the device) -> float one-hot [B, 4, L]; code 4 gives zeros."""
    return F.one_hot(codes.long(), 5)[..., :4].permute(0, 2, 1).float()


def revcomp_onehot(x):
    """reverse complement of a one-hot batch [B, 4, L] with channel order A C G T."""
    return x.flip([1, 2])


class CovNet(nn.Module):
    def __init__(self, n_cov, d=32):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(n_cov, 64), nn.ReLU(), nn.Linear(64, d), nn.ReLU())

    def forward(self, c):
        return self.f(c)


class CNN(nn.Module):
    """Candidate 1: DeepCRE-style CNN from scratch (Peleke et al. 2024 layout: three blocks of two k=8
    convolutions (64, 128, 64 channels) + max-pool 8 + dropout, then dense 128/64), batch norm added."""

    def __init__(self, n_cov, n_out=10, drop=0.25, L=3020):
        super().__init__()

        def block(ci, co):
            return [nn.Conv1d(ci, co, 8, padding="same"), nn.BatchNorm1d(co), nn.ReLU(),
                    nn.Conv1d(co, co, 8, padding="same"), nn.BatchNorm1d(co), nn.ReLU(),
                    nn.MaxPool1d(8), nn.Dropout(drop)]

        self.trunk = nn.Sequential(*block(4, 64), *block(64, 128), *block(128, 64), nn.Flatten())
        n_flat = 64 * (((L // 8) // 8) // 8)
        self.cov = CovNet(n_cov)
        self.head = nn.Sequential(nn.Linear(n_flat + 32, 128), nn.ReLU(), nn.Dropout(drop),
                                  nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, n_out))

    def forward(self, x, c):
        return self.head(torch.cat([self.trunk(x), self.cov(c)], 1))


class PC2Head(nn.Module):
    """Candidate 2: small network on frozen PlantCAD2-Small features [B, 60 bins, 1536]
    (bins 0-29: TSS -1000/+500, bins 30-59: TTS -500/+1000; 50 bp each).
    Per-bin layer norm + projection to d, learned bin position, two residual 1-D convolutions over bins,
    attention pooling over bins separately for every task, then [pooled ; covariates] -> shared hidden layer
    -> one output vector per task."""

    def __init__(self, n_cov, n_out=10, d_in=1536, d=128, n_bins=60, drop=0.2):
        super().__init__()
        self.norm = nn.LayerNorm(d_in)
        self.proj = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(drop))
        self.pos = nn.Parameter(torch.zeros(n_bins, d))
        self.conv1 = nn.Conv1d(d, d, 5, padding=2)
        self.conv2 = nn.Conv1d(d, d, 5, padding=2)
        self.drop = nn.Dropout(drop)
        self.att = nn.Linear(d, n_out)
        self.cov = CovNet(n_cov)
        self.hid = nn.Linear(d + 32, 64)
        self.w = nn.Parameter(torch.randn(n_out, 64) * 0.05)
        self.b = nn.Parameter(torch.zeros(n_out))

    def forward(self, e, c):
        h = self.proj(self.norm(e)) + self.pos                     # [B, 60, d]
        t = h.transpose(1, 2)
        t = t + self.drop(F.gelu(self.conv1(t)))
        t = t + self.drop(F.gelu(self.conv2(t)))
        h = t.transpose(1, 2)                                      # [B, 60, d]
        a = torch.softmax(self.att(h), dim=1)                      # [B, 60, T] attention over bins per task
        pooled = torch.einsum("bit,bid->btd", a, h)                # [B, T, d]
        z = torch.cat([pooled, self.cov(c).unsqueeze(1).expand(-1, pooled.shape[1], -1)], 2)
        z = F.relu(self.hid(z))                                    # [B, T, 64]
        return (z * self.w).sum(-1) + self.b                       # [B, T]
