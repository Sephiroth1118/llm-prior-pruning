"""Real FLOPs reduction of a level policy, from the active heads / KV groups / MLP dimensions of a level database.

ExactFlops counts the active parts of every (layer, level) cell exactly as the evaluator's mask
synchronization does: slices cast to bf16, a head (o_proj column block) or MLP dimension (down_proj
column) is active when its norm exceeds 1e-8, and a KV group is active when any of its query heads is.
"""
import os

import numpy as np

# Read from each model's config.json.
MODEL_SPECS = {
    "qwen15b": {"hidden": 1536, "intermediate": 8960, "heads": 12, "kv_heads": 2},
    "qwen3b": {"hidden": 2048, "intermediate": 11008, "heads": 16, "kv_heads": 2},
    "llama31": {"hidden": 4096, "intermediate": 14336, "heads": 32, "kv_heads": 8},
}


def build_flops_table(db_path, spec, num_layers):
    """Active query heads, KV groups and MLP dims for every (layer, level) of the database at db_path."""
    import torch

    head_dim, group = spec["hidden"] // spec["heads"], spec["heads"] // spec["kv_heads"]
    meta = torch.load(os.path.join(db_path, "metadata.pth"), map_location="cpu", weights_only=False)
    n_attn, n_mlp = len(meta["cols_removed_attn"]), len(meta["cols_removed_mlp"])

    def weight(module, layer, level):
        path = os.path.join(db_path, f"model.layers.{layer}.{module}", f"{level}.pth")
        return torch.load(path, map_location="cpu").to(torch.bfloat16).float()

    table = {"heads": [], "kv_groups": [], "mlp_dims": []}
    for layer in range(num_layers):
        heads, groups, dims = [], [], []
        for level in range(n_attn):
            w = weight("self_attn.o_proj", layer, level)
            active = w.view(w.shape[0], spec["heads"], head_dim).norm(dim=(0, 2)) > 1e-8
            heads.append(int(active.sum()))
            groups.append(int(active.view(spec["kv_heads"], group).any(1).sum()))
        for level in range(n_mlp):
            dims.append(int((weight("mlp.down_proj", layer, level).norm(dim=0) > 1e-8).sum()))
        table["heads"].append(heads)
        table["kv_groups"].append(groups)
        table["mlp_dims"].append(dims)
    return table


class ExactFlops:
    """Real FLOPs reduction of a policy, matching the evaluator's FLOPs estimate."""

    def __init__(self, table, spec):
        h, m = float(spec["hidden"]), float(spec["intermediate"])
        kv_hidden = float(spec["kv_heads"] * (spec["hidden"] // spec["heads"]))
        self.attn = (2 * h * h * np.asarray(table["heads"], float) / spec["heads"]
                     + 2 * h * kv_hidden * np.asarray(table["kv_groups"], float) / spec["kv_heads"])
        self.mlp = 3 * h * m * np.asarray(table["mlp_dims"], float) / m
        self.L = self.attn.shape[0]
        self.dense_total = self.L * (2 * h * h + 2 * h * kv_hidden + 3 * h * m)

    def __call__(self, levels):
        """levels: (n, 2L) array, attention levels first; returns the reduction for each policy."""
        levels = np.atleast_2d(np.asarray(levels, dtype=np.int64))
        layers = np.arange(self.L)
        sparse = self.attn[layers, levels[:, :self.L]].sum(1) + self.mlp[layers, levels[:, self.L:]].sum(1)
        return np.clip(1.0 - sparse / self.dense_total, 0.0, 1.0)


