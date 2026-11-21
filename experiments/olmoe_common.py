"""OLMoE-1B-7B-0924 under the fineweb_exact protocol: paths, gene order, decoder and the MoE prompt.

A gene is (layer, group): 16 layers x 4 router-importance groups of 16 experts, 11 levels, gene index = 4 * layer + group.
Compute is replaced by expert parameters (attention is not pruned), so the band [0.500, 0.502] applies to the parameters of
the experts. Cell = group + 4 * (layer in the second half): cells 0-3 are groups 1-4 of the first eight layers, cells 4-7
the same groups of the last eight, where group 1 holds the most important experts. The search code works on two arrays of
equal length, so genes 0-31 and 32-63 travel as Policy.attn_levels and Policy.mlp_levels; every gene costs the same, so the
labels carry no meaning here. SPI_MOE_ROOT moves all OLMoE outputs (default <work>/moe).
"""
import hashlib
import os
from pathlib import Path

import numpy as np

from common import DB_DIR, HF_DIR, REPO, WORK, read_json
import llm_templates
import spi_search

MOE = Path(os.environ.get("SPI_MOE_ROOT", WORK / "moe"))
MODEL = HF_DIR / "OLMoE-1B-7B-0924"
DB = DB_DIR / "olmoe"
GROUPS_FILE = MOE / "expert_groups_fineweb.json"
NAME, MODEL_KEY = "OLMoE-1B-7B-0924", "olmoe"
LAYERS, GROUPS, EXPERTS, HIDDEN, INTERMEDIATE = 16, 4, 64, 2048, 1024
PSEUDO_LAYERS = LAYERS * GROUPS // 2
PROMPT_FILE = REPO / "prompts" / "llm_template_system_moe.txt"    # the prompt system_prompt() builds (for reading)
ALPHABET = (0, .25, .5, 1, 2, 4)
RULE_WEIGHTS = [.25, .5, 1, 2] * 2
GPU_MEM_FRACTION = .42     # optional cap on the share of GPU memory one process may use (olmoe_prepare.py); lower it on shared GPUs


def level_bounds():
    return {"num_layers": PSEUDO_LAYERS,
            "layers": {str(i): {"attn_min": 0, "attn_max": 10, "mlp_min": 0, "mlp_max": 10} for i in range(PSEUDO_LAYERS)}}


class MoEDecoder(spi_search.StructureDecoder):
    """The structure decoder on 64 genes; only the map from genes to cells differs."""

    def __init__(self, remaining_costs, dense_total):
        super().__init__(remaining_costs, dense_total)
        layer, group = np.divmod(np.arange(len(remaining_costs)), GROUPS)
        self.cells = group + GROUPS * (layer >= LAYERS // 2)


def make_decoder():
    import torch
    cols = torch.load(DB / "metadata.pth", map_location="cpu", weights_only=False)["cols_removed_mlp"]
    per_gene = EXPERTS // GROUPS * 3 * HIDDEN * (INTERMEDIATE - np.asarray(cols, float))     # gate, up and down projections
    costs = np.tile(per_gene, (LAYERS * GROUPS, 1))
    return MoEDecoder(costs, costs[:, 0].sum())


def policy_json(policy):
    rows = [[int(policy[GROUPS * layer + group]) for group in range(GROUPS)] for layer in range(LAYERS)]
    return {"policy_type": "moe_group_level_policy", "num_layers": LAYERS, "num_groups": GROUPS, "moe_group_levels": rows}


def system_prompt():
    """SYSTEM_ORIG with the module order and the unit of the budget replaced; everything else is the prompt of the main runs."""
    text = llm_templates.SYSTEM_ORIG
    for old, new in (
        ("Use architectural priors and only the observations",
         "The model is a mixture of experts: every layer has 64 expert MLPs, of which the router activates 8 per token, and only\n"
         "the experts are pruned. Within each layer the experts are divided into four equal groups by router importance (usage\n"
         "frequency times mean routing probability on calibration text): group 1 is the most important, group 4 the least.\n"
         "Use architectural priors and only the observations"),
        ("The order is attention\ndepth quarters 1-4, then MLP depth quarters 1-4.",
         "The order is groups 1-4\nin the first eight layers, then groups 1-4 in the last eight layers."),
        ("RELATIVE REMOVAL FRACTIONS per\nmodule, NOT shares of the total removed FLOPs.",
         "RELATIVE REMOVAL FRACTIONS per\nexpert, NOT shares of the total removed parameters."),
        ("larger weights mean that module/depth should be pruned more.", "larger weights mean that group/depth should be pruned more."),
        ("removes 50.1% of the evaluator's dense FLOPs estimate,", "removes 50.1% of the expert parameters,"),
    ):
        if old not in text:
            raise ValueError(f"prompt anchor not found: {old[:40]!r}")
        text = text.replace(old, new)
    return text


def prompt_sha():
    return hashlib.sha256(system_prompt().encode()).hexdigest()


def job_path(name):
    return MOE / "jobs" / f"{name}.json"


def run_dir(name):
    return MOE / "runs" / name


def load_entries(name):
    return read_json(run_dir(name) / "results.json")["entries"]
