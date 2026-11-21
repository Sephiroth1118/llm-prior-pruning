"""GP-guided mutation (EA with GP-guided mutation), using the frozen GP program evoprune/gp_rf_program.py.

The program (FAMILY="gp", GUIDANCE=True, corrected transfer rule with shrink prior) is imported as a module and called
in-process.

Schedule: first fit once `warmup` evaluations exist, refit every `interval` evaluations, each fit on the trailing
`window` parent->child transitions. Training rows are built with evoprune.EA.program_kernel.training_data() from a
history view of our entries, so gains, reward deltas and transfer-kind labels follow the same definitions.
"""
from dataclasses import replace

from common import REPO, load_module

PROGRAM_PATH = REPO / "evoprune" / "gp_rf_program.py"
REFERENCE = {"ridge": 1.0, "temperature": 1.0, "min_weight": 0.25, "max_weight": 4.0}


def _kernel_api():
    from evoprune.EA.operators import _proxy_level_costs
    from evoprune.EA.mutation_kernel import MutationKernel, TRANSFER_KINDS
    from evoprune.EA.program_kernel import training_data, validate_kernel, API_VERSION
    return _proxy_level_costs, MutationKernel, TRANSFER_KINDS, training_data, validate_kernel, API_VERSION


class GPGuide:
    def __init__(self, bounds, cost_fn, policy_cls, layers, *, warmup=80, interval=80, window=80, minimum_pairs=8,
                 min_weight=0.25, max_weight=4.0):
        self.bounds, self.cost_fn, self.policy_cls, self.layers = bounds, cost_fn, policy_cls, layers
        self.reference = {**REFERENCE, "min_weight": float(min_weight), "max_weight": float(max_weight)}
        self.warmup, self.interval, self.window, self.minimum_pairs = warmup, interval, window, minimum_pairs
        self.program = load_module("gp_program_corrected", PROGRAM_PATH)
        (self._level_costs, self.MutationKernel, self.TRANSFER_KINDS,
         self._training_data, self._validate_kernel, self.api_version) = _kernel_api()
        self.schema, self.last_fit, self.version, self.log = None, None, 0, []

    def observe(self, policy):
        """Schema from live bounds, as the program adapter does (once, from the first parent)."""
        if self.schema is not None:
            return
        parent = self.policy_cls.from_arrays(list(policy[:self.layers]), list(policy[self.layers:]))
        costs = self._level_costs(parent, self.bounds, self.cost_fn)
        refs = list(self.bounds.iter_genes())
        level_costs = [costs.get(ref, 0.0) for ref in refs]
        by_module = {}
        for ref, cost in zip(refs, level_costs):
            by_module.setdefault(ref.module, []).append(cost)
        expensive = max(by_module, key=lambda m: sum(by_module[m]) / len(by_module[m]))
        self.schema = {"api_version": self.api_version, "num_layers": self.layers,
                       "modules": [ref.module for ref in refs],
                       "minima": [self.bounds.get(ref).minimum for ref in refs],
                       "maxima": [self.bounds.get(ref).maximum for ref in refs],
                       "level_costs": level_costs, "target_flops_reduction": 0.5,
                       "transfer_kinds": list(self.TRANSFER_KINDS),
                       "normalization": "positive preferences normalized over each CURRENT legal gene set",
                       "expensive_module": expensive, "reference": dict(self.reference)}

    def due(self, evaluations):
        if evaluations < self.warmup:
            return False
        return self.last_fit is None or evaluations - self.last_fit >= self.interval

    def history_view(self, entries):
        """Our entries in the field layout training_data() expects."""
        byid = {e["id"]: e for e in entries}
        view = []
        for e in entries:
            if e.get("error"):
                continue
            item = {"history_id": e["id"], "order": e["id"], "attn_levels": list(e["policy"][:self.layers]),
                    "mlp_levels": list(e["policy"][self.layers:]), "kl": e["kl"], "reward": e["reward"]}
            parent = byid.get(e.get("parent")) if e.get("parent") is not None else None
            if parent is not None and e.get("mutation_edits") is not None:
                item.update(mutation_parent_history_id=parent["id"], mutation_parent_reward=parent["reward"],
                            mutation_parent_policy={"attn_levels": list(parent["policy"][:self.layers]),
                                                    "mlp_levels": list(parent["policy"][self.layers:])},
                            mutation_edits=e["mutation_edits"])
            view.append(item)
        return view

    def fit(self, entries, evaluations):
        if self.schema is None:
            self.observe(entries[0]["policy"])
        data, ids = self._training_data(self.history_view(entries), self.layers, self.window, self.schema["expensive_module"])
        self.last_fit = evaluations
        if len(data) < max(2, self.minimum_pairs):
            self.log.append({"evaluations": evaluations, "records": len(data), "fitted": False})
            return None
        state = self.program.fit_model(data, self.schema)
        value = self.program.make_kernel(state, None, self.schema)
        self.version += 1
        kernel = self._validate_kernel(value, self.layers, source="gp_program", version=self.version,
                                       hypothesis="isolated history-adaptive program (corrected transfer rule)",
                                       metadata={"sample_count": len(data), "fit_history_ids": ids})
        k = state["kernel"]
        self.log.append({"evaluations": evaluations, "records": len(data), "fitted": True, "version": self.version,
                         "fit": {key: state["fit"][key] for key in ("length", "amplitude", "noise")},
                         "target_mean": state["mean"], "target_scale": state["scale"],
                         "transfer_weights": k["transfer_weights"], "transfer_diagnostics": state.get("transfer_diagnostics"),
                         "mean_weights": {key: sum(k[key]) / len(k[key]) for key in ("attn_compress", "attn_decompress", "mlp_compress", "mlp_decompress")},
                         "kernel": {key: k[key] for key in ("attn_compress", "attn_decompress", "mlp_compress", "mlp_decompress")},
                         "warnings": state.get("warnings")})
        return kernel
