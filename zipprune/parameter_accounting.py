"""Separate allocated tensor slots, nonzero values, and retained structures.

Structural counts cover q/k/v/o attention and gated MLP projections (including
MoE experts). Other parameters, e.g. embeddings and norms, retain their shapes.
These counts are not a measurement of memory consumption or inference latency.
"""

import torch


PARAMETER_COUNT_VERSION = 2


@torch.no_grad()
def count_parameters(model, metric="total"):
    if metric not in {"total", "nonzero", "structured"}:
        raise ValueError(f"Unknown parameter metric: {metric}")
    parameters = list(model.parameters())  # Tied weights counted once.
    if any(p.is_meta for p in parameters):
        raise ValueError("Cannot count parameters while weights are on meta device")
    counts = {
        "Total": sum(p.numel() for p in parameters),
        "Trainable": sum(p.numel() for p in parameters if p.requires_grad),
        "NonZero": sum(int(torch.count_nonzero(p).item()) for p in parameters),
    }
    if metric == "structured":
        retained = {}
        matched = 0

        def record(parameter, count):
            if parameter is None:
                return
            key = id(parameter)
            if key in retained and retained[key] != count:
                raise ValueError("Conflicting structural counts for a shared parameter")
            retained[key] = int(count)

        def projection(linear, width, input_axis=False):
            shape = linear.weight.shape
            axis = 1 if input_axis else 0
            if not 0 <= width <= shape[axis]:
                raise ValueError("Retained width exceeds projection shape")
            record(linear.weight, width * shape[1 - axis])
            bias = getattr(linear, "bias", None)
            if bias is not None:
                record(bias, bias.numel() if input_axis else width)

        for module in model.modules():
            if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj")):
                matched += 1
                head_dim = getattr(module, "head_dim", None)
                if head_dim is None:
                    head_dim = getattr(getattr(model, "config", None), "head_dim", None)
                if not head_dim or head_dim <= 0:
                    raise ValueError("Structural attention count requires the actual head_dim")
                head_dim = int(head_dim)
                q_width = module.q_proj.weight.shape[0]
                kv_width = module.k_proj.weight.shape[0]
                if (q_width % head_dim or kv_width % head_dim or
                        module.v_proj.weight.shape[0] != kv_width or
                        module.o_proj.weight.shape[1] != q_width):
                    raise ValueError("Inconsistent Q/K/V/O projection shapes")
                q_heads, kv_heads = q_width // head_dim, kv_width // head_dim
                if kv_heads <= 0 or q_heads % kv_heads:
                    raise ValueError("Unsupported query-to-KV head grouping")
                weight = module.o_proj.weight.detach().float()
                active = weight.reshape(weight.shape[0], q_heads, head_dim).norm(dim=(0, 2)) > 1e-8
                active_kv = active.reshape(kv_heads, q_heads // kv_heads).any(dim=1)
                q_active = int(active.sum().item()) * head_dim
                kv_active = int(active_kv.sum().item()) * head_dim
                projection(module.q_proj, q_active)
                projection(module.k_proj, kv_active)
                projection(module.v_proj, kv_active)
                projection(module.o_proj, q_active, input_axis=True)
            if all(hasattr(module, name) for name in ("gate_proj", "up_proj", "down_proj")):
                matched += 1
                down = module.down_proj.weight.detach().float()
                if (module.gate_proj.weight.shape[0] != down.shape[1] or
                        module.up_proj.weight.shape[0] != down.shape[1]):
                    raise ValueError("Inconsistent gate/up/down projection shapes")
                active = int((down.norm(dim=0) > 1e-8).sum().item())
                projection(module.gate_proj, active)
                projection(module.up_proj, active)
                projection(module.down_proj, active, input_axis=True)
        if not matched:
            raise ValueError("No supported attention or gated MLP structures found")
        counts["EffectiveStructural"] = sum(retained.get(id(p), p.numel()) for p in parameters)
    key = {"total": "Total", "nonzero": "NonZero", "structured": "EffectiveStructural"}[metric]
    counts.update(Reported=counts[key], ReportedMetric=metric, SchemaVersion=PARAMETER_COUNT_VERSION)
    return counts
