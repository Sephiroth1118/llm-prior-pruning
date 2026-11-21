"""Helpers to build random feasible level policies (used for the shared initial designs)."""
import random
from typing import Any, Dict, List, Optional, Tuple

from evoprune.data_utils import get_data
from evoprune.EA import Policy


def get_eval_data(dataset_name: str, tokens: int, sequence_length: int, tokenizer,
                  max_samples: Optional[int] = None):
    eval_data = get_data(dataset_name, tokens, sequence_length, tokenizer, train=False)
    if max_samples is not None:
        eval_data = eval_data[:max_samples]
    return eval_data


def policy_arrays_from_db_policy(policy_json: Dict[str, Any], num_layers: int) -> Dict[str, List[int]]:
    layers = policy_json.get("layers", {})
    return {
        "attn_levels": [int(layers[str(i)]["attn_level"]) for i in range(num_layers)],
        "mlp_levels": [int(layers[str(i)]["mlp_level"]) for i in range(num_layers)],
    }


def build_random_initial_policy(
    policy_bounds: Dict[str, Any],
    seed: int = 0,
    *,
    cost_fn=None,
    target: Optional[float] = None,
    tolerance: float = 0.001,
    max_attempts: int = 10_000,
) -> Dict[str, Any]:
    """Sample uniformly, optionally conditioning on a narrow proxy-budget band."""

    if (cost_fn is None) != (target is None):
        raise ValueError("cost_fn and target must be provided together")
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if max_attempts <= 0:
        raise ValueError("max_attempts must be positive")

    rng = random.Random(seed)
    num_layers = int(policy_bounds["num_layers"])
    for _ in range(max_attempts):
        layers = {}
        for layer_idx in range(num_layers):
            bounds = policy_bounds["layers"][str(layer_idx)]
            layers[str(layer_idx)] = {
                "attn_level": rng.randint(int(bounds["attn_min"]), int(bounds["attn_max"])),
                "mlp_level": rng.randint(int(bounds["mlp_min"]), int(bounds["mlp_max"])),
            }
        result = {
            "policy_type": "db_level_policy",
            "num_layers": num_layers,
            "layers": layers,
        }
        if cost_fn is None:
            return result
        arrays = policy_arrays_from_db_policy(result, num_layers)
        policy = Policy.from_arrays(arrays["attn_levels"], arrays["mlp_levels"])
        if abs(float(cost_fn(policy)) - float(target)) <= tolerance:
            return result

    raise ValueError(
        f"could not sample a random policy within {tolerance} of target {target} "
        f"after {max_attempts} attempts"
    )


