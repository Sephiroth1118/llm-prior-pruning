#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evoprune.base_agent import Agent

from .policy import build_policy_json_from_moe_group_levels, validate_moe_group_levels_shape


DEFAULT_SYSTEM_INSTRUCTION = """
You are a policy search agent for MoE structured model compression.
Your task is to propose 8 complete candidate policies.

You are given evaluated policy examples from previous search steps.
Each example contains:
- a policy, represented by moe_group_levels
- its evaluation metrics: kl, ppl, reward, flops_reduction

The model is a Mixture-of-Experts language model.
Only expert MLP modules are prunable.
Each layer has 4 fixed expert groups, ordered by importance:
- group 0: most important experts
- group 1: next 25%
- group 2: next 25%
- group 3: least important experts

Each layer has 4 integer pruning levels, one per group:
- 0 means no pruning
- higher integer means higher compression level

Your job is to infer the mapping from layer-wise MoE MLP pruning structure to performance.
Use the evaluated examples to learn which structural patterns tend to produce:
- lower kl
- lower ppl
- appropriate flops_reduction for the target compression setting
- higher reward

The examples in weak_examples are weak evaluated examples.
The examples in strong_examples are strong evaluated examples.
Compare them carefully and infer what differs structurally between weak and strong policies.
If recent_history is provided, use it to understand the recent search trajectory, detect repeated patterns, and notice whether the search appears to be stagnating or over-exploring a narrow structural family.
If the search appears stuck in a local minimum or trapped in a repetitive structural neighborhood, deliberately propose some candidates that explore meaningfully different layer-wise pruning patterns instead of making only small variations around recent policies.

Pay attention to MoE-specific structure:
- some layers may tolerate heavier expert-MLP pruning than others,
- early and late layers may have different sensitivity,
- group 0 should usually be pruned less than group 3,
- if history suggests it is safe, lower-importance groups can often be pruned more.

Optimization goals:
- Lower kl is better.
- Lower ppl is better.
- Higher flops_reduction means stronger compression.
- Higher reward is better.

Rules:
1. Return valid JSON only.
2. Return exactly 8 policies.
3. Each policy must contain moe_group_levels.
4. moe_group_levels must contain exactly num_layers rows.
5. Each row must contain exactly 4 integers.
6. Every integer must respect the provided bounds for that layer.
7. Use both weak and strong evaluated examples as the main optimization signal.
8. Do not output deltas or partial edits. Output complete policies only.
9. Do not simply copy historical policies. Propose new policies that are not identical to the historical ones.
10. Prefer promising policies, but maintain diversity across the 8 outputs.
11. In reasoning, briefly explain what structural patterns you learned from the evaluated examples and how they influenced your proposals.

Output schema:
{
  "reasoning": "short explanation",
  "policies": [
    {
      "moe_group_levels": [
        [0, 1, 2, 3],
        [1, 1, 2, 3]
      ]
    }
  ]
}
""".strip()


def build_moe_array_policy_system_instruction(policy_bounds: Dict[str, Any]) -> str:
    constraints = compact_moe_array_policy_constraints(policy_bounds)
    num_layers = int(constraints["num_layers"])
    default_bounds = constraints.get("default_bounds")
    layer_overrides = constraints.get("layer_overrides") or {}

    if default_bounds is None:
        bounds_text = "No valid expert pruning levels were detected in the database."
    else:
        bounds_text = (
            f"For this run, each moe_group_levels array must contain exactly {num_layers} rows.\n"
            f"Each row must contain exactly {int(constraints['num_groups'])} integers.\n"
            f"Default valid layer MLP level range: {int(default_bounds['mlp_min'])}.."
            f"{int(default_bounds['mlp_max'])} inclusive."
        )
        if layer_overrides:
            bounds_text += (
                "\nSome layers have non-default expert counts or bounds. "
                "Use these exact per-layer overrides:\n"
                + json.dumps(layer_overrides, ensure_ascii=False, sort_keys=True)
            )

    return f"{DEFAULT_SYSTEM_INSTRUCTION}\n\nRun-specific policy bounds:\n{bounds_text}"


@dataclass
class MoEArrayPolicyProposal:
    moe_group_levels: List[List[int]]

    def to_policy_json(self, policy_bounds: Dict[str, Any]) -> Dict[str, object]:
        expert_groups = policy_bounds["expert_groups"]
        return build_policy_json_from_moe_group_levels(policy_bounds, expert_groups, self.moe_group_levels)


@dataclass
class MoEArrayPolicyAgentResult:
    reasoning: str
    policies: List[MoEArrayPolicyProposal]
    raw_response: str
    prompt_json: Dict[str, Any]
    validation_errors: List[str]


class OpenAIMoEArrayPolicyAgent(Agent):
    def __init__(
        self,
        model_name: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.5,
        max_output_tokens: int = 8192,
        max_retries: int = 3,
        base_delay_s: int = 2,
        api_key: Optional[str] = None,
    ):
        super().__init__(
            model_name=model_name,
            system_instruction=system_instruction or DEFAULT_SYSTEM_INSTRUCTION,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            base_delay_s=base_delay_s,
            api_key=api_key,
        )

    def propose_policies(
        self,
        observation: Dict[str, Any],
        policy_bounds: Dict[str, Any],
    ) -> MoEArrayPolicyAgentResult:
        policy_constraints = compact_moe_array_policy_constraints(policy_bounds)
        prompt_json = {
            "task": "Propose 8 complete MoE fixed-group pruning policy candidates in array format.",
            "policy_constraints": policy_constraints,
            "observation": observation,
        }
        message = json.dumps(prompt_json, ensure_ascii=False)
        raw_response = ""
        for attempt in range(self.max_retries + 1):
            raw_response = self.send(message=message, json_mode=True)
            if raw_response.strip():
                break
            if attempt >= self.max_retries:
                raise ValueError("agent returned an empty response")
            wait_time = self.base_delay_s * (2 ** attempt)
            print(
                f"[WARNING] Agent returned an empty response. "
                f"Retrying {attempt + 1}/{self.max_retries} in {wait_time}s."
            )
            time.sleep(wait_time)
        response_json = json.loads(raw_response)
        reasoning, policies, validation_errors = validate_moe_array_policy_response(response_json, policy_bounds)
        return MoEArrayPolicyAgentResult(
            reasoning=reasoning,
            policies=policies,
            raw_response=raw_response,
            prompt_json=prompt_json,
            validation_errors=validation_errors,
        )


def compact_moe_array_policy_constraints(policy_bounds: Dict[str, Any]) -> Dict[str, Any]:
    layers = policy_bounds.get("layers", {})
    num_layers = int(policy_bounds.get("num_layers", 0))
    if not layers:
        return {
            "policy_type": "moe_array_level_bounds",
            "num_layers": num_layers,
            "num_groups": 4,
            "default_bounds": None,
            "layer_overrides": {},
        }

    first_key = sorted(layers.keys(), key=int)[0]
    first_layer = layers[first_key]
    first_expert_key = sorted(first_layer["experts"].keys(), key=int)[0]
    default_bounds = {
        "mlp_min": int(first_layer["experts"][first_expert_key]["min_level"]),
        "mlp_max": int(first_layer["experts"][first_expert_key]["max_level"]),
    }

    overrides = {}
    for layer_key, layer_obj in layers.items():
        mins = {int(bounds["min_level"]) for bounds in layer_obj["experts"].values()}
        maxs = {int(bounds["max_level"]) for bounds in layer_obj["experts"].values()}
        if len(mins) != 1 or len(maxs) != 1:
            raise ValueError(
                f"Layer {layer_key} has heterogeneous expert bounds, which is incompatible with "
                "layer-wise shared sparsity policies."
            )
        layer_bounds = {
            "mlp_min": int(next(iter(mins))),
            "mlp_max": int(next(iter(maxs))),
            "num_experts": int(layer_obj["num_experts"]),
        }
        if (
            layer_bounds["mlp_min"] != default_bounds["mlp_min"]
            or layer_bounds["mlp_max"] != default_bounds["mlp_max"]
            or layer_bounds["num_experts"] != int(first_layer["num_experts"])
        ):
            overrides[layer_key] = layer_bounds

    return {
        "policy_type": "moe_array_level_bounds",
        "num_layers": num_layers,
        "num_groups": int(policy_bounds["expert_groups"]["num_groups"]),
        "default_bounds": default_bounds,
        "layer_overrides": overrides,
    }


def _bounds_for_layer(policy_bounds: Dict[str, Any], layer_idx: int) -> Dict[str, int]:
    layer_obj = policy_bounds["layers"][str(layer_idx)]
    first_expert_key = sorted(layer_obj["experts"].keys(), key=int)[0]
    expert_obj = layer_obj["experts"][first_expert_key]
    return {
        "mlp_min": int(expert_obj["min_level"]),
        "mlp_max": int(expert_obj["max_level"]),
    }


def _validate_moe_group_levels(value: Any, policy_bounds: Dict[str, Any]) -> tuple[List[List[int]], List[str]]:
    num_layers = int(policy_bounds["num_layers"])
    num_groups = int(policy_bounds["expert_groups"]["num_groups"])
    validated, warnings = validate_moe_group_levels_shape(value, num_layers, num_groups)
    for layer_idx, row in enumerate(validated):
        bounds = _bounds_for_layer(policy_bounds, layer_idx)
        for group_idx, level in enumerate(row):
            if level < bounds["mlp_min"] or level > bounds["mlp_max"]:
                raise ValueError(
                    f"moe_group_levels[{layer_idx}][{group_idx}]={level} must be between "
                    f"{bounds['mlp_min']} and {bounds['mlp_max']} inclusive"
                )
    return validated, warnings


def validate_moe_array_policy_response(
    response_json: Dict[str, Any],
    policy_bounds: Dict[str, Any],
) -> tuple[str, List[MoEArrayPolicyProposal], List[str]]:
    if not isinstance(response_json, dict):
        raise ValueError("agent response must be a dict")
    reasoning = str(response_json.get("reasoning", "")).strip()
    if not reasoning:
        raise ValueError("agent response must contain a non-empty 'reasoning'")

    policies_json = response_json.get("policies")
    if not isinstance(policies_json, list):
        raise ValueError("agent response must contain a 'policies' list")

    policies: List[MoEArrayPolicyProposal] = []
    validation_errors: List[str] = []
    if len(policies_json) != 8:
        validation_errors.append(
            f"expected 8 policies but received {len(policies_json)}; invalid entries will be skipped"
        )
    for idx, item in enumerate(policies_json):
        try:
            if not isinstance(item, dict):
                raise ValueError(f"policy {idx} must be an object")
            moe_group_levels, clamp_warnings = _validate_moe_group_levels(
                item.get("moe_group_levels"),
                policy_bounds,
            )
            policies.append(MoEArrayPolicyProposal(moe_group_levels=moe_group_levels))
            validation_errors.extend(clamp_warnings)
        except Exception as exc:
            validation_errors.append(str(exc))

    if not policies:
        error_summary = "; ".join(validation_errors) if validation_errors else "no valid policies returned"
        raise ValueError(f"agent response contained no valid policies: {error_summary}")
    return reasoning, policies, validation_errors
