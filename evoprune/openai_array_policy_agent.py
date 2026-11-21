#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from evoprune.base_agent import Agent


DEFAULT_SYSTEM_INSTRUCTION = """
You are a policy search agent for structured model compression.
Your task is to propose the exact number of complete candidate policies requested in the task.

You are given evaluated policy examples from previous search steps.
Each example contains:
- a policy, represented by attn_levels and mlp_levels
- its evaluation metrics: kl, ppl, reward, flops_reduction

Your job is to infer the mapping from policy structure to performance.
Use the evaluated examples to learn which structural patterns tend to produce:
- lower kl
- lower ppl
- appropriate flops_reduction for the target compression setting
- higher reward

The examples in weak_examples are weak evaluated examples.
The examples in strong_examples are strong evaluated examples.
Compare them carefully and infer what differs structurally between weak and strong policies.
If recent_history is provided, use it to understand the recent search trajectory, detect repeated patterns, and notice whether the search appears to be stagnating or over-exploring a narrow structural family.
If the search appears stuck in a local minimum or trapped in a repetitive structural neighborhood, deliberately propose some candidates that explore meaningfully different pruning patterns instead of making only small variations around recent policies.

Use the observation to infer not just which policies are strong or weak, but why they may be strong or weak in terms of model behavior:
- which regions of depth appear more sensitive,
- whether attention or MLP can absorb more compression,
- whether preserving some layers or modules helps maintain quality,
- whether recent search history suggests over-explored or stalled structural patterns.

Prefer candidates that reflect a plausible pruning rationale, not only candidates that resemble previously strong templates.

Optimization goals:
- Lower kl is better.
- Lower ppl is better.
- Higher flops_reduction means stronger compression.
- Higher reward is better.

Rules:
1. Return valid JSON only.
2. Return exactly the number of policies requested in the task.
3. Each policy must contain two arrays: attn_levels and mlp_levels.
4. Each array must have the required number of integers for this run.
5. Every integer must respect the level bounds for this run.
6. Higher integer means higher compression level.
7. Use both weak and strong evaluated examples as the main optimization signal.
8. Do not output deltas or partial edits. Output complete policies only.
9. Do not simply copy historical policies. Propose new policies that are not identical to the historical ones.
10. Prefer promising policies, but maintain diversity across all outputs.
11. In reasoning, briefly explain what structural patterns you learned from the evaluated examples and how they influenced your proposals.
12. The evaluator's reward already handles compression-budget error. Do not assume that a later repair step will alter your policy.

When search_mode is "global_novelty":
- The goal is to discover structural regions outside every known species, not to refine one current species.
- Compare against known_species representatives and champions.
- Use clearly different macro strategies across the returned batch, such as module-allocation inversion, depth relocation, periodic/phase changes, or a structured restart.
- Avoid making only scattered +/-1 edits around an existing representative.
- Think through the alternatives internally, then return only a concise, auditable reasoning summary.
- Every policy must include a short novelty_strategy naming the structural strategy used.

When search_mode is "nursery_probe":
- The source lineage is structurally novel but has not earned a permanent species slot.
- Preserve a recognizable part of its macro allocation pattern in some candidates while trying
  targeted repairs suggested by its metrics and nearby active species.
- Include both conservative repairs and at least one larger structured variation.
- The goal is to test whether this lineage can improve, not merely to maximize distance.
- Explain the retained motif and attempted repairs in the concise reasoning summary.

Output schema:
{
  "reasoning": "short explanation",
  "policies": [
    {
      "attn_levels": [0, 0, 1, ...],
      "mlp_levels": [0, 1, 1, ...],
      "novelty_strategy": "required only for global_novelty mode"
    }
  ]
}
""".strip()


PATTERN_REACT_ROLES = (
    "supported_pattern",
    "contrastive_test",
    "competing_hypothesis",
    "structured_restart",
)


PATTERN_REACT_V1_SYSTEM_INSTRUCTION = """
You are the macro-pattern exploration component of an evolutionary search system
for structured model compression. A downstream mutation operator performs local
numerical optimization around every policy you propose. Your job is therefore to
test distinct high-level structural hypotheses, not to spend the batch on small
variations of one template.

Each evaluated example contains complete attn_levels and mlp_levels arrays plus
kl, ppl, reward, and flops_reduction. You may also receive previous_experiments,
which links the previous round's hypotheses to the results of the direct LLM
policy and its locally mutated child.

COMPRESSION-LEVEL SEMANTICS
- A larger integer always means stronger compression and less retained capacity.
- Level 0 means least compressed and most protected; the maximum valid level
  means most compressed and least protected.
- A region described as protected must use lower levels than a region described
  as compressed. Before returning each experiment, verify that its policy arrays
  agree with its protected_regions and compression_regions descriptions.

OBSERVE
- Compare strong and weak examples structurally, prioritizing comparisons at
  similar flops_reduction.
- Identify depth allocation, protected and compressed regions, attention-versus-
  MLP allocation, repeated recent patterns, and evidence for or against previous
  hypotheses.
- Separate evaluated evidence from uncertain hypotheses and untested priors.
- A low-KL or low-PPL policy below the compression target is useful evidence, but
  not a successful solution. Test relocation of the missing compression budget
  instead of uniformly increasing every level.

UPDATE HYPOTHESES
- Decide whether relevant previous hypotheses should continue, be revised, or be
  replaced. One relative winner from one round does not establish a global rule.
- A numeric range such as "levels 3 to 10" is not a structural hypothesis. State
  where compression is placed across depth and between attention and MLP.
- Do not let all four actions collapse onto the current best pattern. Local
  numerical refinement is the mutation operator's responsibility.

ACT
Return exactly four experiments in this exact role order:
1. supported_pattern: test the best-supported macro pattern.
2. contrastive_test: test a structural difference separating strong and weak
   examples.
3. competing_hypothesis: test a plausible macro pattern different from the
   supported pattern.
4. structured_restart: test an underexplored coherent structural region outside
   the dominant recent family; scattered +/-1 edits are not a restart.

When only one initialization example exists, treat all four actions as uncertain
prior probes while retaining the four roles. Across the batch, policies must
differ meaningfully in at least two of: attention depth shape, MLP depth shape,
protected region, primary compression region, and attention-versus-MLP allocation.
Do not hard-code assumptions such as early layers always being sensitive; update
such claims from evaluated evidence.

REPORT
Return concise, auditable evidence summaries rather than hidden chain-of-thought.
For every experiment, state its structural hypothesis, evidence basis, pattern
descriptor, prediction, and falsification condition, followed by the complete
policy arrays.

Optimization goals:
- lower kl and ppl are better;
- higher reward is better;
- account for the target flops_reduction;
- reward already handles budget error and no later repair changes the policy.

Return valid JSON only with this schema:
{
  "observation_summary": {
    "supported_evidence": ["short statement"],
    "negative_evidence": ["short statement"],
    "budget_confounders": ["short statement"],
    "trajectory_state": "exploring|improving|stagnating|collapsed"
  },
  "experiments": [
    {
      "hypothesis_id": "short unique id",
      "role": "supported_pattern|contrastive_test|competing_hypothesis|structured_restart",
      "hypothesis": "concise structural claim",
      "status": "continue|revise|new",
      "evidence_basis": "concise reference to observations",
      "pattern_descriptor": {
        "attn_depth_shape": "rising|falling|hump|valley|periodic|block|mixed",
        "mlp_depth_shape": "rising|falling|hump|valley|periodic|block|mixed",
        "protected_regions": "concise description",
        "compression_regions": "concise description",
        "module_allocation": "attention_heavier|mlp_heavier|balanced|depth_dependent"
      },
      "prediction": "observable expected result",
      "falsification_condition": "observable result that weakens the hypothesis",
      "attn_levels": [0, 0, 1],
      "mlp_levels": [0, 1, 1]
    }
  ]
}

Return exactly four experiments and exactly one instance of each role in the
specified order. Do not add a separate long reasoning field.
""".strip()


def build_array_policy_system_instruction(
    policy_bounds: Dict[str, Any],
    prompt_mode: str = "standard",
) -> str:
    constraints = compact_array_policy_constraints(policy_bounds)
    num_layers = int(constraints["num_layers"])
    default_bounds = constraints.get("default_bounds")
    overrides = constraints.get("layer_overrides") or {}

    if default_bounds is None:
        bounds_text = "No valid pruning levels were detected in the database."
    else:
        bounds_text = (
            f"For this run, each attn_levels array must contain exactly {num_layers} integers, "
            f"and each mlp_levels array must contain exactly {num_layers} integers.\n"
            f"Default valid attention level range: {int(default_bounds['attn_min'])}.."
            f"{int(default_bounds['attn_max'])} inclusive.\n"
            f"Default valid MLP level range: {int(default_bounds['mlp_min'])}.."
            f"{int(default_bounds['mlp_max'])} inclusive."
        )
        if overrides:
            bounds_text += (
                "\nSome layers have non-default bounds. Use these exact per-layer overrides:\n"
                + json.dumps(overrides, ensure_ascii=False, sort_keys=True)
            )

    instructions = {
        "standard": DEFAULT_SYSTEM_INSTRUCTION,
        "pattern_react_v1": PATTERN_REACT_V1_SYSTEM_INSTRUCTION,
    }
    if prompt_mode not in instructions:
        raise ValueError(f"unsupported array-agent prompt mode: {prompt_mode}")
    return f"{instructions[prompt_mode]}\n\nRun-specific policy bounds:\n{bounds_text}"


@dataclass
class ArrayPolicyProposal:
    attn_levels: List[int]
    mlp_levels: List[int]
    novelty_strategy: Optional[str] = None
    pattern_experiment: Optional[Dict[str, Any]] = None


@dataclass
class ArrayPolicyAgentResult:
    reasoning: str
    policies: List[ArrayPolicyProposal]
    raw_response: str
    prompt_json: Dict[str, Any]
    validation_errors: List[str]
    observation_summary: Optional[Dict[str, Any]] = None


class OpenAIArrayPolicyAgent(Agent):
    def __init__(
        self,
        model_name: Optional[str] = None,
        system_instruction: Optional[str] = None,
        temperature: float = 0.5,
        max_output_tokens: int = 8192,
        max_retries: int = 3,
        base_delay_s: int = 2,
        api_key: Optional[str] = None,
        seed: Optional[int] = None,
        provider: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
        prompt_mode: str = "standard",
    ):
        super().__init__(
            model_name=model_name,
            system_instruction=system_instruction or DEFAULT_SYSTEM_INSTRUCTION,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            max_retries=max_retries,
            base_delay_s=base_delay_s,
            api_key=api_key,
            seed=seed,
            provider=provider,
            reasoning_effort=reasoning_effort,
        )
        self.prompt_mode = prompt_mode

    def propose_policies(
        self,
        observation: Dict[str, Any],
        policy_bounds: Dict[str, Any],
        num_policies: int = 8,
        search_mode: str = "standard",
    ) -> ArrayPolicyAgentResult:
        if num_policies <= 0:
            raise ValueError("num_policies must be positive")
        policy_constraints = compact_array_policy_constraints(policy_bounds)
        prompt_mode = getattr(self, "prompt_mode", "standard")
        prompt_json = {
            "task": (
                f"Run one Pattern-ReAct macro-search turn and propose exactly "
                f"{num_policies} complete policy experiments."
                if prompt_mode == "pattern_react_v1"
                else f"Propose exactly {num_policies} complete policy candidates in array format."
            ),
            "num_policies": int(num_policies),
            "search_mode": search_mode,
            "policy_constraints": policy_constraints,
            "observation": observation,
        }
        message = json.dumps(
            prompt_json,
            ensure_ascii=False,
            separators=(",", ":"),
        )
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
        reasoning, policies, validation_errors, observation_summary = validate_array_policy_response(
            response_json,
            policy_bounds,
            expected_num_policies=num_policies,
            require_novelty_strategy=search_mode == "global_novelty",
            prompt_mode=prompt_mode,
        )
        policies = policies[:num_policies]
        if prompt_mode == "pattern_react_v1" and (
            validation_errors or len(policies) != num_policies
        ):
            details = "; ".join(validation_errors) or (
                f"expected {num_policies} valid experiments but received {len(policies)}"
            )
            raise ValueError(f"invalid Pattern-ReAct response: {details}")
        return ArrayPolicyAgentResult(
            reasoning=reasoning,
            policies=policies,
            raw_response=raw_response,
            prompt_json=prompt_json,
            validation_errors=validation_errors,
            observation_summary=observation_summary,
        )


def compact_array_policy_constraints(policy_bounds: Dict[str, Any]) -> Dict[str, Any]:
    layers = policy_bounds.get("layers", {})
    num_layers = int(policy_bounds.get("num_layers", 0))
    if not layers:
        return {
            "policy_type": "array_level_bounds",
            "num_layers": num_layers,
            "default_bounds": None,
            "layer_overrides": {},
        }

    first_key = sorted(layers.keys(), key=int)[0]
    default_bounds = {
        "attn_min": int(layers[first_key]["attn_min"]),
        "attn_max": int(layers[first_key]["attn_max"]),
        "mlp_min": int(layers[first_key]["mlp_min"]),
        "mlp_max": int(layers[first_key]["mlp_max"]),
    }
    overrides = {}
    for layer_key, bounds in layers.items():
        compact_bounds = {
            "attn_min": int(bounds["attn_min"]),
            "attn_max": int(bounds["attn_max"]),
            "mlp_min": int(bounds["mlp_min"]),
            "mlp_max": int(bounds["mlp_max"]),
        }
        if compact_bounds != default_bounds:
            overrides[layer_key] = compact_bounds

    return {
        "policy_type": "array_level_bounds",
        "num_layers": num_layers,
        "default_bounds": default_bounds,
        "layer_overrides": overrides,
    }


def _validate_level_array(
    name: str,
    value: Any,
    policy_bounds: Dict[str, Any],
    level_kind: str,
) -> List[int]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be a list")
    num_layers = int(policy_bounds["num_layers"])
    if len(value) != num_layers:
        raise ValueError(f"{name} must contain exactly {num_layers} integers")
    try:
        levels = [int(item) for item in value]
    except Exception as exc:
        raise ValueError(f"{name} must contain integers only") from exc
    min_key = f"{level_kind}_min"
    max_key = f"{level_kind}_max"
    for idx, item in enumerate(levels):
        bounds = policy_bounds["layers"][str(idx)]
        min_level = int(bounds[min_key])
        max_level = int(bounds[max_key])
        if item < min_level or item > max_level:
            raise ValueError(
                f"{name}[{idx}]={item} must be between {min_level} and {max_level} inclusive"
            )
    return levels


def validate_array_policy_response(
    response_json: Dict[str, Any],
    policy_bounds: Dict[str, Any],
    expected_num_policies: int = 8,
    require_novelty_strategy: bool = False,
    prompt_mode: str = "standard",
) -> tuple[str, List[ArrayPolicyProposal], List[str], Optional[Dict[str, Any]]]:
    if not isinstance(response_json, dict):
        raise ValueError("agent response must be a dict")
    observation_summary = None
    if prompt_mode == "pattern_react_v1":
        observation_summary = response_json.get("observation_summary")
        if not isinstance(observation_summary, dict):
            raise ValueError("Pattern-ReAct response must contain an observation_summary object")
        for key in ("supported_evidence", "negative_evidence", "budget_confounders"):
            if not isinstance(observation_summary.get(key), list):
                raise ValueError(f"observation_summary.{key} must be a list")
        if observation_summary.get("trajectory_state") not in {
            "exploring",
            "improving",
            "stagnating",
            "collapsed",
        }:
            raise ValueError("observation_summary.trajectory_state is invalid")
        reasoning = json.dumps(
            observation_summary,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        policies_json = response_json.get("experiments")
        policies_key = "experiments"
    else:
        reasoning = str(response_json.get("reasoning", "")).strip()
        if not reasoning:
            raise ValueError("agent response must contain a non-empty 'reasoning'")
        policies_json = response_json.get("policies")
        policies_key = "policies"
    if not isinstance(policies_json, list):
        raise ValueError(f"agent response must contain a '{policies_key}' list")

    policies: List[ArrayPolicyProposal] = []
    validation_errors: List[str] = []
    if len(policies_json) != expected_num_policies:
        validation_errors.append(
            f"expected {expected_num_policies} policies but received {len(policies_json)}; "
            "invalid entries will be skipped"
        )
    for idx, item in enumerate(policies_json):
        try:
            if not isinstance(item, dict):
                raise ValueError(f"policy {idx} must be an object")
            attn_levels = _validate_level_array(
                f"policies[{idx}].attn_levels",
                item.get("attn_levels"),
                policy_bounds,
                "attn",
            )
            mlp_levels = _validate_level_array(
                f"policies[{idx}].mlp_levels",
                item.get("mlp_levels"),
                policy_bounds,
                "mlp",
            )
            novelty_strategy = str(item.get("novelty_strategy", "")).strip() or None
            if require_novelty_strategy and novelty_strategy is None:
                raise ValueError(
                    f"policies[{idx}].novelty_strategy is required in global_novelty mode"
                )
            pattern_experiment = None
            if prompt_mode == "pattern_react_v1":
                required_strings = (
                    "hypothesis_id",
                    "role",
                    "hypothesis",
                    "status",
                    "evidence_basis",
                    "prediction",
                    "falsification_condition",
                )
                values = {
                    key: str(item.get(key, "")).strip()
                    for key in required_strings
                }
                missing = [key for key, value in values.items() if not value]
                if missing:
                    raise ValueError(
                        f"experiments[{idx}] has empty fields: {', '.join(missing)}"
                    )
                if values["role"] not in PATTERN_REACT_ROLES:
                    raise ValueError(f"experiments[{idx}].role is invalid")
                if values["status"] not in {"continue", "revise", "new"}:
                    raise ValueError(f"experiments[{idx}].status is invalid")
                descriptor = item.get("pattern_descriptor")
                descriptor_keys = (
                    "attn_depth_shape",
                    "mlp_depth_shape",
                    "protected_regions",
                    "compression_regions",
                    "module_allocation",
                )
                if not isinstance(descriptor, dict) or any(
                    not str(descriptor.get(key, "")).strip()
                    for key in descriptor_keys
                ):
                    raise ValueError(
                        f"experiments[{idx}].pattern_descriptor is incomplete"
                    )
                pattern_experiment = {
                    **values,
                    "pattern_descriptor": {
                        key: str(descriptor[key]).strip()
                        for key in descriptor_keys
                    },
                }
            policies.append(
                ArrayPolicyProposal(
                    attn_levels=attn_levels,
                    mlp_levels=mlp_levels,
                    novelty_strategy=novelty_strategy,
                    pattern_experiment=pattern_experiment,
                )
            )
        except Exception as exc:
            validation_errors.append(str(exc))

    if not policies:
        error_summary = "; ".join(validation_errors) if validation_errors else "no valid policies returned"
        raise ValueError(f"agent response contained no valid policies: {error_summary}")
    if prompt_mode == "pattern_react_v1" and len(policies) == 4:
        roles = [item.pattern_experiment["role"] for item in policies]
        if roles != list(PATTERN_REACT_ROLES):
            validation_errors.append(
                "Pattern-ReAct experiments must use each required role once in the specified order"
            )
        hypothesis_ids = [item.pattern_experiment["hypothesis_id"] for item in policies]
        if len(set(hypothesis_ids)) != len(hypothesis_ids):
            validation_errors.append("Pattern-ReAct hypothesis_id values must be unique")
    return reasoning, policies, validation_errors, observation_summary
