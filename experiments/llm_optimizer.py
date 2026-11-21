#!/usr/bin/env python3
"""The LLM as the optimizer.

The agent prompt is the one of the original array-policy agent (prompts/llm_optimizer_system.txt, also available as
evoprune.openai_array_policy_agent.DEFAULT_SYSTEM_INSTRUCTION): every call gets the ten best and ten worst evaluated
allocations and the twenty most recent ones (attn_levels, mlp_levels, kl, reward = -kl, flops_reduction) and returns
eight complete allocations with a short rationale in JSON. Two changes for a fair comparison: (1) the search starts from
the shared 16 random feasible allocations (the runner provides them); (2) the compute band is met by self-correction:
the prompt lists the dense-compute fraction removed at every level of every module, proposals outside [0.500, 0.502]
are sent back with their exact reduction for revision (at most twice; no divergence is evaluated and nothing is charged
to the budget), and proposals still outside are repaired by the greedy repair shared with TPE / DE / PSO / GA. Repeats of
evaluated allocations are perturbed and repaired as for those optimizers; unparsable slots are filled with a random
feasible allocation. Every step is logged per proposal.

Ablation switches (2x2 design): `anonymize` hides all structural meaning (generic integer-vector prompt, gene order
permuted per seed, no attention / MLP / depth names); `shuffle_scores` permutes the divergences across the evaluated
allocations before each call, so the feedback carries no information about which is better.

Providers: OpenAI (gpt-5-mini, gpt-5.6-luna) and DeepSeek (deepseek-v4-flash, OpenAI-compatible endpoint). Keys come
from the environment variables OPENAI_API_KEY and DEEPSEEK_API_KEY and are never printed. Every request and response is
stored by content hash (<work>/api/opt_responses/), so a rerun replays without new requests. Costs go to the optional
cost log <work>/api/ledger.jsonl; the optional spending cap is SPI_API_CAP_USD (settled + reserved + next call).
Prices are the providers' list prices when the runs were made (deepseek-v4-flash at the peak-hour rate).
"""
import fcntl
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True
from common import API_DIR, read_json, save_json   # noqa: E402
import classic_optimizers                                                          # noqa: E402
import llm_templates                                                              # noqa: E402  (shared cost log and lock)

classic_optimizers.BATCH.setdefault("llm", 8)
STORE = API_DIR / "opt_responses"
BAND = (0.500, 0.502)
PRICES = {  # USD per 1M tokens: input, cached input, output
    "gpt-5-mini": (0.25, 0.025, 2.00),            # OpenAI list price
    "gpt-5.6-luna": (0.20, 0.02, 1.20),           # list price
    "deepseek-v4-flash": (0.30, 0.006, 1.20),     # DeepSeek peak-hour rate (off-peak is half); conservative
}
PROVIDER = {"gpt-5-mini": "openai", "gpt-5.6-luna": "openai", "deepseek-v4-flash": "deepseek"}
MAX_OUT = 8192


class BudgetExceeded(RuntimeError):
    pass


def _key(provider):
    name = "DEEPSEEK_API_KEY" if provider == "deepseek" else "OPENAI_API_KEY"
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set")
    return value


def _cost(model, usage):
    if not usage:
        return None
    pin, pcached, pout = PRICES[model]
    prompt = usage.get("prompt_tokens") or 0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or usage.get("prompt_cache_hit_tokens") or 0)
    completion = usage.get("completion_tokens") or 0
    return ((prompt - cached) * pin + cached * pcached + completion * pout) / 1e6


def _worst(model, messages):
    pin, _, pout = PRICES[model]
    return (sum(len(m["content"]) for m in messages) / 3.0 * pin + MAX_OUT * pout) / 1e6


def _cap():
    return llm_templates.CAP_USD


def chat(model, messages, params, run_name, tag):
    """One stored / ledgered chat completion. Returns (value dict, info dict)."""
    payload = {"model": model, "params": params, "messages": messages}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    target = STORE / f"{key}.json"
    STORE.mkdir(parents=True, exist_ok=True)
    with open(STORE / f"{key}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            return read_json(target)["response"], {"source": "store", "key": key, "cost_usd": 0.0}
        with llm_templates._locked():
            settled, reserved, _ = llm_templates.spending()
            worst = _worst(model, messages)
            if settled + reserved + worst > _cap():
                raise BudgetExceeded(f"cap {_cap()} USD: settled {settled:.3f} + reserved {reserved:.3f} + next {worst:.3f}")
            request_id = f"{time.time():.6f}-{os.getpid()}-{key[:12]}"
            llm_templates._append({"kind": "reserve", "request_id": request_id, "key": key, "run": run_name, "tag": tag,
                            "model": model, "worst_case_usd": worst, "unix": time.time()})
        from openai import OpenAI
        provider = PROVIDER[model]
        client = OpenAI(api_key=_key(provider), timeout=600, max_retries=0,
                        **({"base_url": "https://api.deepseek.com"} if provider == "deepseek" else {}))
        kwargs = {"model": model, "messages": messages, "response_format": {"type": "json_object"}}
        if provider == "deepseek":
            kwargs.update(max_tokens=MAX_OUT, temperature=params.get("temperature", 0.5),
                          extra_body={"thinking": {"type": "disabled"}})
        else:
            kwargs.update(max_completion_tokens=MAX_OUT, reasoning_effort=params.get("reasoning_effort", "low"))
        value, actual, t0 = None, None, time.time()
        for attempt in range(4):
            try:
                reply = client.chat.completions.create(**kwargs)
                usage = reply.usage.model_dump() if reply.usage else None
                value = {"raw_response": reply.choices[0].message.content or "", "model": reply.model,
                         "finish_reason": reply.choices[0].finish_reason, "usage": usage,
                         "elapsed_seconds": time.time() - t0, "attempts": attempt + 1}
                actual = _cost(model, usage)
                break
            except Exception as exc:              # retry transient failures, then give up for this call
                status = getattr(exc, "status_code", None)
                value = {"error_type": type(exc).__name__, "status_code": status, "elapsed_seconds": time.time() - t0}
                if attempt == 3 or (status is not None and status < 500 and status != 429):
                    break
                time.sleep(5 * 2 ** attempt)
        with llm_templates._locked():
            llm_templates._append({"kind": "settle", "request_id": request_id, "key": key, "run": run_name, "tag": tag,
                            "model": model, "usage": value.get("usage"), "cost_usd": worst if actual is None else actual,
                            "status": "ok" if "raw_response" in value else "failed", "unix": time.time()})
        if "raw_response" in value:
            save_json(target, {"key": key, "request": payload, "response": value, "run": run_name, "tag": tag})
        return value, {"source": "api", "key": key, "cost_usd": worst if actual is None else actual}


ANON_SYSTEM = """
You are an optimizer for a constrained black-box minimization problem over integer vectors.
Your task is to propose the exact number of complete candidate vectors requested in the task.

Each candidate is a vector x of integers; position j may take any integer in its bounds. You are given evaluated
examples: each has its vector x, its objective value (lower is better) and its cost fraction. You are also given, for
every position and every value, the cost fraction that value contributes; the cost fraction of a vector is the sum
over its positions. A vector is feasible only if its cost fraction lies inside the band given in the task.

Use the evaluated examples to infer which patterns tend to give low objective values. The examples in weak_examples
are weak, those in strong_examples are strong, and recent_history shows the recent trajectory. Compare them and infer
what differs between weak and strong vectors. If the search appears stuck, deliberately propose some candidates that
explore meaningfully different patterns instead of only small variations of recent vectors.

Rules:
1. Return valid JSON only.
2. Return exactly the number of vectors requested in the task.
3. Each vector must have the required length and respect the bounds of every position.
4. Do not copy evaluated vectors; propose new ones and keep the batch diverse.
5. Prefer feasible vectors: their cost fraction must lie inside the band.
6. In reasoning, briefly explain what patterns you learned and how they shaped the proposals.

Output schema:
{"reasoning": "short explanation", "policies": [{"x": [0, 1, 2, ...]}]}
""".strip()


class LLMOptimizer:
    name = "llm"

    def __init__(self, space, seed, *, decoder, level_bounds, layers, run_name, model="gpt-5-mini",
                 params=None, anonymize=False, shuffle_scores=False, n_strong=10, n_weak=10, n_recent=20,
                 max_revisions=2, **_):
        from evoprune.openai_array_policy_agent import DEFAULT_SYSTEM_INSTRUCTION, compact_array_policy_constraints
        self.space, self.seed, self.decoder, self.L = space, seed, decoder, layers
        self.bounds = level_bounds
        self.run_name, self.model, self.params = run_name, model, dict(params or {})
        self.anonymize, self.shuffle = anonymize, shuffle_scores
        self.n_strong, self.n_weak, self.n_recent, self.max_rev = n_strong, n_weak, n_recent, max_revisions
        self.history, self.calls, self.proposals, self.n_calls = [], [], [], 0
        lo = [int(level_bounds["layers"][str(i)]["attn_min"]) for i in range(layers)] + \
             [int(level_bounds["layers"][str(i)]["mlp_min"]) for i in range(layers)]
        hi = [int(level_bounds["layers"][str(i)]["attn_max"]) for i in range(layers)] + \
             [int(level_bounds["layers"][str(i)]["mlp_max"]) for i in range(layers)]
        self.lo, self.hi = lo, hi
        removed = np.asarray(decoder.removed, float)
        if anonymize:
            self.order = list(range(2 * layers))
            random.Random(seed * 104729 + 1).shuffle(self.order)
            self.system = ANON_SYSTEM
            self.cost_table = [[round(float(removed[g, k]), 5) for k in range(lo[g], hi[g] + 1)] for g in self.order]
            self.constraints = {"length": 2 * layers, "bounds": [[lo[g], hi[g]] for g in self.order]}
        else:
            self.order = None
            # The original agent instruction, with its budget rule (12) replaced by the hard band and the self-correction step.
            rule12 = "12. The evaluator's reward already handles compression-budget error. Do not assume that a later repair step will alter your policy."
            assert rule12 in DEFAULT_SYSTEM_INSTRUCTION
            text = DEFAULT_SYSTEM_INSTRUCTION.replace(rule12, (
                "12. Every policy must have flops_reduction between 0.500 and 0.502 inclusive (exact database lookup). "
                "The task lists the fraction of the dense FLOPs removed by every level of every module; flops_reduction is "
                "their sum. Policies outside the band are returned to you with their exact flops_reduction for revision; "
                "policies still outside after two revisions are changed by a greedy repair."))
            text = text.replace("- its evaluation metrics: kl, ppl, reward, flops_reduction",
                                "- its evaluation metrics: kl, reward (= -kl), flops_reduction (ppl is not measured in this run)")
            self.system = text + "\n\nRun-specific policy bounds:\n" + json.dumps(compact_array_policy_constraints(level_bounds))
            self.cost_table = {"attn_removed_fraction_by_level": [[round(float(removed[i, k]), 5) for k in range(lo[i], hi[i] + 1)] for i in range(layers)],
                               "mlp_removed_fraction_by_level": [[round(float(removed[layers + i, k]), 5) for k in range(lo[layers + i], hi[layers + i] + 1)] for i in range(layers)]}
            self.constraints = compact_array_policy_constraints(level_bounds)

    # ------------------------------------------------------------ views
    def _view(self, policy):
        if self.anonymize:
            return {"x": [int(policy[g]) for g in self.order]}
        return {"attn_levels": [int(v) for v in policy[:self.L]], "mlp_levels": [int(v) for v in policy[self.L:]]}

    def _from_view(self, item):
        if self.anonymize:
            x = item.get("x") if isinstance(item, dict) else None
            if not isinstance(x, list) or len(x) != 2 * self.L:
                raise ValueError("x has the wrong length")
            policy = [0] * (2 * self.L)
            for j, g in enumerate(self.order):
                policy[g] = int(x[j])
        else:
            a, m = item.get("attn_levels"), item.get("mlp_levels")
            if not isinstance(a, list) or not isinstance(m, list) or len(a) != self.L or len(m) != self.L:
                raise ValueError("attn_levels / mlp_levels have the wrong length")
            policy = [int(v) for v in a] + [int(v) for v in m]
        if any(v < self.lo[g] or v > self.hi[g] for g, v in enumerate(policy)):
            raise ValueError("level out of bounds")
        return tuple(policy)

    def _example(self, row, kl):
        entry = dict(self._view(row["policy"]))
        if self.anonymize:
            entry.update(objective=round(kl, 5), cost_fraction=round(row["flops"], 5))
        else:
            entry.update(kl=round(kl, 5), reward=round(-kl, 5), flops_reduction=round(row["flops"], 5))
        return entry

    def tell(self, policy, kl):
        self.history.append({"policy": tuple(int(v) for v in policy), "kl": float(kl), "flops": self.space.budget(policy)})

    def _observation(self):
        kls = [h["kl"] for h in self.history]
        if self.shuffle:                          # feedback without information: divergences permuted across allocations
            rng = random.Random(self.seed * 7919 + self.n_calls)
            kls = kls[:]
            rng.shuffle(kls)
        ranked = sorted(range(len(self.history)), key=lambda i: kls[i])
        pick = lambda idx: [self._example(self.history[i], kls[i]) for i in idx]
        obs = {"target_flops_reduction": 0.501, "band": list(BAND),
               "weak_examples": pick(sorted(ranked[-self.n_weak:], key=lambda i: kls[i])),
               "strong_examples": pick(ranked[:self.n_strong]),
               "recent_history": pick(range(max(0, len(self.history) - self.n_recent), len(self.history)))}
        if self.anonymize:
            obs = {"band": list(BAND), "weak_examples": obs["weak_examples"], "strong_examples": obs["strong_examples"],
                   "recent_history": obs["recent_history"]}
        return obs

    def _parse(self, value, n):
        """Policies from a response: list of (policy or None, error)."""
        try:
            items = json.loads(value.get("raw_response", ""))["policies"]
            if not isinstance(items, list):
                raise ValueError("policies is not a list")
        except Exception as exc:
            return [(None, f"unparsable: {type(exc).__name__}")] * n
        out = []
        for i in range(n):
            if i >= len(items):
                out.append((None, "missing"))
                continue
            try:
                out.append((self._from_view(items[i]), None))
            except Exception as exc:
                out.append((None, str(exc)))
        return out

    def ask(self, count, seen):
        self.n_calls += 1
        call = self.n_calls
        task = {"task": f"Propose exactly {count} complete {'candidate vectors' if self.anonymize else 'policy candidates in array format'}.",
                "num_policies": int(count), ("problem_constraints" if self.anonymize else "policy_constraints"): self.constraints,
                ("cost_fraction_by_position_and_value" if self.anonymize else "flops_removed_fraction_by_level"): self.cost_table,
                "observation": self._observation()}
        messages = [{"role": "system", "content": self.system},
                    {"role": "user", "content": json.dumps(task, ensure_ascii=False, separators=(",", ":"))}]
        value, info = chat(self.model, messages, self.params, self.run_name, f"call{call}.propose")
        self.calls.append({"call": call, "step": "propose", **info, "usage": value.get("usage"),
                           "elapsed": value.get("elapsed_seconds"), "error": value.get("error_type")})
        parsed = self._parse(value, count)
        state = [{"slot": i, "policy": p, "error": e, "raw_feasible": p is not None and self.space.feasible(p),
                  "revisions": 0} for i, (p, e) in enumerate(parsed)]
        for rev in range(1, self.max_rev + 1):
            bad = [s for s in state if s["policy"] is not None and not self.space.feasible(s["policy"])]
            if not bad or "raw_response" not in value:
                break
            request = {"task": (f"Revise the {len(bad)} {'vectors' if self.anonymize else 'policies'} below so that "
                                f"{'cost_fraction' if self.anonymize else 'flops_reduction'} lies in [{BAND[0]:.3f}, {BAND[1]:.3f}], "
                                "changing as few values as possible and keeping what each was meant to test. "
                                f"Return JSON with exactly {len(bad)} entries under \"policies\", in the same order."),
                       "to_revise": [{**self._view(s["policy"]),
                                      ("cost_fraction" if self.anonymize else "flops_reduction"): round(self.space.budget(s["policy"]), 5)}
                                     for s in bad]}
            messages = messages + [{"role": "assistant", "content": value["raw_response"]},
                                   {"role": "user", "content": json.dumps(request, ensure_ascii=False, separators=(",", ":"))}]
            value, info = chat(self.model, messages, self.params, self.run_name, f"call{call}.revise{rev}")
            self.calls.append({"call": call, "step": f"revise{rev}", **info, "usage": value.get("usage"),
                               "elapsed": value.get("elapsed_seconds"), "error": value.get("error_type")})
            for s, (p, e) in zip(bad, self._parse(value, len(bad))):
                if p is not None:
                    s["policy"], s["revisions"] = p, rev
        out, taken = [], set()
        for s in state:
            record = {"call": call, "slot": s["slot"], "raw_feasible": s["raw_feasible"], "revisions": s["revisions"],
                      "parse_error": s["error"]}
            p = s["policy"]
            if p is None:
                p = self.space.random_feasible()
                record["status"] = "format_fallback"
            elif self.space.feasible(p):
                record["status"] = "feasible" if s["revisions"] == 0 else "feasible_after_revision"
            else:
                repaired = self.space.repair(p)
                record["status"] = "repaired"
                record["repair_changed"] = int(sum(a != b for a, b in zip(p, repaired))) if repaired else None
                p = repaired if repaired is not None else self.space.random_feasible()
            if p in seen or p in taken:
                p = self.space.unique(list(p), seen | taken)
                record["duplicate_perturbed"] = True
            taken.add(p)
            record["policy"] = list(p)
            self.proposals.append(record)
            out.append(p)
        return out

    def report(self):
        n = len(self.proposals) or 1
        status = [p["status"] for p in self.proposals]
        repaired = [p.get("repair_changed") for p in self.proposals if p["status"] == "repaired" and p.get("repair_changed") is not None]
        usage = [c["usage"] for c in self.calls if c.get("usage")]
        return {"llm_summary": {
                    "model": self.model, "params": self.params, "anonymize": self.anonymize, "shuffle_scores": self.shuffle,
                    "proposals": len(self.proposals),
                    "raw_feasible_share": sum(p["raw_feasible"] for p in self.proposals) / n,
                    "feasible_after_revision_share": sum(s in ("feasible", "feasible_after_revision") for s in status) / n,
                    "repaired_share": status.count("repaired") / n, "format_fallback_share": status.count("format_fallback") / n,
                    "duplicate_share": sum(bool(p.get("duplicate_perturbed")) for p in self.proposals) / n,
                    "mean_repair_changed": float(np.mean(repaired)) if repaired else None,
                    "requests": sum(c["source"] == "api" for c in self.calls), "calls": len(self.calls),
                    "prompt_tokens": sum(u.get("prompt_tokens") or 0 for u in usage),
                    "completion_tokens": sum(u.get("completion_tokens") or 0 for u in usage),
                    "cost_usd_estimate": sum(c.get("cost_usd") or 0 for c in self.calls)},
                "llm_calls": self.calls, "llm_proposals": self.proposals}
