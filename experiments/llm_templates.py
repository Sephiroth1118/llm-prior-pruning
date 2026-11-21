"""LLM template proposals (structural prior injection) with an optional cost cap and a content-addressed response store.

* Responses are stored by the SHA-256 of (model, parameters, system, user) under <work>/api/responses/. An identical
  prompt in any run reuses the stored response without a new request.
* Optional cost log: every physical API request is reserved (worst case) in <work>/api/ledger.jsonl before it is sent
  and settled with the reported usage afterwards, under an exclusive file lock. When settled cost plus open reservations
  plus the next worst case would exceed the cap (CAP_USD, overridable by the environment variable SPI_API_CAP_USD),
  the call is refused and the caller falls back to an annotated random template. The log files are created on demand.
* The API key is read from the environment variable OPENAI_API_KEY only.
"""
import fcntl
import hashlib
import json
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path

from common import API_DIR, REPO, save_json, read_json

MODEL = "gpt-5-mini"
PARAMS = {"reasoning_effort": "low", "max_completion_tokens": 4096, "response_format": "json_object"}
PRICE = {"input": 0.25, "cached_input": 0.025, "output": 2.00}     # USD per 1M tokens, standard tier
CAP_USD = float(os.environ.get("SPI_API_CAP_USD", "inf"))   # optional spending cap in USD
API = API_DIR
STORE = API / "responses"
LEDGER = API / "ledger.jsonl"


def original_system_prompt():
    """The system prompt of the template rounds (prompts/serial_structure_system.txt; response keys hash this exact text)."""
    return (REPO / "prompts" / "serial_structure_system.txt").read_text()


SYSTEM_ORIG = original_system_prompt()

SYSTEM_V2 = """You propose coarse pruning STRUCTURES during the early steps of a serial search.
Each structure is decoded to a budget-feasible anchor, then that anchor receives 15 ordinary random
mutations in a local top-eight EA (16 total evaluations including the anchor). Only after this short
refinement can its survivors join the global top-eight population. This run has {rounds} structure
rounds with {per_round} structures each; after the last round, ordinary EA (random budget-preserving
mutations of the global top eight) continues WITHOUT further calls until {budget} total evaluations.
What matters is the best KL at the END of the whole {budget}-evaluation search, so prefer structures
whose descendants can keep improving, not just a lucky raw anchor.
Use architectural priors and only the observations from this run supplied below.

NOVELTY RULE: never return a weight vector that was already tried in this run (all tried vectors are
listed in "tried_templates") and never return two identical vectors. Use previous refinement results:
you may exploit a promising region with a clearly different variant, but at least one structure per
round must test a materially different hypothesis (for example another depth profile or another
attention/MLP split).

Return JSON {{"structures":[{{"name":str,"weights":[8 numbers],"rationale":str}}, ...]}} with
exactly {per_round} structures. The weight alphabet is [0,0.25,0.5,1,2,4]. The order is attention
depth quarters 1-4, then MLP depth quarters 1-4. These are RELATIVE REMOVAL FRACTIONS per
module, NOT shares of the total removed FLOPs. All eight equal nonzero weights = uniform
fractional pruning; larger weights mean that module/depth should be pruned more. Scaling
all weights equally changes nothing. Zero means unpruned at the anchor, not during later EA.

A shared non-learned decoder solves the common scalar strength so the continuous target
removes 50.1% of the evaluator's dense FLOPs estimate, clamps at each gene's capacity, then
uses seeded stochastic rounding and profile-preserving discretization repair. The evaluated
anchor MUST lie in [50.0%,50.2%] by the actual database lookup. There is no KL-based repair.
Positive-weight cells must jointly have capacity at least 50.1% (capacity is supplied).
Discrete rounding means uniform is approximate. A structure is an initialization template,
not a permanent species boundary. You cannot change mutation, scoring, budget, or selection.
You receive current-run strong/weak/recent structures and previous anchor/refinement results.
No heldout data, other arms, or future results are supplied. Do not return code or policies.
"""


class BudgetExceeded(RuntimeError):
    pass


@contextmanager
def _locked():
    API.mkdir(parents=True, exist_ok=True)
    with open(API / "ledger.lock", "a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def _ledger():
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def spending():
    """(settled USD, open reservations USD, number of physical requests)."""
    rows = _ledger()
    settled = {r["request_id"]: r for r in rows if r["kind"] == "settle"}
    reserved = [r for r in rows if r["kind"] == "reserve"]
    total = sum(r["cost_usd"] for r in settled.values())
    open_reservations = sum(r["worst_case_usd"] for r in reserved if r["request_id"] not in settled)
    return total, open_reservations, len(reserved)


def _append(row):
    with open(LEDGER, "a") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def usage_cost(usage):
    if not usage:
        return None
    prompt = usage.get("prompt_tokens") or 0
    completion = usage.get("completion_tokens") or 0
    # Conservative: cached input is billed at the full input price here.
    return (prompt * PRICE["input"] + completion * PRICE["output"]) / 1e6


def worst_case(system, user):
    prompt_tokens = (len(system) + len(json.dumps(user))) / 3.0   # generous chars-per-token estimate
    return (prompt_tokens * PRICE["input"] + PARAMS["max_completion_tokens"] * PRICE["output"]) / 1e6


def prompt_key(system, user):
    payload = {"model": MODEL, "params": PARAMS, "system": system, "user": user}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _api_key():
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return key


def get_response(system, user, run_name, tag, allow_api=True):
    """Return (response_value, info). info['source'] is 'store' (no new request) or 'api'."""
    key = prompt_key(system, user)
    target = STORE / f"{key}.json"
    STORE.mkdir(parents=True, exist_ok=True)
    # Serialize identical prompts across processes so a prompt is never paid twice.
    with open(STORE / f"{key}.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if target.exists():
            stored = read_json(target)
            return stored["response"], {"source": "store", "key": key, "cost_usd": 0.0}
        if not allow_api:
            raise BudgetExceeded("API calls disabled for this run and no stored response matches")
        with _locked():
            settled, reserved, _ = spending()
            cost = worst_case(system, user)
            if settled + reserved + cost > CAP_USD:
                raise BudgetExceeded(f"cap {CAP_USD} USD: settled {settled:.4f} + reserved {reserved:.4f} + next {cost:.4f}")
            request_id = f"{time.time():.6f}-{os.getpid()}-{key[:12]}"
            _append({"kind": "reserve", "request_id": request_id, "key": key, "run": run_name, "tag": tag,
                     "worst_case_usd": cost, "unix": time.time()})
        from openai import OpenAI
        client = OpenAI(api_key=_api_key(), base_url="https://api.openai.com/v1", timeout=180, max_retries=0)
        t0 = time.time()
        try:
            reply = client.chat.completions.create(
                model=MODEL, messages=[{"role": "system", "content": system},
                                       {"role": "user", "content": json.dumps(user)}],
                response_format={"type": PARAMS["response_format"]},
                max_completion_tokens=PARAMS["max_completion_tokens"],
                reasoning_effort=PARAMS["reasoning_effort"])
            usage = reply.usage.model_dump() if reply.usage else None
            value = {"raw_response": reply.choices[0].message.content, "model": reply.model,
                     "finish_reason": reply.choices[0].finish_reason, "usage": usage,
                     "elapsed_seconds": time.time() - t0}
            actual = usage_cost(usage)
        except Exception as exc:                  # a failed request is still settled at worst case
            value = {"error_type": type(exc).__name__, "status_code": getattr(exc, "status_code", None),
                     "elapsed_seconds": time.time() - t0}
            usage, actual = None, None
        with _locked():
            _append({"kind": "settle", "request_id": request_id, "key": key, "run": run_name, "tag": tag,
                     "usage": usage, "cost_usd": cost if actual is None else actual,
                     "status": "ok" if "raw_response" in value else "failed", "unix": time.time()})
        if "raw_response" in value:
            save_json(target, {"key": key, "request": {"system": system, "user": user}, "response": value,
                               "run": run_name, "tag": tag})
        return value, {"source": "api", "key": key, "cost_usd": cost if actual is None else actual}


def evidence(state, decoder, round_id, extra=None, model_name="Qwen2.5-1.5B", layers=28):
    """The evidence payload of the template prompt, plus optional extras."""
    def view(row):
        return {"id": row["id"], "kl": row["kl"], "phase": row["phase"], "origin": row["origin"],
                **decoder.describe(row["policy"])}
    ordered = sorted(state.entries, key=state.rank)
    payload = {"round": round_id + 1, "model": model_name, "layers": layers,
               "scoring": "FineWeb search KL, 8 x 2048 tokens; lower is better",
               "evaluations_used": len(state.entries), "global_best_kl": ordered[0]["kl"],
               "cell_max_removed_dense_fraction": [float(decoder.removed[decoder.cells == k, -1].sum()) for k in range(8)],
               "strong": [view(r) for r in ordered[:4]], "weak": [view(r) for r in ordered[-4:]],
               "recent": [view(r) for r in state.entries[-8:]], "previous_refinements": state.branches}
    if extra:
        payload.update(extra)
    return payload


def parse_structures(value, decoder, count, rng, tried=None):
    """Validate exactly `count` structures; invalid or (optionally) repeated ones become annotated random fallbacks."""
    errors, valid = [], []
    try:
        templates = json.loads(value["raw_response"])["structures"]
        if not isinstance(templates, list) or len(templates) != count:
            raise ValueError(f"Expected {count} structures")
    except (KeyError, TypeError, ValueError):
        templates = [None] * count
    seen = [list(t) for t in (tried or [])]
    for index, item in enumerate(templates):
        try:
            if not isinstance(item, dict) or any(not isinstance(item.get(k), str) for k in ("name", "rationale")):
                raise ValueError("Missing name or rationale")
            decoder.validate(item)
            if tried is not None and list(item["weights"]) in seen:
                raise ValueError("Repeated template")
            valid.append({k: item[k] for k in ("name", "weights", "rationale")})
            if tried is not None:
                seen.append(list(item["weights"]))
        except (ValueError, TypeError, AttributeError) as exc:
            errors.append({"slot": index, "reason": str(exc)})
            valid.append({**decoder.random_template(rng), "fallback": True})
    return valid, errors


if __name__ == "__main__":
    print(json.dumps({"spending": spending(), "system_orig_chars": len(SYSTEM_ORIG)}))


WORDS = {1: "ONE", 2: "TWO", 3: "THREE", 4: "FOUR", 6: "SIX", 8: "EIGHT"}


def system_prompt_for(per_round=2, rounds=2, alphabet=(0, .25, .5, 1, 2, 4), budget=320, refine=16):
    """The template system prompt with only the factual numbers substituted. Default arguments return
    SYSTEM_ORIG unchanged, so stored responses keep matching."""
    text = SYSTEM_ORIG
    repl = []
    if per_round != 2 or rounds != 2 or refine != 16:
        after = budget - 16 - rounds * per_round * refine
        repl.append(("exactly TWO structures", f"exactly {WORDS[per_round]} structures"))
        repl.append(("Two structures are tried\nper round; there are two rounds, then 240 evaluations of ordinary EA",
                     f"{WORDS[per_round].capitalize()} structures are tried\nper round; there are {WORDS[rounds].lower()} rounds, "
                     f"then {after} evaluations of ordinary EA"))
    if refine != 16:
        repl.append(("then that anchor receives 15 ordinary random mutations",
                     f"then that anchor receives {refine - 1} ordinary random mutations"))
        repl.append(("in a local top-eight EA (16 total evaluations including the anchor)",
                     f"in a local top-eight EA ({refine} total evaluations including the anchor)"))
    alpha = tuple(float(a) for a in alphabet)
    if alpha != (0., .25, .5, 1., 2., 4.):
        shown = "[" + ",".join(("%g" % a) for a in alpha) + "]"
        repl.append(("The weight alphabet is [0,0.25,0.5,1,2,4].", f"The weight alphabet is {shown}."))
    for old, new in repl:
        if old not in text:
            raise ValueError(f"prompt anchor not found: {old[:40]!r}")
        text = text.replace(old, new)
    return text
