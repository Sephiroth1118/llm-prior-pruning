"""LLM template rounds with re-asking until every template is valid and decodable .

Used by the three runners when a job has "template_protocol": "reask"; jobs without it keep the original behaviour.
Original protocol: an invalid structure, a failed request, or a structure the decoder cannot place in the compute band
was replaced by an annotated random template. New protocol, for the source "llm" only:
  1. a failed request (no response) is repeated, with a pause, up to `max_retries` times;
  2. every returned structure is checked as before (alphabet, eight weights, name and rationale, enough capacity in the
     positive-weight cells) and is additionally decoded with exactly the random stream its branch will use
     (random.Random(seed * 10000 + 100 + branch), as in ResearchSearch.refine_branch), so a structure is accepted only
     if its anchor can be built;
  3. rejected structures are returned to the LLM with the reasons and the accepted ones, and the LLM is asked for
     replacements; up to `max_reasks` re-asks per round;
  4. if a round still lacks valid structures, the run stops with an error. No random template is ever substituted.
A run in which every first answer is valid is identical to the original protocol: the same request, the same stored
response and the same templates, and no random number is drawn from any stream the search uses.
"stop_after_templates": true ends the run (exit code 0) right after the last round's templates are accepted and writes
templates_ready.json; evaluations made so far stay in the evaluation cache and the responses in the response store, so
the full run later replays both.
The response store and cost log are those of llm_templates (optional spending cap: SPI_API_CAP_USD).
"""
import json
import random
import time

from common import read_json, save_json
import llm_templates

def _request(system, user, run_name, tag, allow_api, max_retries, log):
    """One logical request; a failed request (no response) is repeated after a pause."""
    for attempt in range(max_retries + 1):
        try:
            value, info = llm_templates.get_response(system, user, run_name, f"{tag}.try{attempt}", allow_api=allow_api)
        except llm_templates.BudgetExceeded as exc:
            raise RuntimeError(f"template request refused ({exc}); no random template is substituted") from exc
        log.append({"tag": tag, "try": attempt, "response_source": info["source"], "key": info.get("key"),
                    "ok": "raw_response" in value, "error_type": value.get("error_type"),
                    "usage": value.get("usage"), "model": value.get("model")})
        if "raw_response" in value:
            return value
        time.sleep(min(240, 10 * 3 ** attempt))
    raise RuntimeError(f"{tag}: request failed {max_retries + 1} times")


def check(item, decoder, seed, branch, seen):
    """None if `item` is acceptable for `branch`, otherwise the reason it is not."""
    if not isinstance(item, dict) or any(not isinstance(item.get(k), str) for k in ("name", "rationale")):
        return "missing name or rationale"
    try:
        decoder.validate(item)
    except (ValueError, TypeError) as exc:
        return f"invalid structure: {exc}"
    rng = random.Random(seed * 10000 + 100 + branch)          # the stream refine_branch uses for this branch
    try:
        for _ in range(32):
            anchor, _ = decoder.decode(item, rng)
            if anchor not in seen:
                return None
    except ValueError as exc:
        return (f"decoding failed: {exc}. Rounding and repair could not bring the allocation into the compute band "
                f"[50.0%, 50.2%] with the cells that have positive weight")
    return "decoding produced only allocations that were already evaluated"


def propose(job, out, decoder, state, round_id, n, system, user, total_rounds):
    seed, run = job["seed"], job["name"]
    max_reasks, max_retries = int(job.get("reask_max", 10)), int(job.get("request_retries", 5))
    allow_api = job.get("allow_api", False)
    first_branch = len(state.branches)
    slots, rejected, log = [None] * n, [], []
    request = user
    for reask in range(max_reasks + 1):
        need = [i for i, s in enumerate(slots) if s is None]
        value = _request(system, request, run, f"round{round_id + 1}.ask{reask}", allow_api, max_retries, log)
        try:
            items = json.loads(value["raw_response"])["structures"]
            if not isinstance(items, list):
                raise ValueError
        except (KeyError, TypeError, ValueError):
            items = []
            rejected.append({"weights": None, "reason": "response is not a JSON object with a list 'structures'"})
        for index, item in enumerate(items):
            if not need:
                break
            # first answer: structure i belongs to slot i (as in the original protocol); replacements fill the open slots
            if reask == 0:
                if index >= n:
                    break
                slot = index
            else:
                slot = need[0]
            reason = check(item, decoder, seed, first_branch + slot, state.seen)
            if reason is None:
                slots[slot] = {k: item[k] for k in ("name", "weights", "rationale")}
                need.remove(slot)
            else:
                rejected.append({"weights": item.get("weights") if isinstance(item, dict) else None, "reason": reason,
                                 "ask": reask})
        if not need:
            break
        accepted = [s["weights"] for s in slots if s is not None]
        request = {**user, "accepted_structures": accepted,
                   "rejected_structures": [{"weights": r["weights"], "reason": r["reason"]} for r in rejected],
                   "instruction": (f"Some structures were rejected for the reasons given in rejected_structures. Return "
                                   f"exactly {len(need)} new structure(s) in the same JSON format to replace them. Every "
                                   f"structure must follow all rules of the system message and must be decodable into "
                                   f"the compute band.")}
    else:
        save_json(out / f"round{round_id + 1}_failed.json", {"rejected": rejected, "requests": log})
        raise RuntimeError(f"round {round_id + 1}: no valid structures after {max_reasks} re-asks")
    metadata = {"round": round_id + 1, "source": "llm", "protocol": "reask", "fallbacks": [], "reasks": reask,
                "rejected": rejected, "attempts": log,
                "usage": [r.get("usage") for r in log], "model": next((r["model"] for r in log if r.get("model")), None)}
    save_json(out / f"round{round_id + 1}_accepted.json", {"templates": slots, **metadata})
    if job.get("stop_after_templates") and round_id == total_rounds - 1:
        rounds = [read_json(out / f"round{r + 1}_accepted.json") for r in range(total_rounds)]
        save_json(out / "templates_ready.json", {"run": run, "seed": seed, "rounds": rounds,
                                                 "evaluations_used": len(state.entries)})
        raise SystemExit(0)
    return slots, metadata
