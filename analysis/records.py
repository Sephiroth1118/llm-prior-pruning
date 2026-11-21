"""Record layout shared by all analysis scripts.

Every script takes the RECORDS DIRECTORY as its first positional argument (pass "-" to use the environment variable
SPI_RECORDS instead). The records directory is the directory that holds the search records, benchmark results and the
FLOPs tables; the layout is (only the parts the analysis scripts read are marked with *; parts marked with - are not in
the published run records):

    <records>/
      runs/                                   *  one directory per search
        <run_name>/results.json               *  entries (id, kl, flops_reduction, policy, phase, ...), stats, llm calls
        <run_name>/progress.json              -  running state of an unfinished search
        <run_name>/job.json                   -  the job file the search was started from (in configs/)
        <run_name>/eval_cache.jsonl           -  cache of evaluated allocations
        <run_name>/round<k>_request.json, round<k>_accepted.json   template requests and accepted templates
        <run_name> = <group>_s<seed>  or  <group>_rk_s<seed>  (re-ask rerun, preferred when present; see resolve_run)
      benchmark/
        <model>/<job>/benchmark_results.json  *  downstream evaluation of a compressed model
        <model>/<job>/policy.json                the evaluated allocation
        <model>/<job>/{benchmark_command.json, benchmark.log, benchmark_table.md, deployment.json, timing.json,
                       compact_model/}        -
        jobs/                                 -  benchmark job files (in configs/benchmark/), queue files
      reports/
        benchmark/metrics.json                *  rows "<model>/<group>/s<seed>" with ppl, avg, task scores, search_kl
        <other report directories>            -  held-out selection, family matrix, confirmation runs, ...
      flops/<model_key>.json                  *  per-module FLOPs tables ("heads", "kv_groups", "mlp_dims");
                                                 model keys: qwen15b, qwen3b, qwen7b, llama32_1b, llama32_3b, llama31,
                                                 olmo1b, olmo7b
      speed/                                     inference speed measurements (llama31_speed_*.json)
      recovery/runs/<row>/                       post-pruning recovery training (llmpruner, ours, uniform, ...): train_config.json,
                                                 train_summary.json, gpu_peak_mib.json (jobs in configs/recovery/)
      moe/                                       the OLMoE experiment
        runs/<run_name>/results.json             olmoe_{ea,llm,own,random,rule}_nogp_s<seed>
        benchmark/<job>/                         downstream evaluation (jobs in configs/moe/bench_jobs/)
        expert_groups_fineweb.json, own_template.json, heldout.json
      api/{responses,opt_responses}/<key>.json   stored LLM requests and responses (copy to <work>/api/ to replay)
      baseline_gate2/seed<k>/results.json        shared initial design of the Qwen2.5-1.5B searches (seeds 0-4)

A search group is named <model-prefix>_<method>[_<variant>]; the groups each script reads are listed at the top of
the script and in MAPPING.txt.
"""
import os
from pathlib import Path

ENV_VAR = "SPI_RECORDS"


def records_dir(arg=None):
    """The records directory: `arg`, or $SPI_RECORDS when `arg` is None or "-"."""
    if arg in (None, "-"):
        arg = os.environ.get(ENV_VAR)
    if not arg:
        raise SystemExit(f"records directory missing: pass it as the first argument or set ${ENV_VAR}")
    return Path(arg)


def runs_path(records):
    return Path(records) / "runs"


def benchmark_path(records):
    return Path(records) / "benchmark"


def reports_path(records):
    return Path(records) / "reports"


def flops_table_path(records, model):
    """FLOPs table of a model key (qwen15b, llama31, qwen3b, ...)."""
    return Path(records) / "flops" / f"{model}.json"


def resolve_run(runs, group, seed):
    """Run directory for (group, seed): the rerun under the re-ask protocol (<group>_rk_s<seed>) when one exists,
    otherwise the original run <group>_s<seed>. Runs without a rejected template are identical under both protocols."""
    runs = Path(runs)
    rk = runs / f"{group}_rk_s{seed}"
    return rk if (rk / "results.json").exists() else runs / f"{group}_s{seed}"
