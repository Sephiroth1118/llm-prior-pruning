# Prior knowledge or optimization ability? LLMs in evolutionary search for structured pruning

Code accompanying the paper "Prior Knowledge or Optimization Ability? Disentangling the Role of Large Language Models in
Evolutionary Search for Structured Pruning" (submitted to Swarm and Evolutionary Computation).

This is a **reference release**: it contains the code that produced the paper's results, cleaned for publication. It has
been checked to compile and import on a CPU; end-to-end GPU runs were not repeated for the release.

License: Apache License 2.0 (see LICENSE and NOTICE). The level-database builders restored from DarwinLM are not covered;
see "Third-party origins".
Run records: https://doi.org/10.5281/zenodo.24410518 (Zenodo)

## What the paper does

The paper asks what a large language model contributes when it takes part in an evolutionary search that decides how much
of each attention and MLP module of an LLM to prune under a compute budget. It compares the LLM used as the optimizer with
conventional optimizers (EA, EA with GP-guided mutation, GP-BO, TPE, DE, PSO, CMA-ES, GA, random search) on the divergence
(KL) to the dense model on FineWeb-Edu text, and finds that its optimization is unremarkable while its prior knowledge
matters. It then proposes structural prior injection (SPI): an LLM writes coarse templates (8 cell weights) in two early
rounds, a decoder turns them into feasible allocations, each anchor gets a short refinement, and a plain (mu+lambda) EA
continues. Models: Qwen2.5-1.5B/3B/7B, Llama-3.2-1B/3B, Llama-3.1-8B, OLMo-1B/7B-0724, and OLMoE-1B-7B (experts).

## Layout

```
evoprune/      search and evaluation core: policy and EA operators (EA/), evaluator, model stitching, MoE support (MoE/),
               exact-FLOPs tables, the structure decoder and serial search (serial_search.py), the frozen GP surrogate
zipprune/      level-database building blocks, compact model export/loading (ZipLM/compact.py), parameter accounting,
               recovery-training utilities, magnitude 2:4 baseline (SemiStructured/)
experiments/   entry points (see "Experiments" below) and the single configuration module common.py
analysis/      scripts that regenerate the paper's tables, figures and numbers from the run records
prompts/       the system prompts sent to the LLMs (byte-exact)
configs/       job files of the searches, benchmarks and recovery runs behind the paper's results
third_party/   darwinlm/: restores the level-database builders from a DarwinLM clone (see "Level databases")
MANIFEST.md    release file -> original file, one row per file
```

## Installation

```bash
conda env create -f environment.yml && conda activate spi-pruning      # search and evaluation (or: pip install -r requirements.txt)

# downstream benchmark: a separate environment, because it pins other versions
conda create -n spi-benchmark python=3.10 && conda activate spi-benchmark
pip install -r requirements-benchmark.txt
```

Search and evaluation ran with torch 2.6.0 and transformers 4.45.2 (`requirements.txt`). The downstream benchmark ran with
transformers 4.44.2, lm-evaluation-harness 0.4.5 (`requirements-benchmark.txt`); OLMoE needs transformers 4.45.2.
For reproducible runs set `PYTHONHASHSEED=0`, `OMP_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `MKL_NUM_THREADS=1` and
`TOKENIZERS_PARALLELISM=false` (recommended, not enforced; `experiments/common.py` warns when they are missing).

Everything the scripts read or write lives in one **work directory**, `$SPI_WORK` (default `<repo>/work`); relative paths in
job files refer to it. Scripts are run from `experiments/` (they put the repository root on `sys.path` themselves).

```
work/
  hf_models/<name>/            model directories (Hugging Face, see below)
  db/<model_key>/              level databases
  models/<model_key>/{fineweb/, flops_table.json, teacher_search_8x2048_bf16.pt}
  runs/<run_name>/             search runs;  moe/ for OLMoE;  benchmark/, reports/, recovery/, speed/, api/
```

API keys are read only from the environment variables `OPENAI_API_KEY` and `DEEPSEEK_API_KEY`. Responses are stored by
content hash under `work/api/`; an optional cost log and spending cap (`SPI_API_CAP_USD`) are available. Nothing is read from
configuration files.

## Data preparation

**Models** (Hugging Face; download into `work/hf_models/<name>`, e.g. `huggingface-cli download <id> --local-dir work/hf_models/<name>`):

| key | Hugging Face id | directory name |
|---|---|---|
| qwen15b | Qwen/Qwen2.5-1.5B (revision 8faed761d45a263340a0528343f099c05c9a4323) | Qwen2.5-1.5B |
| qwen3b | Qwen/Qwen2.5-3B-Instruct | Qwen2.5-3B-Instruct |
| qwen7b | Qwen/Qwen2.5-7B-Instruct | Qwen2.5-7B-Instruct |
| llama32_1b | meta-llama/Llama-3.2-1B | Llama-3.2-1B |
| llama32_3b | meta-llama/Llama-3.2-3B | Llama-3.2-3B |
| llama31 | meta-llama/Llama-3.1-8B (revision d04e592bb4f6aa9cfee91e2e20afa771667e1d4b) | Llama-3.1-8B |
| olmo1b | allenai/OLMo-1B-0724-hf | OLMo-1B-0724-hf |
| olmo7b | allenai/OLMo-7B-0724-hf | OLMo-7B-0724-hf |
| olmoe | allenai/OLMoE-1B-7B-0924 | OLMoE-1B-7B-0924 |

The registry with all constants is `experiments/model_registry.py`. Baselines for the downstream tables additionally use
Shengkun/DarwinLM-4.6B-Pruned (`work/hf_models/DarwinLM-4.6B-Pruned`).

**Calibration and held-out text** (FineWeb-Edu, `HuggingFaceFW/fineweb-edu`, sample-10BT). Documents for search and held-out windows
come from the first half of the split, the final perplexity benchmark uses the second half, so they never share a document. The
documents are frozen once with Qwen2.5-1.5B's tokenizer (8 search windows and 32 held-out windows of 2048 tokens, whole documents
joined by the tokenized `"\n\n"`) and re-tokenized for every other model with the same documents:

```bash
cd experiments
python build_fineweb_calibration.py --tokenizer ../work/hf_models/Qwen2.5-1.5B --out ../work/models/qwen15b/fineweb
```

The re-tokenized windows of the other models, the exact-FLOPs tables and the dense teacher logits are prepared after the level
databases are built (see the end of "Level databases").

**Level databases** (the pruning level of every module, built once per model with the ZipLM/OBS procedure; inputs to all searches).
The builders are modified versions of files of DarwinLM (https://github.com/IST-DASLab/DarwinLM), which declares no license,
so this repository does not contain them. Restore them first; `restore.py` copies the original files from a DarwinLM clone,
applies our changes (`third_party/darwinlm/patches.json`) and checks that the result is byte-identical to the files we used:

```bash
git clone https://github.com/IST-DASLab/DarwinLM.git ../work/DarwinLM
git -C ../work/DarwinLM checkout 3596da3973224da83255363a4fbe6757d569dc8c
python ../third_party/darwinlm/restore.py ../work/DarwinLM     # restores experiments/struct_prune*.py, zipprune/struct_pruner.py,
                                                               # zipprune/fast_obc_struct.py, zipprune/legacy/, evoprune/struct_cap.py,
                                                               # evoprune/MoE/{struct_pruner,fast_obc_struct,cli_build_db}.py
```

The searches, the evaluation and the analysis do not need these files. `experiments/struct_prune.py` builds the databases;
for the models with a generic recipe in the registry:

```bash
python model_registry.py --build-db llama31 --gpus 0,1            # = torchrun struct_prune.py with the settings below
```

Settings: FineWeb-Edu calibration (`--calibration_data fineweb_edu`), 8,388,608 tokens, bf16, sdpa, `--prunable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj)$'`,
`--attn_prune_name o_proj --mlp_prune_name down_proj`, `--rows_in_parallel 128` for the Llama/OLMo databases. Per model:

| key | levels | calibration sequence | removed attention columns / MLP columns |
|---|---|---|---|
| llama31 | 11 | 4096 | 0,512,896,1408,1792,2304,2816,3328,3584,3968,4096 / 0,1440,2880,4288,5728,7168,8608,10048,11456,12896,14336 |
| llama32_1b, llama32_3b, olmo1b, olmo7b | 11 | 4096 | in `model_registry.REGISTRY[key]["build"]` (`--build-db <key>`) |
| qwen15b | 13 | 8192 | attention 0..1536 in steps of 128; MLP 0,736,1504,2240,2976,3744,4480,5216,5984,6720,7456,8224,8960; calibration file from `prepare_qwen15b_calibration.py` (pinned FineWeb-Edu shard, 1024 x 8192 tokens) passed as `--calibration_data`; built with `struct_prune_legacy.py` on four GPUs (command in its docstring) |
| qwen3b | 17 | 8192 | attention 0..2048 in steps of 128; MLP 0..11008 in steps of 688 |
| qwen7b | 11 | 8192 | attention 0,384,768,1152,1536,1792,2176,2560,2944,3200,3584; MLP 0,1920,3840,...,18944 (steps listed in `model_registry.py`) |

Example (Qwen2.5-3B, four GPUs):

```bash
cd experiments
torchrun --nnodes=1 --nproc-per-node=4 struct_prune.py --model_name_or_path ../work/hf_models/Qwen2.5-3B-Instruct \
  --prunable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj)$' --pre_block_modules model.embed_tokens model.rotary_emb \
  --block_modules model.layers --calibration_data fineweb_edu --calibration_tokens 8388608 --calibration_sequence_length 8192 \
  --cols_removed_attn 0 128 256 384 512 640 768 896 1024 1152 1280 1408 1536 1664 1792 1920 2048 \
  --cols_removed_mlp 0 688 1376 2064 2752 3440 4128 4816 5504 6192 6880 7568 8256 8944 9632 10320 11008 \
  --mlp_prune_name down_proj --attn_prune_name o_proj --low_cpu_mem_usage --cpu_offload_modules --cpu_offload_activations \
  --dtype bfloat16 --attn_implementation sdpa --save_dir ../work/db/qwen3b
```

When the database of a model is complete, prepare its search inputs:

```bash
python model_registry.py --prepare qwen15b --device cuda:0         # exact-FLOPs table and dense teacher logits
python model_registry.py --prepare llama31                          # same for every other key: re-tokenized windows, FLOPs table, teacher
```

OLMoE-1B-7B (expert groups of 16 experts, 11 levels; also needs the restored files):

```bash
PYTHONPATH=.. torchrun --nnodes=1 --nproc-per-node=4 -m evoprune.MoE.cli_build_db --model_name_or_path ../work/hf_models/OLMoE-1B-7B-0924 \
  --pre_block_modules model.embed_tokens model.rotary_emb --block_modules model.layers --calibration_data fineweb_edu \
  --calibration_tokens 8388608 --calibration_sequence_length 4096 --num_levels 11 --save_dir ../work/db/olmoe --dtype bfloat16 \
  --attn_implementation sdpa --expert_down_proj_regex '^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\.down_proj$' \
  --low_cpu_mem_usage --cpu_offload_modules --cpu_offload_activations
python olmoe_prepare.py tokens; python olmoe_prepare.py teacher; python olmoe_prepare.py groups     # OLMoE windows, teacher, expert groups
```

Run `python olmoe_prepare.py check` to see what is missing.

## Experiments

Every search is described by a job file (`configs/`), resumable, and writes `work/runs/<name>/{progress,results}.json`. Start any
search with

```bash
cd experiments
python run_job.py --job ../configs/section6_plain_ea/llama31/llama31_llm_nogp_s0.json --device cuda:0
```

(`run_job.py` picks `run_qwen15b.py`, `run_llama31.py`, `run_model.py`, `run_classic.py` or `run_olmoe.py`.)
Searches with LLM templates request new responses only with `"allow_api": true` in the job and an API key; the LLM
optimizer (`run_classic.py`) requests a response whenever it is not stored and an API key is set. With stored responses
(`work/api/`) both replay without requests.

| Paper part | Code | Job files |
|---|---|---|
| Sec. 4, optimizer comparison (EA with GP-guided mutation, GP-BO, TPE, DE, PSO, CMA-ES, GA, random, LLM as the optimizer incl. anonymized/other models) | `run_classic.py`, `classic_optimizers.py`, `llm_optimizer.py`, `gp_guide.py` (EA with GP-guided mutation via `run_qwen15b.py`/`run_llama31.py`) | `configs/section4_optimizers/` |
| Sec. 5 and 6.1, SPI vs. plain EA, random and own templates | `run_qwen15b.py`, `run_llama31.py`, `run_model.py`, `spi_search.py`, `llm_templates.py`, `template_reask.py`, `evoprune/serial_search.py` | `configs/section6_plain_ea/` |
| Sec. 6.2, template sources (own, transferred, hand-written, human depth) | same runners; "fixed" template rounds | `configs/section6_plain_ea/`, `guided_snapshot/` |
| Sec. 6.3, cost and inference speed | cost: `analysis/sec6_cost.py` from run records; speed: `speed.py`, `speed_unified.py`, `compact_gqa.py` | (none) |
| Sec. 6.4, downstream evaluation | `select_champions.py`, `heldout_qwen15b.py`, `heldout_models.py` (held-out KL), `make_benchmark_jobs.py`, `benchmark_allocation.py` + `benchmark.py` (compact export, perplexity, 8 lm-eval tasks), `benchmark_report.py` | `configs/benchmark/` |
| OLMoE (Appendix) | `olmoe_prepare.py`, `run_olmoe.py`, `olmoe_common.py`, `olmoe_benchmark.py` | `configs/moe/` |
| Recovery training (100M tokens) | `recovery_rows.py`, `recovery_job.py`, `run_fullft_recovery.py` | `configs/recovery/` |
| Sec. 7, analyses | `analysis/sec7_*.py`, `fig_match.py`, `fig_two_scales.py` | (run records) |
| Appendix C, hyperparameter screening | `run_qwen15b.py` | `configs/appendix_c_screening/` |

Downstream evaluation of one allocation: `python benchmark_allocation.py --job ../configs/benchmark/<job>.json`; the magnitude 2:4
baseline is `PYTHONPATH=.. python -m zipprune.SemiStructured.cli --model_name_or_path <dir> --output_dir <dir>`. Uniform allocation is a
benchmark job with the same level for every module: level 5, and level 8 for Qwen2.5-3B, which has 17 levels
(`configs/benchmark/*uniform*`). Paths in baseline jobs (`checkpoint`, `darwinlm`, `source` below `outputs/`) must be pointed to your own copies.

## Regenerating tables and figures

The run records are published on Zenodo (https://doi.org/10.5281/zenodo.24410518). Extract them to a directory `<records>` with the layout
documented in `analysis/records.py` (it equals the work directory layout, with FLOPs tables also under `flops/<model_key>.json`).

```bash
cd analysis
mkdir -p out
python sec4_llm_optimizer.py <records> out/sec4.json
python sec6_tables.py <records> out/sec6.json plain       # tab:spi-main; see the docstring of every script, and MAPPING.txt
python fig_kl_ppl.py <records> out/fig_kl_ppl.pdf
```

`analysis/MAPPING.txt` lists which table, figure or number each script produces and which run groups it reads.

## Third-party origins

- EvoPress (https://github.com/IST-DASLab/EvoPress, Apache-2.0): `evoprune/common_utils.py`, `evoprune/data_utils.py`,
  `evoprune/dist_utils.py`, `evoprune/linalg_utils.py`, `evoprune/metrics.py`, `evoprune/model_utils.py`, and the function
  `load_compressed_weights` in `experiments/benchmark.py`. The six files come to us through
  DarwinLM, which also added a few lines to them (`read_yaml_config`, `parse_block_size`, a seed argument for data loading);
  our own additions are the FineWeb-Edu chunk loaders and `barrier`. `evoprune/common_utils.py` also contains a function adapted
  from delu (https://github.com/Yura52/delu, MIT).
- DarwinLM (https://github.com/IST-DASLab/DarwinLM, no license declared): not redistributed. The level-database builders that we
  derived from its files are restored from a clone by `third_party/darwinlm/restore.py`; `patches.json` holds only line ranges,
  references to lines of DarwinLM files (some with changed indentation) and the lines we wrote.
- Developed for this study: the search, the evaluator and model stitching (`evoprune/`), compact export and recovery utilities
  (`zipprune/`), the MoE support, the experiments and the analysis scripts.
- Downstream evaluation uses EleutherAI's lm-evaluation-harness 0.4.5 (https://github.com/EleutherAI/lm-evaluation-harness, MIT),
  installed from pip.
- LLM-Pruner baselines were produced with the upstream implementation https://github.com/horseee/LLM-Pruner (Apache-2.0) at commit
  128a07d977f9b205d60ab14cfbc6a78f8a8e39d2 (files `llama3.py`, `LLMPruner/pruner/hf_llama_pruner.py`, `LLMPruner/torch_pruning/...`);
  it is not vendored here.

## Not included

Model weights, level databases, caches, run records, LLM response stores and API ledgers; the ZipLM one-shot search and LLM-Pruner baselines;
the scheduling scripts used to queue jobs on a shared machine; the earlier exploratory pilots.
