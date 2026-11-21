Run configurations (job files) of the searches, benchmarks and recovery runs that the analysis scripts in ../analysis read.

Each file is the job file the run was started from, copied unchanged except for paths: absolute paths below the data
directory (the records directory of ../analysis/records.py) are made relative to it, e.g. "moe/expert_groups_fineweb.json",
"runs/llama_baseline_s0/results.json", "benchmark/llama31/dense". The file name is the run name, which is also the name of
the run directory <records>/runs/<name>/ (job file == <records>/runs/<name>/job.json). A search is started with
    python experiments/run_job.py --job <file> [--device cuda:0]
which picks the runner from job["runner"], job["optimizer"] or job["model"]:
    run_qwen15b.py    Qwen2.5-1.5B (jobs without a "model" key)
    run_llama31.py    Llama-3.1-8B (without a "runner" key)
    run_model.py      the other models of experiments/model_registry.py
    run_classic.py    jobs with an "optimizer" key (conventional and LLM optimizers; most of them also carry
                      "runner": "ar_run_classic.py", the runner's old name)
    run_olmoe.py      OLMoE (phase "MOE")
Benchmark jobs: experiments/benchmark_allocation.py --job; recovery jobs: experiments/recovery_job.py --job.
Relative paths inside job files are relative to the work directory ($SPI_WORK, see experiments/common.py).

Files per folder (1,083 job files in total)
  section4_optimizers/            186  EA with guided mutation (tuned2gp_baseline_s*, llama_tuned_baseline_s*), conventional
                                       optimizers (qwen_classic_*, llama_classic_*: random, tpe, gpbo, de, pso, ga, cmaes;
                                       gpbo_1k) and LLM optimizers (qwen_llmopt_*, llama_llmopt_*: gpt5mini, gpt5mini_anon,
                                       deepseek, luna; gpt5mini_1k)
  section6_plain_ea/<model>/      290  plain-EA searches (no guided mutation, groups *_nogp), 320 or 1000 evaluations:
                                       qwen15b 72, qwen3b 28, qwen7b 28, llama32_1b 30, llama32_3b 29, llama31 50, olmo1b 29, olmo7b 24
                                       (<model>_{ea,llm,own,humandepth}_nogp, <model>_tmpl_<source>_nogp, qwen15b/llama31_random_nogp,
                                       qwen15b_own_nogp_1k, qwen15b_tmpl_llama31_nogp_1k; re-ask reruns <group>_rk_s<seed>)
  guided_snapshot/                266  earlier snapshot of the matrix with guided mutation (llm_tuned2gp, tuned2gp_random,
                                       qwen_tmpl_*, llama_tuned_llm, llama_tmpl_*, <model>_tuned_baseline, <model>_tuned_llm,
                                       <model>_tmpl_*)
  moe/                            21   OLMoE: 20 search jobs (olmoe_{ea,llm,own,random,rule}_nogp_s0-3) and own_template.json
  moe/bench_jobs/                  3   OLMoE downstream benchmark jobs (dense, uniform_l5, ours_s3)
  benchmark/                      80   downstream benchmark jobs (<model>__<job>.json): dense, uniform, magnitude 2:4, DarwinLM,
                                       the champions of the searches (compression_<run>, <group>_s<seed>) and the recovery rows;
                                       read through benchmark/<model>/<job>/benchmark_results.json and reports/benchmark/metrics.json
  recovery/                        4   post-pruning recovery training jobs (llmpruner, llmpruner_official, ours, uniform)
  appendix_c_screening/          233   Appendix C hyperparameter screening on Qwen2.5-1.5B: 180 one-factor screening searches (hpo_*) and
                                       53 confirmation searches (llmgp_*, tuned2*, tuned3*); the reference configurations they are compared
                                       with are runs of the main matrix (see the paper's appendix)
Each job file equals the job.json stored in its run directory. The 18 runs tuned2gp_baseline_s* and tuned2gp_random_s*
belong to two parts of the paper, so each of them has a copy in appendix_c_screening/ and one in section4_optimizers/ or
guided_snapshot/; the two copies are identical (1,065 distinct runs and own_template.json).

Run group -> paper section / table / script (groups are <prefix>_s<seed>; every seed that exists is included)
  <model>_{ea,llm,own,humandepth}_nogp, <model>_tmpl_<source>_nogp, qwen15b_random_nogp, llama31_random_nogp
        Section 5/6/7: tab:spi-main, tab:matrix, tab:matrix-summary, fig_matrix, fig_spi_gain, fig_match, fig_two_scales,
        tab:cost, sec7_robustness (sec6_tables.py, sec6_matrix_plain.py, sec6_sources_text.py, fig_matrix.py,
        fig_spi_gain.py, sec7_analysis.py, sec7_robustness.py, sec6_cost.py); sec5_macro_profile.py and sec7_llm_attention.py
        read qwen15b_ea_nogp / llama31_ea_nogp
  qwen15b_tmpl_llama31_nogp_1k, qwen15b_own_nogp_1k, qwen_classic_gpbo_1k, qwen_llmopt_gpt5mini_1k
        1000-evaluation cells of tab:spi-main, fig_spi_gain (Qwen2.5-1.5B panel), sec6_cost
  qwen_classic_*, llama_classic_*, qwen_llmopt_*, llama_llmopt_*, tuned2gp_baseline, llama_tuned_baseline
        Section 4: tab:llm-opt, tab:llm-feasibility, fig_optimizers_gain, sec4_llm_optimizer.py, sec4_numbers.py,
        sec5_macro_profile.py, sec7_llm_attention.py (tab:llm-attention), sec6_cost.py
  llm_tuned2gp, tuned2gp_random, qwen_tmpl_own, qwen_tmpl_transfer, qwen15b_tmpl_olmo7b, llama_tuned_llm,
  llama_tmpl_own, llama_tmpl_transfer, llama31_tmpl_olmo7b, <model>_tuned_baseline, <model>_tuned_llm,
  <model>_tmpl_{own,human,humandepth,<source>}
        guided-mutation snapshot: sec6_tables.py and sec7_analysis.py without a mode
  olmoe_{ea,llm,own,random,rule}_nogp (moe/)
        tab:moe (tab_downstream.tex); no analysis script reads these runs,
        the table was filled from <records>/moe/runs/*/results.json and moe/benchmark
  benchmark/ and recovery/ jobs
        fig_kl_ppl (reports/benchmark/metrics.json, all rows of qwen15b and llama31), sec6_downstream_seeds
        (tab:downstream-seeds), tab_downstream.tex (tab:llama-downstream, tab:cross-downstream, tab:posttrain)

Some groups named by the guided-snapshot code were never run (no run directory, hence no job file); the scripts skip them.
Every group that has run directories has a job file. Group names indicate the method:
  qwen_classic_<o>/llama_classic_<o>   conventional optimizer <o> (random, tpe, gpbo, de, pso, ga, cmaes)   run_classic.py
  *_llmopt_*                           LLM as the optimizer (optimizer "llm")                                run_classic.py
  *_nogp, *_nogp_1k, *_rk_s*           plain (mu+lambda) EA, with or without template rounds; 1k = 1000 evaluations; _rk_ = re-ask rerun
  *_tuned_*, *_tmpl_*                  EA with GP-guided mutation (earlier snapshot) and template-source variants
  olmoe_*                              OLMoE                                                                  run_olmoe.py

Job-file keys that name other files or runs (the code has to resolve them)
  search jobs:  "expert_groups" (OLMoE only): path of the expert grouping file, relative to the data directory
                (moe/expert_groups_fineweb.json; not copied here, 50 KB, produced by the OLMoE pipeline);
                "pair": name of the job whose 16 initial allocations are shared (a run in runs/); "extends": run this 1000-evaluation
                run continues; "source_run": run a re-ask rerun was derived from; "runner": script that executes the job;
                "rounds[].templates" (fixed/own templates, inline); own_template.json is the OLMoE own template (inline in the job too).
  benchmark jobs (benchmark/, moe/bench_jobs/): "source": a runs/<run>/results.json, a benchmark/<model>/<job>/compact_model
                directory, or a path below outputs/ (see below); "out_dir": benchmark/<model>/<job>; "checkpoint", "darwinlm"
                (+ "revision"): baseline checkpoints; "policy": the allocation (inline); "expert_groups" as above.
  recovery jobs: "source" (compact model), "policy_file", "one_shot_results": paths below benchmark/ or outputs/.

PATHS NOT BELOW THE DATA DIRECTORY (reported, not private data): the original job files also contain absolute paths below the
project root, outside the data directory. These were made relative (prefix removed) and are resolved against the
work directory, like all other paths; the values start with "outputs/" and must be edited to point to your local copies of these
baseline checkpoints:
  outputs/serial_structure_b1000_qwen15b_20261010/{llm,random}/seed<k>/results.json   (benchmark jobs qwen15b__llm_orig_s*, random_orig_s*; original serial-structure runs)
  outputs/semi_structured/2to4_*_run01/checkpoint_2to4                                (magnitude 2:4 checkpoints)
  outputs/llmpruner_reproduction/llama31_target4.6b_bookcorpus_seed42[_gqa_corrected]/ (LLM-Pruner reproduction; recovery and benchmark jobs)
  outputs/cross_model_baselines/model_cache/models--Shengkun--DarwinLM-4.6B-Pruned/snapshots/<hash>   (public DarwinLM checkpoint)
Notes inside two values were removed (a plan reference in the rationale of moe/olmoe_rule_nogp_s*.json, a note in the method string of
recovery/llama31__recovery_llmpruner_official.json); no value that the code reads was changed.
No other absolute path, user name or credential occurs in these files.
