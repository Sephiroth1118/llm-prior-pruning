System prompts sent to the LLMs, as the code builds them. Files are written byte-exact (UTF-8, no conversion); the
presence or absence of a trailing newline is part of the string, because stored responses are keyed by a hash of the prompt.

serial_structure_system.txt   2076 chars, ends with "\n".  Structural prior injection (template generation), all models
    except OLMoE: the string SYSTEM of the serial-structure script, passed through system_prompt_for(per_round, rounds,
    alphabet, budget, refine). With the defaults of every search of the paper (two templates per round, two rounds, 16
    evaluations per template, alphabet [0,0.25,0.5,1,2,4]) no substitution happens, so this file is sent unchanged.
    Only for other settings the sentences "exactly TWO structures", "Two structures are tried per round; there are two
    rounds, then 240 evaluations of ordinary EA", "15 ordinary random mutations", "(16 total evaluations including the
    anchor)" and "The weight alphabet is [0,0.25,0.5,1,2,4]." are rewritten. The user message (the request: round,
    evaluations used, best KL, per-cell removable fractions, strong/weak/recent examples) is built in code.
    Paper version paper/prompts/llm_template_system.txt: IDENTICAL.

llm_template_system_moe.txt   2435 chars, ends with "\n".  OLMoE template prompt. Not a standalone string in the code: it is
    serial_structure_system.txt with five replacements (mixture-of-experts description inserted before "Use architectural
    priors ...", module order "groups 1-4 in the first eight layers, then groups 1-4 in the last eight layers", "per expert" /
    "of the expert parameters" instead of "per module" / "of the evaluator's dense FLOPs estimate", "group/depth" instead of
    "module/depth"); the code that does this is system_prompt() of the MoE helper module. The file is the result, equal to the
    approved prompt stored with the OLMoE runs (sha256 below is the approved hash).
    Paper version paper/prompts/llm_template_system_moe.txt: IDENTICAL.

llm_optimizer_system.txt   4347 chars, no trailing newline.  LLM as the optimizer (named gpt-5-mini, gpt-5.6-luna,
    DeepSeek-V4-Flash): the default system instruction of a previously published LLM policy-search agent, taken byte-exact
    BEFORE any edit. The code sends this text after three changes, made in code:
      (1) rule 12 ("12. The evaluator's reward already handles compression-budget error. Do not assume that a later repair
          step will alter your policy.") is replaced by
          "12. Every policy must have flops_reduction between 0.500 and 0.502 inclusive (exact database lookup). The task lists
          the fraction of the dense FLOPs removed by every level of every module; flops_reduction is their sum. Policies outside
          the band are returned to you with their exact flops_reduction for revision; policies still outside after two
          revisions are changed by a greedy repair."   (one line)
      (2) the line "- its evaluation metrics: kl, ppl, reward, flops_reduction" is replaced by
          "- its evaluation metrics: kl, reward (= -kl), flops_reduction (ppl is not measured in this run)"
      (3) "\n\nRun-specific policy bounds:\n" + json.dumps(compact) is appended, where compact is
          {"policy_type": "array_level_bounds", "num_layers": <L>, "default_bounds": {"attn_min": .., "attn_max": .., "mlp_min": ..,
          "mlp_max": ..} (the bounds of the first layer), "layer_overrides": {<layer>: bounds, for layers whose bounds differ}}
          (json.dumps default separators; "layer_overrides" is {} when all layers share the bounds).
    The text contains instructions for two search modes ("global_novelty", "nursery_probe") that this study never activates;
    they are part of what is sent, so they are kept here.
    Paper version paper/prompts/llm_optimizer_system.txt: NOT identical. It shows the text after changes (1) and (2) and an
    example of (3) (Llama-3.1-8B bounds), and replaces the two search-mode blocks (global_novelty, nursery_probe; 'When
    search_mode is "global_novelty": ...' to the end of the nursery_probe list) by the bracketed note
    "[Two blocks of instructions for search modes that this paper does not use are omitted here; the output format below
    still names their field.]". Apart from that, line by line equal to what the code sends.

llm_optimizer_system_anon.txt   1603 chars, no trailing newline.  The same optimizer with semantics removed (the "anonymized"
    arm, jobs with "anonymize": true): the string ANON_SYSTEM (the code applies .strip() to it). Positions are shuffled per seed
    and presented as a flat vector x; the cost table per position and value replaces the module names.
    Paper version paper/prompts/llm_optimizer_system_anon.txt: identical except that the paper file ends with one extra "\n".

Placeholders: none of these files contains a template field ({rounds}, {per_round}, {budget} belong to an alternative prompt
that no job of the paper uses, and are not included). Substitutions happen only through the replacements listed above.
User messages (requests) are assembled in code from the search state and are not text files.

None of the strings contains the name of the neighbouring project, a path, a name or an API key.

sha256 (of the file bytes):
  serial_structure_system.txt     021e4f43b48a4e6b5126db3f7c90b5dc31c4088673bcaa8fa5f8479da535f817
  llm_template_system_moe.txt     91390b577de4ebf93985e003f4432b7fcface9471b77fa729326a7c6c4a94310
  llm_optimizer_system.txt        762f09a6e2725b0f01b100fdf9ae09e296a297ae61b3e3f95c1edb35eb166097
  llm_optimizer_system_anon.txt   97b555e7d192dc1d95dfeed9b7e895b6f04f3792a59e15c74fd2c217f9c7e839
