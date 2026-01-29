# Step 1 — Dataset Sourcing, Verification, and Final Selection (Jan 2026)

This document is an auditable record of the actions performed to source, test, and finalize the public datasets used in this project.

## 1. Objective

Select public (non-proprietary) datasets that support the project’s training design:

- Supervised fine-tuning (SFT) with an LM objective on assistant/supporter responses
- An auxiliary emotion classification head (supervising the user’s latest turn)
- An auxiliary support strategy classification head (supervising the intended strategy for the next assistant/supporter turn)

Key constraints:

- Use a modern Hugging Face `datasets` workflow (avoid deprecated dataset-script loading)
- Prevent label leakage into model inputs

## 2. Environment and tooling

### 2.1 Installed dependencies

The following packages were installed/updated:

```bash
pip install -U datasets transformers accelerate peft bitsandbytes
```

Operational note:

- Windows may emit symlink cache warnings from `huggingface_hub`; downloads still function correctly.

### 2.2 Inspection scripts

Two repository-local scripts were used:

- [scripts/inspect_datasets.py](scripts/inspect_datasets.py) — loads datasets and prints split names and sample keys (keys only)
- [scripts/materialize_canonical_samples.py](scripts/materialize_canonical_samples.py) — materializes one canonical sample per dataset and prints leak-safe summaries (roles + text lengths + masking statistics)

## 3. Candidate datasets

Initial candidate repositories provided:

- EmpatheticDialogues: `facebook/empathetic_dialogues`
- ESConv: `thu-coai/esconv`, `Ashokajou51/ESConv_Original`, `Ashokajou51/ESConv_Sorted`
- GoEmotions: `google-research-datasets/go_emotions`

## 4. Dataset verification results

### 4.1 EmpatheticDialogues

#### 4.1.1 Primary candidate: `facebook/empathetic_dialogues`

Result: not usable with the current `datasets` version.

Observed failure:

- `RuntimeError: Dataset scripts are no longer supported, but found empathetic_dialogues.py`

Repository contents confirmed:

- `README.md`
- `empathetic_dialogues.py` (legacy dataset script)

Conclusion:

- This repo does not provide data files (e.g., Parquet/JSON/CSV) compatible with modern `datasets` loading.

#### 4.1.2 Tested compatible mirrors

Multiple mirrors were tested to identify a modern, data-file-based replacement.

- `lighteval/empathetic_dialogues`
  - Loads successfully
  - Columns: `input`, `references`, `subsplit`
  - Limitation: no explicit emotion label field

- `pixelsandpointers/empathetic_dialogues_for_lm`
  - Loads successfully
  - Columns: `conv`
  - Limitation: no explicit emotion label field

- `Adapting/empathetic_dialogues_v2`
  - Loads successfully
  - Splits: train/validation/test
  - Contains explicit emotion labels
  - Sample keys (keys only): `id`, `chat_history`, `sys_response`, `situation`, `emotion`, `question or not`, `behavior`

Decision:

- Use `Adapting/empathetic_dialogues_v2` as the EmpatheticDialogues-compatible source for SFT dialogue data.

Leakage control for ED-v2:

- Prompt/context is built from `chat_history` only
- Assistant target is `sys_response`
- Do not include: `emotion`, `behavior`, `question or not`, `situation` in model inputs

### 4.2 ESConv

#### 4.2.1 Primary candidate: `thu-coai/esconv`

Result: usable and selected.

Observed schema:

- Splits: train/validation/test
- Columns: `text`

Important detail:

- Each row’s `text` is a JSON object encoded as a string.
- Parsed top-level keys (keys only): `experience_type`, `emotion_type`, `problem_type`, `situation`, `survey_score`, `dialog`, plus question fields.
- `dialog` is a list of turns; turn keys include `text`, `speaker`, and sometimes `strategy`.

Conclusion:

- ESConv provides per-assistant/supporter-turn strategy annotations via `dialog[i].strategy`, which is required for the strategy head.

Leakage control for ESConv:

- Inputs are built only from the dialog turn texts and speaker roles
- Strategy labels are used only as `strategy_label` targets
- Do not inject top-level metadata (`emotion_type`, `problem_type`, `situation`, etc.) or `strategy` text into prompts

#### 4.2.2 Fallback candidates

Two additional repos were validated as viable fallbacks:

- `Ashokajou51/ESConv_Original`
- `Ashokajou51/ESConv_Sorted`

Observed schema:

- Columns include: `emotion_type`, `problem_type`, `situation`, `dialog`
- Dialog turn keys include: `text`, `speaker`, `knowledge`, `strategy`, `heal`

Decision:

- Keep these as fallbacks; default to `thu-coai/esconv`.

### 4.3 GoEmotions

Candidate: `google-research-datasets/go_emotions`

Result: usable and selected.

Observed schema:

- Splits: train/validation/test
- Columns: `text`, `labels`, `id`

Label cardinality (sampled 1000 examples per split during inspection):

- Train: ~83% single-label, ~17% multi-label
- Validation: ~81% single-label, ~19% multi-label
- Test: ~85% single-label, ~15% multi-label

Decision:

- For a clean CE-based emotion head, filter to `len(labels) == 1` and set `emotion_label = labels[0]`.

Leakage control for GoEmotions:

- Input is `text` only
- Labels remain separate (`emotion_label`)

## 5. Final dataset selection

The finalized dataset sources used by this project are:

- Empathy dialogue SFT: `Adapting/empathetic_dialogues_v2`
- Emotional support + strategy supervision: `thu-coai/esconv`
- Auxiliary emotion supervision: `google-research-datasets/go_emotions`

## 6. Label mapping policy (finalized)

### 6.1 Emotion head

- Primary emotion taxonomy: GoEmotions label IDs
- Default behavior:
  - Use GoEmotions single-label subset (`len(labels) == 1`) for CE
  - Do not supervise the emotion head from ED-v2 or ESConv by default (taxonomy mismatch risk)

Optional extension (not enabled by default):

- Add an explicit mapping table from a subset of ED-v2 emotion strings → GoEmotions labels, and drop unmapped examples.

### 6.2 Strategy head

- Primary strategy taxonomy: ESConv `dialog[i].strategy` values
- Default behavior:
  - Train strategy head only on assistant/supporter turns with a non-null strategy
  - Mask strategy loss for all other examples

### 6.3 Train-only vocab construction (anti-leakage)

- Any label vocabulary (e.g., ESConv strategy string → ID) is built from the train split only.

## 7. Canonical schema verification (no label leakage)

To validate that our preprocessing produces clean training inputs, we materialized canonical examples using:

- [scripts/materialize_canonical_samples.py](scripts/materialize_canonical_samples.py)

The script prints:

- canonical keys (`messages`, `input_ids`, `labels`, `emotion_label`, `strategy_label`)
- masked-label statistics showing that prompt tokens are excluded from LM loss
- only role + character counts (not raw text/labels), ensuring the verification step itself does not leak labels

## 8. Artifacts produced

- [scripts/inspect_datasets.py](scripts/inspect_datasets.py)
- [scripts/materialize_canonical_samples.py](scripts/materialize_canonical_samples.py)
- This documentation file.


# Step 2 — Model, Losses, and Training Design (Jan 2026)

This section documents how the implementation satisfies the core modeling and training requirements from
[Projects_requirements_specifications.md](Projects_requirements_specifications.md).

## 1. Compute & base model setup

- Track: Low-compute QLoRA / PEFT on a ~3B open-weights instruction model.
- Default base model: `Qwen/Qwen2.5-3B-Instruct` (Hugging Face Hub).
- Loader implementation: [model.py](model.py)
  - Uses `transformers.AutoModelForCausalLM.from_pretrained` with `output_hidden_states=True` and `use_cache=False`.
  - QLoRA support via `bitsandbytes` 4-bit quantization when CUDA is available, with configuration driven by
    `QuantizationConfig` in [config.py](config.py).
  - PEFT / LoRA adapters implemented with `peft.get_peft_model` using a `LoraConfig` built in [model.py](model.py).
- CPU-only smoke path:
  - When CUDA is not available, `quant.load_in_4bit` must be set to `false` to avoid bitsandbytes errors.
  - A minimal CPU sanity configuration is provided in [config/cpu_smoke.json](config/cpu_smoke.json) to exercise the
    full data → model → loss → logging pipeline without claiming to be a proper QLoRA run.

Design note:

- The architecture and configuration objects are intentionally "architecture fluent" for QLoRA and PEFT even if
  environmental constraints (e.g., CPU-only) prevent true 4-bit training on a given machine.

## 2. Model architecture and auxiliary heads

### 2.1 Core decoder and LM head

- The base decoder and LM head are provided by the chosen HF model (e.g., Qwen 2.5 Instruct) and exposed via
  `AutoModelForCausalLM` in [model.py](model.py).
- We supervise the LM head on assistant tokens only, using a masked language modeling objective where prompt tokens
  are assigned an IGNORE_INDEX label.

### 2.2 Auxiliary heads

Implemented in [heads.py](heads.py):

- Emotion classification head
  - A linear layer mapping from the hidden size of the base model to `num_emotions`.
  - Supervised on an integer `emotion_label` field in the canonical batch.
- Support-strategy classification head
  - A linear layer mapping from hidden size to `num_strategies`.
  - Supervised on an integer `strategy_label` field, derived from ESConv `strategy` annotations.

Pooling strategy:

- Both heads consume a pooled representation of the last prompt token:
  - `pool_last_prompt_token` in [heads.py](heads.py) uses `prompt_length - 1` as the index.
  - This prevents leakage from assistant target tokens into the classification features.

Masking and loss computation:

- Cross-entropy losses are computed via `masked_cross_entropy` in [heads.py](heads.py):
  - Examples with IGNORE_INDEX labels are excluded from loss aggregation.
  - This ensures that, for example, strategy loss is only applied on assistant/supporter turns with a valid strategy
    label.

## 3. Loss functions and weighting

Implemented in [training.py](training.py) with configuration defined in [config.py](config.py).

- Primary LM loss: token-level negative log-likelihood on assistant response tokens.
- Auxiliary emotion loss: CE over emotion classes.
- Auxiliary strategy loss: CE over strategy classes.

The total loss is a weighted sum:

$$
L_{total} = \lambda_{lm} L_{lm} + \lambda_{emo} L_{emo} + \lambda_{strat} L_{strat}
$$

- Weights are configured via the `LossWeights` dataclass inside `TrainConfig` in [config.py](config.py):
  - `loss_weights.lm`
  - `loss_weights.emotion`
  - `loss_weights.strategy`
- The training loop in [training.py](training.py) reads these weights and combines the individual loss scalars
  accordingly.

Explicit skips (per specification):

- Valence/arousal regression head: **not implemented**.
- Safety teacher KL loss: **not implemented**.

However, the configuration surface includes placeholder structures (e.g., `SafetyTeacherConfig`, `DPOConfig`) to make
future extensions straightforward without refactoring the core training loop.

## 4. Canonicalization and dataset mixing

Canonical schema implementation: [dataset.py](dataset.py).

- All raw datasets are mapped into a unified format with fields:
  - `messages`: list of chat messages with `role` and `content`.
  - `input_ids`, `attention_mask`, `labels`: tokenized inputs and masked LM labels.
  - `prompt_length`: index separating prompt tokens from assistant target tokens.
  - `emotion_label`, `strategy_label`: optional integer labels for auxiliary heads.

Dataset-specific behavior:

- `Adapting/empathetic_dialogues_v2`
  - Uses `chat_history` for context and `sys_response` as the assistant target.
  - Drops `emotion`, `behavior`, `question or not`, `situation` from inputs.
- `thu-coai/esconv`
  - Parses JSON-encoded dialog turns and extracts `strategy` for supporter turns.
  - Uses only turn texts and speaker roles as model inputs.
- `google-research-datasets/go_emotions`
  - Filters to `len(labels) == 1` and uses `labels[0]` as `emotion_label`.
  - Inputs are the raw `text` only.

Mixture sampling implementation: [mixture.py](mixture.py).

- Uses temperature-based sampling with parameter $\alpha \in (0, 1]$ as required:
  - Per-dataset probabilities are proportional to $n_i^{\alpha}$, where $n_i$ is the number of examples in dataset $i$.
- The sampler is framework-agnostic and feeds an iterable dataset used in [training.py](training.py), ensuring that no
single dataset dominates training even when sizes differ substantially.

## 5. Training loop and hyperparameters

Training entrypoint: [scripts/run_train.py](scripts/run_train.py).

- Loads configuration via [config.py](config.py), with main example in
  [config/example_config.json](config/example_config.json).
- Builds canonical datasets and mixture dataloaders via [dataset.py](dataset.py) and [mixture.py](mixture.py).
- Loads model, tokenizer, and attaches LoRA adapters via [model.py](model.py).
- Constructs auxiliary heads and orchestrates the multi-objective loss in [training.py](training.py).

Key hyperparameters (config-driven):

- `TrainConfig.max_train_steps`: step-based budget instead of epoch-based control.
- `TrainConfig.max_train_examples`, `max_eval_examples`: optional caps for data used per run.
- `TrainConfig.gradient_accumulation_steps`, `per_device_batch_size`.
- Learning rate, warmup ratio, and scheduler parameters.

Logging and artifacts:

- Training metrics appended to:
  - `artifacts/logs/train_metrics.jsonl`
  - `artifacts/logs/eval_metrics.jsonl`
- Run manifest stored at `artifacts/run_manifest.json` with:
  - Model ID, config snapshot, git hash, device, mixture parameters, and loss weights.
- Checkpoints written under `artifacts/checkpoints/`, with the final adapter at
  `artifacts/checkpoints/final/adapter/` plus `aux_heads.pt`.

Windows / low-resource considerations:

- Entry scripts set `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `TOKENIZERS_PARALLELISM=false`, and `RAYON_NUM_THREADS=1`
  prior to imports to mitigate "OS can't spawn worker thread" errors on Windows.

## 6. Ablations

Ablation configurations required by the specification are provided under `config/`:

- Remove emotion head: [config/ablation_no_emotion.json](config/ablation_no_emotion.json)
  - Sets the emotion loss weight to zero and/or disables the emotion head while keeping the rest of the pipeline
    unchanged.
- Remove strategy head: [config/ablation_no_strategy.json](config/ablation_no_strategy.json)
  - Analogous removal of the strategy loss and head.

Each ablation is trained with `scripts/run_train.py` and evaluated with `scripts/run_eval.py`, writing separate
checkpoint and eval directories (e.g., `artifacts/checkpoints_ablation_*`, `artifacts/eval_ablation_*`).

## 7. Evaluation protocol and deliverables

Evaluation and export logic: [evaluation.py](evaluation.py).

- EQ-Bench-style prompt export (primary comparison set):
  - Default dataset ID (configurable): `pbevan11/EQ-Bench`, split `validation`.
  - Rationale: the official EQ-Bench repo IDs on Hugging Face may be gated/inaccessible in some environments.
  - Exports prompts and model responses for both the base model and the SFT adapter.
- Qualitative analysis:
  - Uses prompts in [prompts/qualitative_prompts.jsonl](prompts/qualitative_prompts.jsonl) to generate side-by-side
    conversations (base vs SFT) for manual inspection.
- Safety / red-team analysis:
  - Uses [prompts/redteam_prompts.jsonl](prompts/redteam_prompts.jsonl) to elicit potentially unsafe behavior and
    document expected vs actual outputs.

All exports are JSONL files under `artifacts/eval/`, each row containing:

- `prompt_id`, `prompt`, `model`, `response`, `meta`, `decoding`.

Additional tooling:

- [scripts/validate_exports.py](scripts/validate_exports.py)
  - Validates schema and, with `--check_pairs`, enforces base vs SFT prompt parity.
- [scripts/plot_loss_curves.py](scripts/plot_loss_curves.py)
  - Renders training/eval loss curves from JSONL logs.
- [scripts/plot_eval_comparisons.py](scripts/plot_eval_comparisons.py)
  - Generates comparison plots from exported JSONLs (response length distributions + a refusal/deflection heuristic).
- [scripts/summarize_eval_stats.py](scripts/summarize_eval_stats.py)
  - Writes a JSON summary of evaluation exports (counts + average response lengths + red-team refusal heuristic).
- [scripts/build_report.py](scripts/build_report.py)
  - Generates a Markdown report stub summarizing EQ-Bench slots and qualitative / safety comparisons.
- [scripts/run_pipeline.py](scripts/run_pipeline.py)
  - One-command runner that executes exports → validation → plots → report and saves run-scoped artifacts.
- [scripts/build_run_report.py](scripts/build_run_report.py)
  - Generates a “fresh document” Markdown report for a specific run folder.

## 8. Preference alignment and safety (status)

Per the requirements, DPO and safety-KL are **optional but rewarded**. In this implementation:

- DPO: configuration scaffolding exists (`DPOConfig`), but no end-to-end DPO training loop has been implemented.
- Safety teacher KL: not implemented; only the standard SFT objectives are active.

The codebase is structured so that adding a DPO phase or safety-regularized loss would primarily involve:

- Implementing an additional training script (e.g., `scripts/run_dpo.py`).
- Reusing the model loading and dataset infrastructure.
- Adding a DPO-specific loss computation using preference pairs.

## 9. Reproducibility and limitations

Reproducibility:

- All runs are driven by explicit JSON configs under `config/`.
- The run manifest records the effective configuration, model ID, and git hash at training time.
- Training logs (JSONL) and optional loss-curve plots provide a transparent view into optimization behavior.

Current limitations:

- True QLoRA training requires a CUDA-capable environment with bitsandbytes; the CPU-only path is intended for smoke
  tests and architectural validation, not for competitive performance.
- DPO, safety KL, and reflection-based decoding policies are not yet implemented; the current system focuses on
  multi-objective SFT with auxiliary emotion and strategy heads.

These limitations are documented explicitly to keep the implementation aligned with the "explicit skip but architecturally
fluent" requirement in the original specification.


# Step 3 — Engineering Log: Issues Encountered & Fixes Applied (Jan 2026)

This section is an auditable record of key runtime issues encountered (Windows + constrained GPU) and the fixes applied.

## 1. CPU-only environment vs QLoRA requirement

- Symptom: 4-bit quantization (bitsandbytes) fails on CPU-only PyTorch.
- Root cause: bitsandbytes 4-bit path is CUDA-oriented.
- Fix:
  - Added CPU smoke config: [config/cpu_smoke.json](config/cpu_smoke.json) (tiny model, no 4-bit) to validate pipeline wiring.
  - Kept the code architecture QLoRA/PEFT-ready for CUDA environments.

## 2. GPU OOM / VRAM constraints (GTX 1650 4GB)

- Symptoms:
  - Training/eval OOM when max sequence length and/or eval batch sizes were too large.
- Fixes:
  - Capped `model.max_seq_len` (example config uses 1024).
  - Reduced eval set sizes and used shorter `max_new_tokens` defaults.
  - Eval path runs inference-only (no LM loss computation during eval) to reduce peak memory.
  - Added cache clearing before eval (`gc.collect()` + `torch.cuda.empty_cache()`).

## 3. FP16 instability / NaNs on small GPUs

- Symptom: occasional NaNs during fp16 training.
- Fix: Enabled AMP (`autocast` + `GradScaler`) and kept conservative gradient clipping.

## 4. Windows stability / thread spawning

- Symptom: intermittent "OS can't spawn worker thread"-style failures.
- Fix: Centralized conservative env defaults (threads + allocator) and `.env` loading via [scripts/bootstrap.py](scripts/bootstrap.py), used by [scripts/run_train.py](scripts/run_train.py) and [scripts/run_eval.py](scripts/run_eval.py).

## 5. Evaluation exporter non-zero exit / crash after completing work

- Symptom: evaluation exports sometimes completed (JSONL written) but the process exited non-zero or crashed on Windows.
- Root cause hypothesis: repeated full model loads in a single process increased memory fragmentation / instability.
- Fix: Refactored [evaluation.py](evaluation.py) to load the base model once, reuse it for all prompt sets, and then attach the adapter in-process for SFT exports.

## 6. EQ-Bench dataset access

- Symptom: attempts to use certain EQ-Bench dataset IDs failed due to access restrictions.
- Fix: switched prompt export to a public dataset mirror (`pbevan11/EQ-Bench`, split `validation`) and made dataset ID configurable via config/CLI.
