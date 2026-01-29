# QLoRA Fine-Tuning With Stable FP16 Training and Memory-Safe Evaluation

## 1. One-Paragraph Overview

This repo contains a minimal QLoRA fine-tuning pipeline for a ~3B instruction model on low-VRAM GPUs (tested on GTX 1650 4GB). It includes FP16 overflow protection via AMP, memory-safe evaluation (inference-only, capped sequence length, batch=1), checkpoint-before-eval ordering, and run-scoped artifact export for Base vs SFT comparisons on EQ-Bench-style prompts plus qualitative and red-team prompts.

## 2. Key Features

- QLoRA fine-tuning (NF4 4-bit + LoRA) with configurable rank/alpha/dropout
- AMP (`autocast` + `GradScaler`) to prevent FP16 NaNs
- Memory-safe eval (batch=1, capped eval seq len, inference-only, no labels)
- Explicit VRAM cleanup (`gc.collect()` + `torch.cuda.empty_cache()` around eval)
- Checkpoints saved before evaluation
- Run-scoped outputs (JSONL exports, plots, stats, report)
- JSON config-driven reproducibility (`config/*.json`)

## 3. Hardware Requirements

- Minimum: GTX 1650 4GB (QLoRA only; full-precision training not feasible)
- Recommended: 8GB+ GPU for larger batches/seq lens and faster eval
- CPU: i5 or better
- RAM: 16GB recommended

## 4. Installation

```powershell
git clone <repo>
cd <repo>
pip install -r requirements.txt
```

Notes:

- Install PyTorch separately for your machine (CPU or CUDA). For 4-bit QLoRA you typically need CUDA.
- If you hit gated model/dataset access limits, add `HF_TOKEN` in a `.env` file (see `.env.example`).

## 5. Training Usage

Two commands (config-driven):

```powershell
python scripts/run_train.py --config config/example_config.json
```

Important config knobs (in `config/example_config.json`):

- `model.max_seq_len`
- `train.per_device_train_batch_size`
- `train.gradient_accumulation_steps`
- `train.learning_rate`
- `train.max_train_steps`
- `quant.load_in_4bit` (set `false` for CPU smoke)

CPU-only smoke test (pipeline verification only):

```powershell
python scripts/run_train.py --config config/cpu_smoke.json
```

Artifacts:

- Checkpoints: `artifacts/checkpoints/`
- Final adapter: `artifacts/checkpoints/final/adapter/` (+ `aux_heads.pt`)
- Training logs: `artifacts/logs/train_metrics.jsonl` and `artifacts/logs/eval_metrics.jsonl`
- Run manifest: `artifacts/run_manifest.json`

## 6. Evaluation Usage

Eval here means “export responses” (for external scoring) and generate plots/reports.

One-command pipeline (recommended):

```powershell
python scripts/run_pipeline.py --config config/example_config.json --adapter_dir artifacts/checkpoints/final/adapter --eqbench_limit 10
```

Windows tip (use venv Python to avoid calling the wrong interpreter):

```powershell
\.\.venv\Scripts\python.exe -u scripts/run_pipeline.py --config config/example_config.json --adapter_dir artifacts/checkpoints/final/adapter --eqbench_limit 10
```

Manual export (advanced):

```powershell
python scripts/run_eval.py --config config/example_config.json --eqbench --adapter_dir artifacts/checkpoints/final/adapter
python scripts/run_eval.py --config config/example_config.json --qual --redteam --adapter_dir artifacts/checkpoints/final/adapter
```

Clarification:

- Evaluation is inference-only by default (no labels) and is designed to avoid OOM on 4GB GPUs.

## 7. Repo Structure

```
.
├── training.py                 # Training loop (AMP, checkpoint-before-eval)
├── evaluation.py               # Export-based evaluation helpers
├── dataset.py                  # Dataset loading + canonical formatting
├── model.py                    # Model + PEFT loading
├── heads.py                    # Auxiliary heads (emotion/strategy)
├── config.py                   # Config dataclasses/merge
├── config/                     # JSON configs (example, smoke, ablations)
├── prompts/                    # Qualitative + red-team prompt JSONLs
├── scripts/
│   ├── run_train.py            # Main training entrypoint
│   ├── run_eval.py             # Export base vs SFT responses
│   ├── run_pipeline.py         # Exports → validate → plots → report
│   ├── validate_exports.py     # JSONL validation + pair checks
│   ├── plot_eval_comparisons.py# Eval plots (length + refusal heuristic)
│   ├── plot_loss_curves.py     # Training loss plot
│   ├── summarize_eval_stats.py # Stats JSON builder
│   ├── build_run_report.py     # Run-scoped report builder
│   ├── common.py               # Shared JSONL + heuristics
│   └── bootstrap.py            # Windows-safe env + .env loading
└── artifacts/                  # Logs, checkpoints, eval exports, plots, reports
```

## 8. Metrics & Results

Only meaningful metrics are included (no fake eval loss):

- Training loss curve: `artifacts/plots/loss_curves.png`
- Eval export stats: `artifacts/reports/run_<run_id>_stats.json`
- Run report: `artifacts/reports/run_<run_id>_report.md`

Example:

![Training Loss](artifacts/plots/loss_curves.png)

## 9. Reproducibility Notes

- Deterministic seeds: `train.seed`
- Model/tokenizer pinned via `model.base_model_id`
- QLoRA dtype: `bnb_4bit_compute_dtype=float16` + AMP in training
- Gradient accumulation: `train.gradient_accumulation_steps`
- VRAM-sensitive eval: inference-only + capped eval seq len + batch=1
- Run-scoped artifacts under `artifacts/eval/run_<run_id>/` and `artifacts/reports/`

## 10. Known Limitations

- GTX 1650 (4GB) limits max sequence length and batch sizes
- Eval loss is not computed by default (inference-only export)
- QLoRA only; full fine-tuning is not supported on low VRAM
- BF16 is typically not supported on GTX 1650; fp16 + AMP is used

### Large artifacts not pushed to Git

Some trained checkpoints and artifacts are intentionally not included in the GitHub repo because GitHub rejects files larger than 100MB. Example error encountered:

```
remote: error: File artifacts/checkpoints/final/adapter/adapter_model.safetensors is 114.25 MB; this exceeds GitHub's file size limit of 100.00 MB
remote: error: GH001: Large files detected. You may want to try Git Large File Storage - https://git-lfs.github.com.
! [remote rejected] main -> main (pre-receive hook declined)
```

Options:
- Reproduce locally by running training to regenerate `artifacts/checkpoints/**` and `artifacts/**` outputs.

## 11. Ablations

Provided configs:

- `config/ablation_no_emotion.json`
- `config/ablation_no_strategy.json`

Train:

```powershell
python scripts/run_train.py --config config/ablation_no_emotion.json
python scripts/run_train.py --config config/ablation_no_strategy.json
```

