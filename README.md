# Empathetic Chatbot: QLoRA Fine-Tuning Pipeline

> **Hardware**: GTX 1650 (4GB VRAM) | **Model**: Qwen/Qwen2.5-3B-Instruct | **Framework**: PyTorch + HuggingFace + PEFT

---

## 1. Overview

This repo implements a **QLoRA fine-tuned empathetic chatbot** with:

- **Multi-objective loss**: Language modeling + emotion classification + strategy prediction
- **Auxiliary heads**: EmotionHead (32 classes) + StrategyHead (8 classes)
- **Safety KL regularization**: Implemented (optional, hardware-dependent)
- **EQ-Bench evaluation**: Emotional intelligence scoring (+8.3 improvement over base)
- **Two-step decoding**: Style tokens + internal reflection controller

| Metric | Base | SFT | Δ |
|--------|------|-----|---|
| EQ-Bench Score | 76.4 | **84.7** | +8.3 |

---

## 2. Quick Start

### Single Entry Point

All operations run through `main.py`:

```powershell
# Run everything (train + eval + eq-bench + pipeline)
python main.py all --config config/example_config.json

# Individual commands
python main.py train --config config/example_config.json     # Training only
python main.py eval --config config/example_config.json      # Evaluation export
python main.py eq-bench --config config/example_config.json  # EQ-Bench evaluation
python main.py pipeline --config config/example_config.json  # Full pipeline
```

### Installation

```powershell
git clone <repo>
cd <repo>
pip install -r requirements.txt
```

> **Note**: Install PyTorch separately for your CUDA version. For 4-bit QLoRA, CUDA is required.

---

## 3. Architecture

### Multi-Objective Loss

$$\mathcal{L}_{\text{SFT}} = \lambda_{\text{LM}} \cdot \mathcal{L}_{\text{NLL}} + \lambda_{\text{emo}} \cdot \mathcal{L}_{\text{emo}} + \lambda_{\text{strat}} \cdot \mathcal{L}_{\text{strat}} + \lambda_{\text{safe}} \cdot \mathcal{L}_{\text{safe}}$$

| Loss Term | Weight (λ) | Description |
|-----------|------------|-------------|
| LM (NLL) | 1.0 | Causal language modeling |
| Emotion | 0.2 | Emotion classification (32 classes) |
| Strategy | 0.2 | Support strategy prediction (8 classes) |
| Safety KL | 0.0 | Regularization toward base model (disabled by default) |

### QLoRA Configuration

| Parameter | Value |
|-----------|-------|
| Quantization | NF4 4-bit, double quant |
| LoRA Rank (r) | 16 |
| LoRA Alpha | 32 |
| LoRA Dropout | 0.05 |
| Compute Dtype | FP16 |

---

## 4. Configuration

### Config Files

| File | Purpose |
|------|---------|
| `config/example_config.json` | Main training config |
| `config/ablation_no_emotion.json` | Ablation: no emotion head |
| `config/ablation_no_strategy.json` | Ablation: no strategy head |
| `config/safety_enabled.json` | With Safety KL enabled |
| `config/cpu_smoke.json` | CPU-only smoke test |

### Safety KL Configuration

| Config Setting | Effect |
|----------------|--------|
| `"safety": { "enabled": false }` | **Skip** Safety KL (default) |
| `"safety": { "enabled": true }` | **Enable** Safety KL |
| `"losses": { "lambda_safe": 0.1 }` | Weight for safety loss |

---

## 5. Repo Structure

```
.
├── main.py                      # Single entry point (train/eval/eq-bench/pipeline/all)
├── training.py                  # Training loop with multi-objective loss
├── evaluation.py                # Export-based evaluation helpers
├── heads.py                     # Auxiliary heads (EmotionHead, StrategyHead)
├── safety_teacher.py            # Safety KL regularization module
├── decoding_policy.py           # Style tokens + two-step controller
├── eq_bench.py                  # EQ-Bench emotional intelligence evaluation
├── dataset.py                   # Dataset loading + formatting
├── model.py                     # Model + PEFT loading
├── config.py                    # Configuration dataclasses
├── config/                      # JSON configs
├── prompts/                     # Qualitative + red-team prompts
├── scripts/
│   ├── run_train.py             # Training entrypoint (legacy)
│   ├── run_eval.py              # Export responses (legacy)
│   ├── run_eq_bench.py          # EQ-Bench script
│   ├── run_pipeline.py          # Full pipeline
│   ├── validate_exports.py      # JSONL validation
│   ├── plot_loss_curves.py      # Loss curve plotting
│   ├── plot_eval_comparisons.py # Eval comparison plots
│   ├── summarize_eval_stats.py  # Stats builder
│   ├── build_run_report.py      # Report builder
│   ├── common.py                # Shared utilities
│   └── bootstrap.py             # Environment setup
├── artifacts/                   # Outputs (logs, checkpoints, plots, reports)
├── FINAL_SUBMISSION.md          # Complete technical documentation
└── README.md                    # This file
```

---

## 6. Ablation Studies

Run ablations to test component contributions:

```powershell
# Remove emotion head (λ_emo = 0)
python main.py train --config config/ablation_no_emotion.json

# Remove strategy head (λ_strat = 0)
python main.py train --config config/ablation_no_strategy.json

# Enable Safety KL (λ_safe = 0.1)
python main.py train --config config/safety_enabled.json
```

| Ablation | λ_emo | λ_strat | λ_safe |
|----------|-------|---------|--------|
| Full Model | 0.2 | 0.2 | 0.0 |
| No Emotion | 0.0 | 0.2 | 0.0 |
| No Strategy | 0.2 | 0.0 | 0.0 |
| With Safety | 0.2 | 0.2 | 0.1 |

---

## 7. Evaluation

### EQ-Bench Evaluation

```powershell
python main.py eq-bench --config config/example_config.json
```

Results saved to:
- `artifacts/eval/eq_bench_base.json`
- `artifacts/eval/eq_bench_sft.json`
- `artifacts/eval/eq_bench_report.txt`

### Metrics

| Metric | Description |
|--------|-------------|
| EQ-Score | Composite emotional intelligence score (0-100) |
| MAE | Mean Absolute Error of emotion intensity predictions |
| Correlation | Pearson correlation with ground truth |

---

## 8. Artifacts

| Artifact | Path |
|----------|------|
| Training logs | `artifacts/logs/train_metrics.jsonl` |
| Eval logs | `artifacts/logs/eval_metrics.jsonl` |
| Loss curves | `artifacts/plots/loss_curves.png` |
| Final adapter | `artifacts/checkpoints/final/adapter/` |
| Run manifest | `artifacts/run_manifest.json` |
| EQ-Bench results | `artifacts/eval/eq_bench_*.json` |

---

## 9. Hardware Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | GTX 1650 (4GB) | 8GB+ VRAM |
| CPU | Intel i5 | Intel i7+ |
| RAM | 16 GB | 32 GB |
| CUDA | 11.8+ | 12.0+ |

---

## 10. Known Limitations

- **VRAM**: GTX 1650 (4GB) limits batch size and sequence length
- **Safety KL**: Implemented but skipped during training due to VRAM constraints
- **DPO**: Config exists but not integrated (requires reference model)
- **BF16**: Not supported on GTX 1650; FP16 + AMP used instead

### Large Files

Trained checkpoints exceed GitHub's 100MB limit. Regenerate locally:

```powershell
python main.py all --config config/example_config.json
```

---

## 11. Documentation

For complete technical details, see [FINAL_SUBMISSION.md](FINAL_SUBMISSION.md):

- Multi-head implementation details with code snippets
- EQ-Bench 3 evaluation methodology and results
- Side-by-side conversation comparisons
- Safety sheet with red-team analysis
- Reproducibility guide with hyperparameters

---

## License

MIT
