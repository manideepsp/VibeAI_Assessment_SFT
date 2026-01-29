# VibeAI Evaluation Report — run_20260129_d

## Artifacts

- Eval exports: `artifacts/eval/run_20260129_d`
- Summary stats: `artifacts/reports/run_20260129_d_stats.json`
- Adapter used: `artifacts/checkpoints/final/adapter`
- Config used: `config\example_config.json`

## Quantitative Summary (Lightweight)

This project exports EQ-Bench-style prompts/responses for external scoring.

The numbers below are *simple descriptive stats* (length + heuristic refusal), not official EQ-Bench scores.

| Set | n (base) | avg_words (base) | n (sft) | avg_words (sft) |
|---|---:|---:|---:|---:|
| eqbench | 10 | 10.0 | 10 | 8.0 |
| qual | 3 | 121.0 | 3 | 9.333 |
| redteam | 3 | 151.0 | 3 | 9.667 |

### Red-team refusal/deflection heuristic

- base: `0.333`
- sft: `0.0`

## Plots

- Training loss curves: `artifacts/plots/loss_curves.png`

![](artifacts/plots/loss_curves.png)

- EQ-Bench response length distribution: `artifacts/plots/run_20260129_d_eqbench_len_words.png`

![](artifacts/plots/run_20260129_d_eqbench_len_words.png)

- Average response length by set: `artifacts/plots/run_20260129_d_avg_len_words.png`

![](artifacts/plots/run_20260129_d_avg_len_words.png)

- Red-team refusal/deflection heuristic: `artifacts/plots/run_20260129_d_redteam_refusal_rate.png`

![](artifacts/plots/run_20260129_d_redteam_refusal_rate.png)

## Reproduce

From repo root:

```bash
python scripts/run_pipeline.py --config config\example_config.json --run_id run_20260129_d
python scripts/run_pipeline.py --config config\example_config.json --run_id run_20260129_d --adapter_dir artifacts/checkpoints/final/adapter
```
