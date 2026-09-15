## Active Defence

SignalScope includes an active-defence analysis module that evaluates the
existing detector against common post-processing attacks without retraining
or changing the production threshold.

Tested attacks include:

- JPEG recompression
- aggressive down/up resizing
- blur
- brightness changes
- contrast changes
- additive noise

The analysis reports prediction-flip rate and AI-probability change per attack.

Run:

```powershell
python src\defense\active_defense.py `
  --checkpoint "models\checkpoints\best_model.pt" `
  --real-dir "TEMP_TEST\REAL" `
  --fake-dir "TEMP_TEST\FAKE" `
  --limit 100 `
  --output-dir "results\active_defense"
```

Outputs:

- `results/active_defense/attack_results.csv`
- `results/active_defense/attack_summary.json`
- `results/active_defense/active_defense_report.md`

The attack set is diagnostic only and is not used to tune the classifier or
production threshold.
