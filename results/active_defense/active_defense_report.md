# SignalScope Active Defence Analysis

Production model weights and threshold were not changed.

- Temperature: `0.95`
- Threshold: `0.56`
- Samples per class: controlled by `--limit`

## Attack Summary

| Attack | Samples | Flip rate | Mean |Δ AI probability| | REAL flip | AI flip |
|---|---:|---:|---:|---:|---:|
| resize_8_roundtrip | 200 | 34.00% | 0.3568 | 21.00% | 47.00% |
| resize_16_roundtrip | 200 | 16.50% | 0.1965 | 9.00% | 24.00% |
| gaussian_blur | 200 | 6.00% | 0.0935 | 5.00% | 7.00% |
| jpeg_50 | 200 | 6.00% | 0.0556 | 6.00% | 6.00% |
| contrast_high | 200 | 5.50% | 0.0452 | 9.00% | 2.00% |
| contrast_low | 200 | 5.00% | 0.0641 | 2.00% | 8.00% |
| gaussian_noise | 200 | 4.00% | 0.0466 | 2.00% | 6.00% |
| brightness_low | 200 | 4.00% | 0.0441 | 5.00% | 3.00% |
| jpeg_70 | 200 | 2.00% | 0.0153 | 3.00% | 1.00% |
| brightness_high | 200 | 1.50% | 0.0314 | 0.00% | 3.00% |

## Interpretation

A prediction flip means the attack changed the model's binary decision at the fixed production threshold. Higher flip rates indicate greater sensitivity to that degradation.

This module is an analysis tool, not a new training stage. The attack set is not used to retune the threshold or model weights.

## Defensive Guidance

When a high flip rate is observed, the system should surface a robustness warning and recommend human review rather than presenting the prediction as definitive.
