# SignalScope

Telling Real From Synthetic in the Age of Generative Media — SIH 2026 Internal Hackathon, LJIET.

## Status
🚧 In progress. Model training not yet complete — the app currently runs
against a placeholder prediction function (see `model/MODEL_TODO.md`).
See `TEAM_TASKS.md` for what can be worked on in parallel right now.

## Modules built
- [x] Core task (in progress)
- [ ] Bonus A — Faithful Explanation
- [ ] Bonus B — Generator Attribution
- [ ] Bonus C — Robustness to Degradation
- [ ] Other bonuses: TBD

## Setup & run instructions

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Upload an image in the browser tab that opens — you should get a
real/fake result with a confidence score in under 10 minutes from a
clean setup.

## Datasets used
- TBD (list source + license once finalized)

## Reported metrics
- TBD (fill in once training + evaluation are complete — see `report/model_report_template.md`)

## Architecture overview
- Pixel-view branch: CNN backbone (transfer learning)
- Frequency-view branch: FFT/DCT-based features
- Fusion layer combining both, calibrated confidence output

## Known limitations
- TBD

## Demo video
- TBD (link once recorded)
