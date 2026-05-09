# Image-Borne Prompt Injection on Gemini 2.5 Flash-Lite

## Goal

A small empirical study testing whether Gemini 2.5 Flash-Lite can be steered into following commands embedded as text overlays on images, instead of following the user's actual instruction. The benchmark is 15 stock photos with progressively harder injection styles (5 clean baselines, 5 soft, 5 hard). No model training or fine-tuning was used.

## Findings

The most syntactically blunt attacks succeeded; the elaborately-framed hard cases were all defended. Attack-success rate on the 10 injection attempts was 3/10 = 30%, with all 3 successes coming from the soft group. A no-training mitigation via a hardened system prompt fully neutralized all three baseline failures (3/3 → 0/3) with no regression on the two controls.

## Scripts

- `build_dataset.py`: composites the 15 stimulus images by overlaying injection text onto royalty-free stock photos.
- `test_gemini.py`: minimal Gemini API connectivity check.
- `run_inference.py`: runs the 15-example baseline sweep with the generic system prompt.
- `run_mitigation.py`: re-runs ids 06, 09, 10, 13, 14 with the hardened system prompt.
- `make_appendix_csvs.py`: generates the appendix-ready CSV and `.tex` files used by my report.

See `report.pdf` for full methodology, results, and analysis.
