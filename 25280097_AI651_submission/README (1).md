# AI651 Assignment 1 — Namra Nadeem (25280097)

**Public repository:** https://github.com/namranadeem-wq/DL4STG-assignment-1-new/tree/main/25280097_AI651_submission

This package contains executed forecasting evidence for both tasks, a draft combined report, its LaTeX source, numbered PDF figures, supplied Task 1 harness files, and Task 2 model checkpoints/results.

## Status

Task 2 training, the three-seed external-data ablations, and final 168-step forecast are complete. Final P = 21,953; declared model-lineage E = 17 (10 validation epochs + 7 refit epochs). All nine validation runs total 76 epochs; with the final refit, project-wide training totals 83 epochs. There is no ensemble. Leaderboard submission is verified from the supplied screenshot: MAE 68.5591, RMSE 99.4716, sMAPE 81.04%, score 99.4801, rank 32 at capture, 1/5 submissions used.

The original Task 1 notebook contains full-preset outputs and passing checks, but includes a mid-run study reset and mixed CPU/CUDA training sessions. Pre-run predictions were not found; do not invent them. A separate clean-rerun notebook has been prepared, but it is UNEXECUTED and does not replace the executed evidence. Report numbers currently reflect the original executed notebook. If rerun numbers change, update the report and figures consistently.

## Files

- `25280097_Task1_executed.ipynb`: original executed full-preset evidence.
- `25280097_Task1_clean_rerun.ipynb`: derived unexecuted notebook with administrative cells and mid-run reset removed.
- `25280097_Task2_Autoformer.ipynb`: original executed Task 2 notebook.
- `task2_run.py`: command-line reconstruction of Task 2, using local CSV files and the pinned Autoformer reference commit.
- `harness/`: supplied Task 1 modules; keep beside the notebook.
- `data/`: the three assignment-provided CSVs. Do not share data beyond the course's permissions.
- `task2_outputs/`: validation logs, predictions, checkpoints, metadata and original reference source/license.
- `figures/`: report supporting figures recovered from notebook outputs.
- `25280097_AI651_report_draft.pdf` and `.tex`: combined draft; highlighted completion items are unresolved.
- `AI_ASSISTANCE.md`: visible AI prompts, generated-code inventory, and limitations of the recorded disclosure.

## Reproduce Task 2

Use Python 3.11 or later, Git and a compatible PyTorch installation. The recorded Colab environment was Python 3.13.15, PyTorch 2.11.0+cu130, Tesla T4. Install the appropriate PyTorch build for your machine; CPU execution is supported but numerical results may differ.

```bash
python -m pip install -r requirements.txt
python task2_run.py
```

Run from this package directory. The script loads `data/`, checks out reference commit `51c7d416ae120b805fd5beef2f4ccf7de496a6ff`, repeats the nine validation experiments and the final refit, and writes `task2_outputs/` and a results ZIP. It requires internet for the first reference clone. Back up existing outputs before rerunning because matching filenames are overwritten. Its Python syntax and protocol were checked; training was not rerun in the packaging environment, which has no PyTorch installed. The original executed notebook is authoritative for the reported run.

For Colab, open the original Task 2 notebook and run top to bottom in a fresh GPU runtime, uploading the three CSVs. Its original reference-clone cell records HEAD rather than pinning it; use the commit above for reproducibility, as the command-line script does.

## Reproduce Task 1

Open the clean-rerun notebook from this directory with `harness/` beside it. Record new, genuine expectations before running the indicated comparison cells. Restart the kernel and run all with `PA1_PRESET=full`. This is a new experiment, not a recovery of missing earlier predictions. Retain executed outputs and generated `results/design/` PDFs/CSVs; update the report to match the single run if used for submission.

## Report and submission

Compile from this directory:

```bash
pdflatex -interaction=nonstopmode -halt-on-error 25280097_AI651_report_draft.tex
```

Finish the report by resolving the documented Task 1 prediction issue honestly, confirming personal edits/AI disclosure, and retaining the recorded leaderboard receipt. Submit the combined PDF, LaTeX source, executed full-preset Task 1 notebook, and every numbered PDF figure used by the report. The task-specific code belongs in the public repository; keep the executed Task 2 notebook there for reproducibility.

Paste `task2_outputs/leaderboard_predictions.txt` into https://ai-651-leaderboard.vercel.app/ with P=21953 and E=17. It contains exactly 168 chronological numeric values. Do not paste CSV headers, indices or brackets. The leaderboard is not a tuning set and permits at most five submissions.

## Attribution

Task 1 uses the supplied AI651 assignment harness. Task 2 adapts Wu et al. (2021), Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting, and https://github.com/thuml/Autoformer at the commit above. Original reference source and license are preserved under `task2_outputs/reference_source/`.

## Repository verification

The public repository was checked on 5 October 2026. The package subdirectory preserves harness/, figures/, data/, task2_outputs/historical_only/ and task2_outputs/reference_source/. Run commands from 25280097_AI651_submission/, not the repository parent. File contents match the prepared package before this report/README link update. Training was not rerun during repository inspection. Remaining report gaps are the authentic Task 1 pre-run expectations and confirmation of the personal AI-edit disclosure.
