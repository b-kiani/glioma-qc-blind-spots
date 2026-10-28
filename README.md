# Two blind spots of foreground uncertainty for review prioritisation in post-treatment glioma segmentation

Code and per-timepoint results for the ISBI 2027 paper by Behnam Kiani Kalejahi, Sajid Khan and Adnan Manzoor
(School of Engineering, Central Asian University, Tashkent, Uzbekistan).

The paper evaluates foreground-restricted uncertainty (8 test-time augmentations + 4 Monte-Carlo DropPath passes)
as a quality-control score that orders automatic segmentations for expert review, on 271 held-out BraTS 2024
post-treatment timepoints. It identifies two blind spots: empty predictions, which receive near-zero uncertainty,
and sub-region failures under an intact whole tumour.

## Contents

| Folder | What it holds |
|---|---|
| `src/` | Data loading, the SwinUNETR-V2 baseline and TopoGraph (GATv2 graph block with a residual gate), TTA + MC-DropPath uncertainty (`src/uncertainty.py`), BraTS sub-region Dice |
| `scripts/` | Training (`06`, `07`), held-out evaluation (`08`), uncertainty (`12`), region-specific uncertainty (`region_uncertainty_eval.py`) |
| `configs/` | The two training configurations used in the paper |
| `splits/` | Patient-level split of the development set (fold 0 was used) |
| `analysis/` | Scripts that compute every number and Fig. 1 of the paper from the per-timepoint table |
| `results/` | Per-timepoint Dice and uncertainty for the 271 evaluation timepoints, both networks |

## Data

The BraTS 2024 adult glioma post-treatment training data are **not** included. Obtain them from the BraTS 2024
organisers, accept their data-use terms, and set `paths.train_dir` and `paths.val_dir` in the two configs.
In the paper, `train_dir` held the 1,350 development timepoints and `val_dir` the 271 held-out evaluation timepoints.

## Environment

PyTorch 2.5.1 and MONAI 1.5.2 (the versions used in the paper); `requirements.txt` lists the exact package versions
of the environment that produced the results.

## Reproducing the paper

```
# 1. Train both networks (200 epochs each, seed 42, fold 0)
python scripts/07_train_topograph.py --config configs/baseline_swin_full.yaml
python scripts/07_train_topograph.py --config configs/topograph_full.yaml

# 2. Per-timepoint Dice on the 271 held-out timepoints
python scripts/08_evaluate_test_set.py --config configs/baseline_swin_full.yaml \
    --checkpoint outputs/baseline_swin_full_fold0/best_model.pt --output outputs/test_eval_baseline_fold0.csv

# 3. Twelve-pass uncertainty (repeat steps 2-3 with the TopoGraph config and checkpoint)
python scripts/12_uncertainty_eval.py --config configs/baseline_swin_full.yaml \
    --checkpoint outputs/baseline_swin_full_fold0/best_model.pt \
    --dice-csv outputs/test_eval_baseline_fold0.csv --output outputs/uncertainty_baseline_fold0.csv --mc-passes 4

# 4. Region-specific (structure-wise) uncertainty, evaluation cohort and fold-0 validation cohort
python scripts/region_uncertainty_eval.py --config configs/baseline_swin_full.yaml \
    --checkpoint outputs/baseline_swin_full_fold0/best_model.pt \
    --subjects outputs/test_eval_baseline_fold0.csv --output outputs/region_unc_baseline_test.csv
python scripts/region_uncertainty_eval.py --config configs/baseline_swin_full.yaml \
    --checkpoint outputs/baseline_swin_full_fold0/best_model.pt \
    --fold-val --output outputs/region_unc_baseline_val.csv

# 5. Every number and Fig. 1 of the paper, from the per-timepoint table (no GPU needed)
python analysis/final_numbers.py results/per_timepoint_evaluation.csv numbers.json 10000
python analysis/figure.py results/per_timepoint_evaluation.csv fig1.png
```

`results/per_timepoint_evaluation.csv` has one row per evaluation timepoint, with anonymised patient and timepoint
identifiers, the per-region Dice of both networks (`_base`, `_topo`) and their uncertainty features.

## Citation

See `CITATION.cff`. Please also cite the BraTS 2024 post-treatment glioma challenge when using the data or results.

## Licence

The code is released under the MIT licence (`LICENSE`). The BraTS data, and results derived from them, remain subject
to the BraTS data-use terms.
