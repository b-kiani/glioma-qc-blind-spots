"""Region-specific (structure-wise) uncertainty for post-treatment glioma QC.

ISBI 2027 revision experiment. Put this file in  D:\\Graph-Based\\topograph_seg\\scripts\\
and run it from  D:\\Graph-Based\\topograph_seg  .

It reuses the project's own, already-validated pipeline and changes nothing in it:
  * model building and EMA weight loading .... scripts/12_uncertainty_eval.py
  * the 12 passes (8 TTA + 4 MC-DropPath) .... src.uncertainty.predictive_distribution
  * the published uncertainty features ....... src.uncertainty.subject_uncertainty_summary
  * the published Dice ....................... src.brats_metrics.brats_subregion_dice
  * the WT / TC / ET label sets .............. src.data_brats

What it adds, per timepoint, from the same 12 softmax volumes:
  * Dice of the deterministic pass (pass 0 = identity TTA, no DropPath), which
    should reproduce scripts/08_evaluate_test_set.py; checked with
    --compare-dice-csv.
  * Structure-wise pass agreement (after Roy et al., NeuroImage 2019): for each
    region r in {WT, TC, ET}, agreement A_r = mean over passes of the Dice between
    that pass's hard mask and the consensus (mean-probability argmax) mask, with
    "both empty" counted as full agreement. Disagreement d_r = 1 - A_r.
  * Region volume coefficient of variation across passes.
  * Variance of the region probability across passes, averaged within the
    predicted region and within the predicted whole tumour.

PRE-SPECIFIED ANALYSIS PLAN (fixed on 28 Oct 2026, before any result existed)
  Primary score   D = mean(d_WT, d_TC, d_ET).
  Primary test    On the 271-timepoint evaluation cohort, failure (mean Dice < 0.5)
                  AUROC and failures captured among the top 54, comparing
                  E_D (empty predictions first, then D) with E_U (empty first,
                  then U = mean_fgvar_fg), both computed from this same run.
  Key subgroup    Sub-region failures: mean Dice < 0.5 with WT Dice >= 0.5,
                  split at WT Dice 0.8.
  Secondary       Fold-0 validation cohort: D versus U for its failures.
  Failure labels  From the published per-timepoint Dice (the paper's failure set);
                  agreement with this run's own deterministic-pass Dice is reported.
  Statistics      Paired patient-clustered bootstrap, 10,000 resamples.
  Everything else (D_max, per-region scores, rank mean of U and D) is exploratory.

Usage (PowerShell, from D:\\Graph-Based\\topograph_seg):
  python scripts/region_uncertainty_eval.py `
      --config configs/baseline_swin_full.yaml `
      --checkpoint outputs/baseline_swin_full_fold0/best_model.pt `
      --output outputs/region_unc_baseline_test.csv `
      --compare-dice-csv outputs/test_eval_baseline_fold0.csv `
      --compare-unc-csv outputs/uncertainty_baseline_fold0.csv --resume

Cohort selection (default = the same folder script 12 uses, cfg paths.val_dir):
  --fold-val         the fold's validation timepoints, selected exactly as in
                     training (cfg paths.train_dir + paths.split_file + data.fold,
                     via src.data_brats.load_split / patients_in_fold).
  --subjects FILE    .txt (one id per line) or .csv/.json with subject ids; keeps
                     only those subjects.
  --split-json FILE --split-key val    keep the ids listed under that key.
  --data-dir DIR     scan this folder instead of cfg paths.val_dir.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

try:
    from src.brats_metrics import brats_subregion_dice
    from src.data_brats import ET_LABELS, TC_LABELS, WT_LABELS, scan_subjects
    from src.dataset import build_dataset, subjects_to_records
    from src.transforms import IMAGE_KEY, LABEL_KEY, val_transforms
    from src.uncertainty import predictive_distribution, subject_uncertainty_summary
    from src.utils import resolve_device, set_seed, setup_logging
except ImportError as e:  # pragma: no cover - environment problem, explain it
    raise SystemExit(
        f"Could not import the project modules ({e}).\n"
        "Run this from D:\\Graph-Based\\topograph_seg with the script placed in scripts\\."
    )

REGIONS = {"WT": tuple(WT_LABELS), "TC": tuple(TC_LABELS), "ET": tuple(ET_LABELS)}
SUBJECT_RE = re.compile(r"BraTS-GLI-(\d+)-(\d+)", re.IGNORECASE)


def _load_script12():
    path = ROOT / "scripts" / "12_uncertainty_eval.py"
    if not path.exists():
        raise SystemExit(f"Expected {path} (the script that produced the published uncertainty).")
    spec = importlib.util.spec_from_file_location("unc12", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    for name in ("load_config", "build_model_from_config", "load_weights", "resolve_path"):
        if not hasattr(mod, name):
            raise SystemExit(f"scripts/12_uncertainty_eval.py has no function '{name}'.")
    return mod


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Region-specific uncertainty (ISBI 2027 revision).")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--mc-passes", type=int, default=4)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-ema", action="store_true")
    p.add_argument("--data-dir", type=Path, default=None)
    p.add_argument("--fold-val", action="store_true",
                   help="Use the fold's validation timepoints (as selected during training).")
    p.add_argument("--subjects", type=Path, default=None)
    p.add_argument("--split-json", type=Path, default=None)
    p.add_argument("--split-key", default="val")
    p.add_argument("--compare-dice-csv", type=Path, default=None)
    p.add_argument("--compare-unc-csv", type=Path, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--allow-missing", action="store_true",
                   help="Continue when some requested subjects are not in the data folder.")
    return p.parse_args()


ID_COLUMNS = ("subject_id", "subject", "case_id", "case", "id", "folder_name", "name")
FULL_ID_RE = re.compile(r"BraTS-GLI-\d+-\d+", re.IGNORECASE)


def id_column(fieldnames) -> str | None:
    low = {c.strip().lower(): c for c in (fieldnames or [])}
    for k in ID_COLUMNS:
        if k in low:
            return low[k]
    return None


def read_ids(path: Path) -> list[str]:
    """Subject ids from .txt / .csv / .json; falls back to any BraTS-GLI-xxxxx-xxx in the file."""
    text = path.read_text(encoding="utf-8-sig")
    ids: list[str] = []
    if path.suffix.lower() == ".json":
        obj = json.loads(text)
        if isinstance(obj, dict):
            for k in ("subjects", "subject_id", "ids"):
                if k in obj:
                    obj = obj[k]
                    break
        if isinstance(obj, list):
            ids = [str(x.get("subject_id", x.get("id"))) if isinstance(x, dict) else str(x) for x in obj]
    elif path.suffix.lower() == ".csv":
        rd = csv.DictReader(text.splitlines())
        col = id_column(rd.fieldnames)
        if col is not None:
            ids = [r[col].strip() for r in rd if r.get(col)]
    else:
        ids = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not any(FULL_ID_RE.fullmatch(i) for i in ids):      # last resort: scan the text
        ids = sorted(set(m.group(0) for m in FULL_ID_RE.finditer(text)))
    if not ids:
        raise SystemExit(f"No subject ids found in {path}.")
    return ids


def read_split(path: Path, key: str) -> list[str]:
    obj = json.loads(path.read_text(encoding="utf-8-sig"))
    if key not in obj:
        raise SystemExit(f"{path} has no key '{key}'. Keys: {list(obj)}")
    return [str(x) if not isinstance(x, dict) else str(x.get("subject_id", x.get("id")))
            for x in obj[key]]


def patient_of(sid: str) -> str:
    m = SUBJECT_RE.search(sid)
    return f"BraTS-GLI-{m.group(1)}" if m else sid.rsplit("-", 1)[0]


def region_mask(lab: torch.Tensor, labels: tuple) -> torch.Tensor:
    out = torch.zeros_like(lab, dtype=torch.bool)
    for v in labels:
        out |= lab == v
    return out


@torch.no_grad()
def region_features(prob_stack: torch.Tensor) -> dict:
    """Structure-wise uncertainty from the (P, C, D, H, W) probability stack.

    Memory-lean: one pass at a time for the hard masks, and float32 accumulation
    without copying the whole stack (the stack is fp16 when AMP is on).
    """
    P = prob_stack.shape[0]
    cons = prob_stack.mean(dim=0, dtype=torch.float32).argmax(dim=0)   # consensus labels
    cmask = {r: region_mask(cons, labels) for r, labels in REGIONS.items()}
    ncons = {r: int(m.sum()) for r, m in cmask.items()}
    agree = {r: [] for r in REGIONS}
    vols = {r: [] for r in REGIONS}
    for p in range(P):
        hp = prob_stack[p].argmax(dim=0)
        for r, labels in REGIONS.items():
            m = region_mask(hp, labels)
            a = int(m.sum())
            vols[r].append(a)
            nc = ncons[r]
            agree[r].append(1.0 if a + nc == 0 else 2.0 * int((m & cmask[r]).sum()) / (a + nc))
        del hp
    f = {}
    wt_cons = cmask["WT"]
    for r, labels in REGIONS.items():
        f[f"cons_vox_{r}"] = ncons[r]
        f[f"dis_{r}"] = 1.0 - float(np.mean(agree[r]))
        mv = float(np.mean(vols[r]))
        f[f"vcv_{r}"] = float(np.std(vols[r]) / mv) if mv > 0 else 0.0
        pr = torch.zeros(prob_stack[:, 0].shape, dtype=torch.float32, device=prob_stack.device)
        for c in labels:
            pr.add_(prob_stack[:, c])
        var_r = pr.var(dim=0, unbiased=False)
        del pr
        f[f"var_{r}_in_{r}"] = float(var_r[cmask[r]].mean()) if ncons[r] > 0 else float("nan")
        f[f"var_{r}_in_WT"] = float(var_r[wt_cons].mean()) if ncons["WT"] > 0 else float("nan")
        del var_r
    f["D_mean"] = float(np.mean([f["dis_WT"], f["dis_TC"], f["dis_ET"]]))
    f["D_max"] = float(np.max([f["dis_WT"], f["dis_TC"], f["dis_ET"]]))
    f["all_pass_empty"] = int(all(v == 0 for v in vols["WT"]))
    return f


def det_dice(prob_stack: torch.Tensor, label: torch.Tensor) -> dict:
    """Dice of the deterministic pass with the project's own metric function."""
    det = prob_stack[0:1].float()                        # (1, C, D, H, W); argmax == logits argmax
    d = brats_subregion_dice(det, label)
    vals = [float(d[r].item()) for r in ("WT", "TC", "ET")]
    mean = float(np.nanmean(vals)) if not all(math.isnan(v) for v in vals) else float("nan")
    return {"dice_WT": vals[0], "dice_TC": vals[1], "dice_ET": vals[2], "dice_mean": mean}


def _num(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _rows_by_id(path: Path) -> dict:
    rd = csv.DictReader(open(path, encoding="utf-8-sig"))
    col = id_column(rd.fieldnames)
    if col is None:
        raise ValueError(f"no subject id column in {path.name}: {rd.fieldnames}")
    return {r[col].strip(): {k.strip().lower(): v for k, v in r.items() if k} for r in rd}


def _mean_dice(r: dict) -> float:
    """Per-timepoint mean Dice as in the paper: nan-mean over WT, TC, ET."""
    v = [_num(r.get(k)) for k in ("dice_wt", "dice_tc", "dice_et")]
    v = [x for x in v if math.isfinite(x)]
    if v:
        return float(np.mean(v))
    return _num(r.get("dice_mean"))


def verify(out_csv: Path, dice_csv: Path | None, unc_csv: Path | None, logger) -> dict:
    """Compare this run with the published per-timepoint files. Never raises."""
    from scipy.stats import spearmanr
    rep: dict = {}
    try:
        new = _rows_by_id(out_csv)
        rep["n_new"] = len(new)
    except Exception as e:  # noqa: BLE001
        rep["error"] = f"could not read {out_csv}: {e}"
        logger.info(f"VERIFY skipped: {rep['error']}")
        return rep
    if dice_csv is not None:
        try:
            old = _rows_by_id(dice_csv)
            ids = sorted(set(new) & set(old))
            a = np.array([_mean_dice(new[i]) for i in ids])
            b = np.array([_mean_dice(old[i]) for i in ids])
            ok = np.isfinite(a) & np.isfinite(b)
            diff = np.abs(a[ok] - b[ok])
            reg = {}
            for k in ("dice_wt", "dice_tc", "dice_et"):
                x = np.array([_num(new[i].get(k)) for i in ids]); y = np.array([_num(old[i].get(k)) for i in ids])
                both = np.isfinite(x) & np.isfinite(y)
                reg[k] = {"max_abs_diff": float(np.abs(x[both] - y[both]).max()) if both.any() else None,
                          "nan_pattern_identical": bool((np.isnan(x) == np.isnan(y)).all())}
            fa, fb = a < 0.5, b < 0.5
            rep["dice"] = {"published_file": dice_csv.name, "n_matched": len(ids),
                           "n_published": len(old),
                           "median_abs_diff": float(np.median(diff)) if diff.size else None,
                           "max_abs_diff": float(diff.max()) if diff.size else None,
                           "n_abs_diff_gt_0.01": int((diff > 0.01).sum()),
                           "failures_new": int(fa.sum()), "failures_published": int(fb.sum()),
                           "failure_set_identical": bool((fa == fb).all()),
                           "mismatched_failures": [i for i, x, y in zip(ids, fa, fb) if x != y][:20],
                           "per_region": reg}
            logger.info(f"VERIFY Dice vs {dice_csv.name}: matched {len(ids)}, max |diff| "
                        f"{rep['dice']['max_abs_diff']}, failures {int(fa.sum())} vs {int(fb.sum())}, "
                        f"identical failure set: {rep['dice']['failure_set_identical']}")
        except Exception as e:  # noqa: BLE001
            rep["dice_error"] = repr(e)
            logger.info(f"VERIFY Dice skipped: {e!r}")
    if unc_csv is not None:
        try:
            old = _rows_by_id(unc_csv)
            ids = sorted(set(new) & set(old))
            rep["uncertainty"] = {"published_file": unc_csv.name, "n_matched": len(ids)}
            for k in ("mean_fgvar_fg", "mean_entropy_fg", "frac_high_entropy"):
                ko = k if (ids and k in old[ids[0]]) else f"{k}_fg"
                if not ids or ko not in old[ids[0]]:
                    continue
                a = np.array([_num(new[i].get(k)) for i in ids]); b = np.array([_num(old[i].get(ko)) for i in ids])
                ok = np.isfinite(a) & np.isfinite(b)
                if ok.sum() > 2:
                    rep["uncertainty"][f"spearman_{k}"] = float(spearmanr(a[ok], b[ok]).statistic)
                    rep["uncertainty"][f"median_rel_diff_{k}"] = float(
                        np.median(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-12)))
            logger.info(f"VERIFY uncertainty vs {unc_csv.name}: {rep['uncertainty']}")
        except Exception as e:  # noqa: BLE001
            rep["uncertainty_error"] = repr(e)
            logger.info(f"VERIFY uncertainty skipped: {e!r}")
    return rep


def subject_from_batch(batch) -> str | None:
    """Subject id from the loaded image's file name, if the loader kept it."""
    try:
        img = batch[IMAGE_KEY]
        meta = getattr(img, "meta", None) or batch.get(f"{IMAGE_KEY}_meta_dict", {})
        fn = meta.get("filename_or_obj")
        while isinstance(fn, (list, tuple)) and fn:
            fn = fn[0]
        m = FULL_ID_RE.search(str(fn)) if fn is not None else None
        return m.group(0) if m else None
    except Exception:  # noqa: BLE001
        return None


def select_fold_val(cfg: dict, s12):
    """The fold's validation timepoints, selected exactly as the training scripts do."""
    try:
        from src.data_brats import filter_subjects_by_patients, load_split, patients_in_fold
    except ImportError as e:
        raise SystemExit(f"--fold-val needs src.data_brats.load_split/patients_in_fold ({e}).")
    paths = cfg["paths"]
    train_dir = s12.resolve_path(Path(paths["train_dir"]))
    split = load_split(s12.resolve_path(Path(paths["split_file"])))
    fold = cfg["data"]["fold"]
    val_pids = patients_in_fold(split, fold, train=False)
    subs = filter_subjects_by_patients(scan_subjects(train_dir, require_seg=True), val_pids)
    return train_dir, fold, len(val_pids), subs


def main() -> int:
    args = parse_args()
    s12 = _load_script12()
    cfg = s12.load_config(args.config)
    args.output = args.output if args.output.is_absolute() else (ROOT / args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(args.output.parent, log_name=f"{args.output.stem}.log")
    logger.info("=" * 72)
    logger.info("Region-specific uncertainty (ISBI 2027 revision)")
    logger.info(f"Config {args.config} | checkpoint {args.checkpoint} | MC passes {args.mc_passes}")
    logger.info(f"Regions: {REGIONS}")

    # Resolve every input path now, so a typo fails in seconds, not after two hours.
    cmp_dice = s12.resolve_path(args.compare_dice_csv) if args.compare_dice_csv else None
    cmp_unc = s12.resolve_path(args.compare_unc_csv) if args.compare_unc_csv else None
    s12.resolve_path(args.checkpoint)

    set_seed(cfg["experiment"]["seed"])
    device = resolve_device(cfg["training"]["device"])
    logger.info(f"Device: {device}")

    # ------------------------------------------------------------- cohort
    if args.fold_val:
        data_dir, fold, n_pids, subjects = select_fold_val(cfg, s12)
        logger.info(f"Fold-{fold} validation cohort from {data_dir}: {n_pids} patients, "
                    f"{len(subjects)} timepoints")
    else:
        data_dir = args.data_dir if args.data_dir is not None else Path(cfg["paths"]["val_dir"])
        data_dir = s12.resolve_path(data_dir)
        subjects = scan_subjects(data_dir, require_seg=True)
    keep = None
    if args.subjects is not None:
        keep = set(read_ids(s12.resolve_path(args.subjects)))
    if args.split_json is not None:
        ids = set(read_split(s12.resolve_path(args.split_json), args.split_key))
        keep = ids if keep is None else keep & ids
    if keep is not None:
        found = {s.folder_name for s in subjects}
        missing = sorted(keep - found)
        if not keep & found:
            raise SystemExit(f"None of the {len(keep)} requested subjects is in {data_dir} "
                             f"(e.g. {sorted(keep)[:2]}). Use --data-dir to point at the right folder.")
        if missing:
            msg = (f"{len(missing)} of {len(keep)} requested subjects are not in {data_dir}, "
                   f"e.g. {missing[:3]}")
            if not args.allow_missing:
                raise SystemExit(msg + ". Point --data-dir (TEST_DATA_DIR in the .ps1) at the folder "
                                       "that holds all of them, or pass --allow-missing.")
            logger.info("WARNING " + msg)
        subjects = [s for s in subjects if s.folder_name in keep]
    n_cohort = len(subjects)

    # ------------------------------------------------------------- resume
    done: set[str] = set()
    if args.resume and args.output.exists():
        with open(args.output, encoding="utf-8") as fh_in:
            rd = csv.DictReader(fh_in)
            fields = rd.fieldnames or []
            rows = [r for r in rd if r.get("subject_id") and r.get(fields[-1]) not in (None, "")]
        if fields:                                           # rewrite without any half-written line
            with open(args.output, "w", newline="", encoding="utf-8") as fh_out:
                w = csv.DictWriter(fh_out, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
        done = {r["subject_id"] for r in rows}
        subjects = [s for s in subjects if s.folder_name not in done]
        logger.info(f"Resume: {len(done)} already done")
    if args.limit is not None:
        subjects = subjects[: args.limit]
    logger.info(f"Data dir {data_dir} | cohort {n_cohort} | to run now: {len(subjects)}")

    if not subjects:
        logger.info("Nothing to run.")
    else:
        records = subjects_to_records(subjects)
        ds = build_dataset(records, val_transforms(seed=cfg["experiment"]["seed"]), backend="cache", cache_dir=None)
        loader = torch.utils.data.DataLoader(ds, batch_size=1, num_workers=0, shuffle=False)

        model = s12.build_model_from_config(cfg).to(device)
        s12.load_weights(model, s12.resolve_path(args.checkpoint), use_ema=not args.no_ema,
                         device=device, logger=logger)
        model.eval()
        patch = tuple(cfg["data"]["patch_size"])
        sw_bs = cfg["training"]["val_sw_batch_size"]
        sw_ov = cfg["training"]["val_sw_overlap"]
        use_amp = bool(cfg["training"].get("amp", False)) and torch.cuda.is_available()
        logger.info(f"patch {patch} | sw batch {sw_bs} | overlap {sw_ov} | AMP {use_amp}")
        expected_passes = 8 + args.mc_passes

        new_file = not done
        fh = open(args.output, "w" if new_file else "a", newline="", encoding="utf-8")
        writer = None
        t_all = time.time()
        for i, batch in enumerate(loader, start=1):
            sid = subjects[i - 1].folder_name
            from_file = subject_from_batch(batch)
            if (from_file is not None and FULL_ID_RE.fullmatch(sid)
                    and from_file.lower() != sid.lower()):
                fh.close()
                raise SystemExit(f"Loader order mismatch: expected {sid}, image file says {from_file}.")
            t0 = time.time()
            img = batch[IMAGE_KEY].to(device, non_blocking=True)
            lab = batch[LABEL_KEY].to(device).long()
            if i == 1:
                vals = sorted(int(v) for v in torch.unique(lab).cpu())
                logger.info(f"First label values: {vals} | image {tuple(img.shape)} | id check: "
                            f"{'file name ' + str(from_file) if from_file else 'not available'}")
                if not set(vals) <= {0, 1, 2, 3, 4}:
                    logger.info("WARNING labels outside 0-4; region definitions may not apply.")
            stack = predictive_distribution(model, img, patch, sw_batch_size=sw_bs, overlap=sw_ov,
                                            use_amp=use_amp, tta=True, mc_droppath_passes=args.mc_passes)
            if i == 1:
                logger.info(f"Probability stack {tuple(stack.shape)} {stack.dtype}")
                if stack.shape[0] != expected_passes:
                    logger.info(f"WARNING {stack.shape[0]} passes, expected {expected_passes}.")
            orig = subject_uncertainty_summary(stack)
            row = {"subject_id": sid, "patient_id": patient_of(sid), "n_passes": int(stack.shape[0])}
            row.update(det_dice(stack, lab))
            labv = lab.squeeze()
            det = stack[0].argmax(dim=0)
            for r, labels in REGIONS.items():
                row[f"gt_vox_{r}"] = int(region_mask(labv, labels).sum())
                row[f"det_vox_{r}"] = int(region_mask(det, labels).sum())
            del det
            row.update({k: orig[k] for k in orig})
            try:
                rf = region_features(stack)
            except RuntimeError as e:                    # CUDA out of memory -> same computation on CPU
                if "out of memory" not in str(e).lower():
                    raise
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"  {sid}: GPU memory full for region features, computing them on CPU")
                rf = region_features(stack.cpu())
            row.update(rf)
            row["duration_s"] = round(time.time() - t0, 2)
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if new_file:
                    writer.writeheader()
            writer.writerow(row)
            fh.flush()
            del stack, img, lab, labv
            if torch.cuda.is_available():
                if i == 1:
                    logger.info(f"Peak GPU memory after first timepoint: "
                                f"{torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
                torch.cuda.empty_cache()
            if i % 10 == 0 or i == len(subjects) or i <= 2:
                el = time.time() - t_all
                logger.info(f"  [{i:4d}/{len(subjects)}] {sid} dice={row['dice_mean']:.3f} "
                            f"U={row['mean_fgvar_fg']:.4f} D={row['D_mean']:.4f} "
                            f"({row['duration_s']:.1f}s; ~{el / i * (len(subjects) - i) / 60:.0f} min left)")
        fh.close()

    rep = verify(args.output, cmp_dice, cmp_unc, logger)
    rep["cohort_size"] = n_cohort
    rep["fold_val"] = bool(args.fold_val)
    rep["analysis_plan"] = "pre-specified 2026-10-28: primary D_mean vs mean_fgvar_fg, both with empty-first"
    Path(str(args.output) + ".verify.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    logger.info(f"Wrote {args.output} and {args.output}.verify.json")
    return 0


def _save_error(text: str) -> None:
    err = ROOT / "outputs" / "region_last_error.txt"
    try:
        err.parent.mkdir(parents=True, exist_ok=True)
        err.write_text(" ".join(sys.argv) + "\n\n" + text + "\n", encoding="utf-8")
        print(f"\nThe error was saved to {err}", file=sys.stderr)
    except OSError:
        pass


if __name__ == "__main__":
    import traceback
    stale = ROOT / "outputs" / "region_last_error.txt"
    if stale.exists():
        stale.unlink()
    try:
        code = main()
    except SystemExit as e:                  # our own clear messages
        if e.code not in (0, None):
            _save_error(str(e.code))
        raise
    except Exception:                        # anything unexpected: keep the full traceback
        _save_error(traceback.format_exc())
        raise
    raise SystemExit(code)
