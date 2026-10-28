"""Analyse region_uncertainty_eval.py outputs (pre-specified plan, 28 Oct 2026).

Primary   E_D vs E_U on the evaluation cohort: AUROC and failures captured in the
          top 54, paired patient-clustered bootstrap (10,000 resamples).
          E_x = empty predictions first, then score x.  U = mean_fgvar_fg.
          D = D_mean (structure-wise pass disagreement).
Subgroup  sub-region failures (WT Dice >= 0.5), split at WT 0.8.
Secondary validation cohort, D vs U.
Exploratory: D_max, per-region scores, rank-mean of U and D.

usage: analyse_region.py out.json name=path.csv[::published_dice.csv] [...] [--boot=N]
       names ending in _test are evaluation cohorts; _val are validation.
       With ::published_dice.csv the failure labels use the published per-timepoint
       Dice (nan-mean of WT, TC, ET), so the failure set is exactly the paper's; the
       agreement with this run's own Dice is reported.
"""
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score

args = [a for a in sys.argv[1:] if not a.startswith("--")]
B = 10000
for a in sys.argv[1:]:
    if a.startswith("--boot="):
        B = int(a.split("=")[1])
OUT = args[0]
FILES = dict(a.split("=", 1) for a in args[1:])


def order(s):
    return np.lexsort((np.arange(len(s)), -s))


def cap(s, y, n):
    return int(y[order(s)[:n]].sum())


def auc(y, s):
    s = np.asarray(s, float)
    fin = np.isfinite(s)
    top = (np.nanmax(s[fin]) if fin.any() else 0.0) + 1.0
    s = np.where(np.isposinf(s), top, s)
    s = np.where(np.isnan(s), -1e9, s)
    return roc_auc_score(y, s) if 0 < y.sum() < len(y) else float("nan")


def scores(d):
    empty = d["pred_fg_voxels"].values == 0
    U = d["mean_fgvar_fg"].values.astype(float)
    D = d["D_mean"].values.astype(float)
    out = {
        "U": U, "D": D,
        "E_U": np.where(empty, np.inf, U), "E_D": np.where(empty, np.inf, D),
        "E_Dmax": np.where(empty, np.inf, d["D_max"].values.astype(float)),
        "E_disET": np.where(empty, np.inf, d["dis_ET"].values.astype(float)),
        "E_disTC": np.where(empty, np.inf, d["dis_TC"].values.astype(float)),
        "E_rankUD": np.where(empty, np.inf, (rankdata(U) + rankdata(D)) / 2.0),
        "V": -np.log1p(d["pred_fg_voxels"].values.astype(float)),
    }
    return out, empty


def analyse(d, n_review, boot):
    d = d.reset_index(drop=True)
    y = (d["dice_mean"].values < 0.5).astype(int)
    wt = d["dice_WT"].values
    S, empty = scores(d)
    pid = d["patient_id"].values
    pats = np.unique(pid)
    idx = {p: np.flatnonzero(pid == p) for p in pats}
    typ = np.full(len(d), "none", dtype=object)
    typ[(y == 1) & empty] = "empty"
    typ[(y == 1) & ~empty & (np.nan_to_num(wt, nan=0) < 0.5)] = "wt_miss"
    typ[(y == 1) & ~empty & (wt >= 0.5) & (wt < 0.8)] = "sub_0.5_0.8"
    typ[(y == 1) & ~empty & (wt >= 0.8)] = "sub_ge_0.8"
    R = {"n": len(d), "patients": len(pats), "failures": int(y.sum()),
         "failure_patients": int(len(set(pid[y == 1]))), "n_empty": int(empty.sum()),
         "empty_failures": int(y[empty].sum()), "mean_dice": float(d["dice_mean"].mean()),
         "n_review": n_review, "types": {t: int((typ == t).sum()) for t in
                                         ("empty", "wt_miss", "sub_0.5_0.8", "sub_ge_0.8")},
         "rho_U_dice": float(spearmanr(S["U"], d["dice_mean"]).statistic),
         "rho_D_dice": float(spearmanr(S["D"], d["dice_mean"]).statistic),
         "rho_U_D": float(spearmanr(S["U"], S["D"]).statistic)}
    for k, s in S.items():
        R[k] = {"auroc": float(auc(y, s)), "cap": cap(s, y, n_review),
                "by_type": {t: int(((typ == t)[order(s)[:n_review]]).sum())
                            for t in ("empty", "wt_miss", "sub_0.5_0.8", "sub_ge_0.8")}}
    sub = np.isin(typ, ["sub_0.5_0.8", "sub_ge_0.8"])
    msub = sub | (y == 0)
    for k in ("U", "D", "V"):
        R[k]["auroc_subregion"] = float(auc(sub[msub].astype(int), S[k][msub])) if sub.sum() else None
    ne = ~empty
    for k in ("U", "D"):
        R[k]["auroc_nonempty"] = float(auc(y[ne], S[k][ne]))
    # paired bootstrap
    rng = np.random.default_rng(20261028)
    keys = ["E_D", "E_U", "D", "U"]
    bt = {k: [] for k in ("dAUC_ED_EU", "dCAP_ED_EU_pp", "dAUCsub_D_U", "dAUCne_D_U",
                          "AUC_E_D", "AUC_E_U")}
    for _ in range(boot):
        ii = np.concatenate([idx[p] for p in rng.choice(pats, len(pats))])
        yb = y[ii]
        if not (0 < yb.sum() < len(ii)):
            continue
        a_ed, a_eu = auc(yb, S["E_D"][ii]), auc(yb, S["E_U"][ii])
        bt["AUC_E_D"].append(a_ed); bt["AUC_E_U"].append(a_eu); bt["dAUC_ED_EU"].append(a_ed - a_eu)
        bt["dCAP_ED_EU_pp"].append((cap(S["E_D"][ii], yb, n_review) - cap(S["E_U"][ii], yb, n_review))
                                   / yb.sum() * 100)
        sb, mb = sub[ii], msub[ii]
        if 0 < sb[mb].sum() < mb.sum():
            bt["dAUCsub_D_U"].append(auc(sb[mb].astype(int), S["D"][ii][mb])
                                     - auc(sb[mb].astype(int), S["U"][ii][mb]))
        nb = ne[ii]
        if 0 < yb[nb].sum() < nb.sum():
            bt["dAUCne_D_U"].append(auc(yb[nb], S["D"][ii][nb]) - auc(yb[nb], S["U"][ii][nb]))
    R["ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] if len(v) > 50 else None
               for k, v in bt.items()}
    R["point"] = {"dAUC_ED_EU": R["E_D"]["auroc"] - R["E_U"]["auroc"],
                  "dCAP_ED_EU": R["E_D"]["cap"] - R["E_U"]["cap"],
                  "dCAP_ED_EU_pp": (R["E_D"]["cap"] - R["E_U"]["cap"]) / max(y.sum(), 1) * 100,
                  "dAUCsub_D_U": (R["D"]["auroc_subregion"] - R["U"]["auroc_subregion"])
                  if R["D"]["auroc_subregion"] is not None else None,
                  "dAUCne_D_U": R["D"]["auroc_nonempty"] - R["U"]["auroc_nonempty"]}
    # ranks of failures with intact whole tumour
    rk = {k: np.empty(len(d), int) for k in ("U", "D")}
    for k in rk:
        rk[k][order(S[k])] = np.arange(1, len(d) + 1)
    hi = typ == "sub_ge_0.8"
    R["ranks_sub_ge_0.8"] = {k: sorted(rk[k][hi].tolist()) for k in rk}
    allf = y == 1
    R["ranks_all_failures_U"] = sorted(rk["U"][allf].tolist())
    return R


def attach_published(d, pub_path):
    """Replace dice_* by the published values (matched on subject_id); report agreement."""
    p = pd.read_csv(pub_path)
    p.columns = [c.strip() for c in p.columns]
    low = {c.lower(): c for c in p.columns}
    idc = next(low[k] for k in ("subject_id", "subject", "case_id", "id") if k in low)
    reg = {r: pd.to_numeric(p[low[f"dice_{r.lower()}"]], errors="coerce") for r in ("WT", "TC", "ET")}
    pub = pd.DataFrame({"subject_id": p[idc].astype(str).str.strip(),
                        **{f"pub_{r}": v for r, v in reg.items()}})
    pub["pub_mean"] = pub[["pub_WT", "pub_TC", "pub_ET"]].mean(axis=1, skipna=True)
    m = d.merge(pub, on="subject_id", how="left")
    miss = int(m["pub_mean"].isna().sum())
    own = m["dice_mean"].values
    agree = {"n": len(m), "unmatched": miss,
             "max_abs_diff_mean": float(np.nanmax(np.abs(own - m["pub_mean"].values))),
             "failures_own": int((own < 0.5).sum()), "failures_published": int((m["pub_mean"] < 0.5).sum()),
             "failure_set_identical": bool(((own < 0.5) == (m["pub_mean"].values < 0.5)).all())}
    for r in ("WT", "TC", "ET"):
        m[f"dice_{r}"] = m[f"pub_{r}"]
    m["dice_mean"] = m["pub_mean"]
    return m, agree


res = {}
for name, spec in FILES.items():
    path, _, pub = spec.partition("::")
    d = pd.read_csv(path)
    agree = None
    if pub:
        d, agree = attach_published(d, pub)
    n_review = 54 if len(d) >= 200 else max(1, int(round(0.2 * len(d))))   # paper: top 54 in both cohorts
    res[name] = analyse(d, n_review, B)
    res[name]["published_dice_agreement"] = agree
    r = res[name]
    if agree:
        print(f"\n[{name}] Dice from published file; agreement with this run: {agree}")
    print(f"\n==== {name}: n={r['n']} patients={r['patients']} failures={r['failures']} "
          f"({r['failure_patients']} pts) empty={r['n_empty']} types={r['types']} mean Dice {r['mean_dice']:.4f}")
    print(f"rho U {r['rho_U_dice']:+.3f}  rho D {r['rho_D_dice']:+.3f}  rho(U,D) {r['rho_U_D']:+.3f}")
    for k in ("U", "D", "E_U", "E_D", "E_Dmax", "E_disET", "E_disTC", "E_rankUD", "V"):
        v = r[k]
        extra = ""
        if "auroc_subregion" in v:
            extra = f"  subregion AUROC {v['auroc_subregion']}"
        if "auroc_nonempty" in v:
            extra += f"  non-empty {v['auroc_nonempty']:.3f}"
        print(f"  {k:9s} AUROC {v['auroc']:.3f}  cap@{r['n_review']} {v['cap']:2d}  {v['by_type']}{extra}")
    print(f"  point {r['point']}")
    print(f"  ci    {r['ci']}")
    print(f"  ranks of WT>=0.8 failures: {r['ranks_sub_ge_0.8']}")
json.dump(res, open(OUT, "w"), indent=1, default=float)
print("\nwrote", OUT)
