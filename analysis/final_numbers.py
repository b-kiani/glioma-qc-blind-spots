"""Every number the revised paper cites from the per-case evaluation file.

Point estimates are full-sample values; intervals are 2.5/97.5 percentiles of a
patient-clustered bootstrap (patients resampled, all timepoints kept).
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

d = pd.read_csv(sys.argv[1]); out = sys.argv[2]; B = int(sys.argv[3])
N = len(d); pats = d.patient_id.unique()
idx = {p: np.flatnonzero(d.patient_id.values == p) for p in pats}
NS = (14, 27, 41, 54, 81)

def order(s): return np.lexsort((np.arange(len(s)), -s))
def cap(s, y, n): return int(y[order(s)[:n]].sum())
def auc(y, s):
    s = np.where(np.isinf(s), np.nanmax(s[np.isfinite(s)]) + 1, s)
    return roc_auc_score(y, s)

R = {"n_timepoints": N, "n_patients": len(pats),
     "gt_fg_empty": int((d.gt_fg_volume == 0).sum()),
     "gt_et_absent": int((~d.gt_et_present).sum()),
     "gt_rc_present": int(d.gt_rc_present.sum())}
rng = np.random.default_rng(2027)
boots = [np.concatenate([idx[p] for p in rng.choice(pats, len(pats))]) for _ in range(B)]

for m, name in (("base", "Baseline"), ("topo", "TopoGraph")):
    dice = d[f"mean_dice_legacy_{m}"].values; y = (dice < 0.5).astype(int)
    u = d[f"mean_fgvar_fg_{m}"].values; lv = d[f"log_pred_fg_volume_{m}"].values
    wt = d[f"dice_WT_{m}"].values; empty = lv == 0; ne = ~empty
    E = np.where(empty, np.inf, u); V = -lv
    typ = np.full(N, "none", dtype=object)
    typ[(y == 1) & empty] = "T1"
    typ[(y == 1) & ne & (wt < 0.5)] = "T2"
    typ[(y == 1) & ne & (wt >= 0.5) & (wt < 0.8)] = "T3a"
    typ[(y == 1) & ne & (wt >= 0.8)] = "T3b"
    M = {"failures": int(y.sum()), "failure_patients": int(d.patient_id[y == 1].nunique()),
         "mean_dice": float(dice.mean()),
         "dice_WT": float(np.nanmean(d[f"dice_WT_{m}"])), "dice_TC": float(np.nanmean(d[f"dice_TC_{m}"])),
         "dice_ET": float(np.nanmean(d[f"dice_ET_{m}"])),
         "n_empty": int(empty.sum()), "empty_failures": int(y[empty].sum()),
         "empty_gt_fg_median": float(np.median(d.gt_fg_volume.values[empty])),
         "rho_U_dice": float(spearmanr(u, dice).statistic)}
    rk = np.empty(N, int); rk[order(u)] = np.arange(1, N + 1)
    M["empty_U_ranks"] = sorted(rk[empty].tolist())
    arms = {"U": u, "V": V, "E": E}
    for a, s in arms.items():
        M[f"auroc_{a}"] = float(auc(y, s))
        for n in NS: M[f"cap_{a}_{n}"] = cap(s, y, n)
        M[f"unreviewed_fail_{a}_54"] = int(y.sum() - cap(s, y, 54))
        M[f"patients_hit_{a}_54"] = int(d.patient_id.values[order(s)[:54]][y[order(s)[:54]] == 1].__len__() and
                                         len(set(d.patient_id.values[order(s)[:54]][y[order(s)[:54]] == 1])))
    M["random_cap_54"] = 54 * y.sum() / N
    M["auroc_U_nonempty"] = float(roc_auc_score(y[ne], u[ne]))
    M["auroc_V_nonempty"] = float(roc_auc_score(y[ne], V[ne]))
    t3 = np.isin(typ, ["T3a", "T3b"]); m3 = t3 | (y == 0)
    M["auroc_U_subregion"] = float(roc_auc_score(t3[m3], u[m3]))
    M["auroc_V_subregion"] = float(roc_auc_score(t3[m3], V[m3]))
    # taxonomy table
    tax = {}
    for t in ("T1", "T2", "T3a", "T3b"):
        sel = typ == t
        tax[t] = {"n": int(sel.sum()), "patients": int(d.patient_id[sel].nunique()),
                  **{a: int(sel[order(s)[:54]].sum()) for a, s in arms.items()},
                  "gt_fg_median": float(np.median(d.gt_fg_volume.values[sel])) if sel.any() else None,
                  "gt_et_lt100": int((sel & (d.gt_et_volume.values > 0) & (d.gt_et_volume.values < 100)).sum()),
                  "ids": d.case_id.values[sel].tolist()}
    M["taxonomy"] = tax
    # bootstrap
    bs = {k: [] for k in ("rho", "U", "V", "E", "EmU", "VmU", "capU", "capV", "capE", "capEmU",
                          "Une", "Vne", "UmVne", "Usub", "Vsub", "UmVsub")}
    for ii in boots:
        yb = y[ii]
        if not (0 < yb.sum() < len(ii)): continue
        bs["rho"].append(spearmanr(u[ii], dice[ii]).statistic)
        a_u, a_v, a_e = auc(yb, u[ii]), auc(yb, V[ii]), auc(yb, E[ii])
        bs["U"].append(a_u); bs["V"].append(a_v); bs["E"].append(a_e)
        bs["EmU"].append(a_e - a_u); bs["VmU"].append(a_v - a_u)
        cu, cv, ce = (cap(s[ii], yb, 54) / yb.sum() * 100 for s in (u, V, E))
        bs["capU"].append(cu); bs["capV"].append(cv); bs["capE"].append(ce); bs["capEmU"].append(ce - cu)
        jj = ii[ne[ii]]
        if 0 < y[jj].sum() < len(jj):
            a1, a2 = roc_auc_score(y[jj], u[jj]), roc_auc_score(y[jj], V[jj])
            bs["Une"].append(a1); bs["Vne"].append(a2); bs["UmVne"].append(a1 - a2)
        kk = ii[m3[ii]]
        if 0 < t3[kk].sum() < len(kk):
            a1, a2 = roc_auc_score(t3[kk], u[kk]), roc_auc_score(t3[kk], V[kk])
            bs["Usub"].append(a1); bs["Vsub"].append(a2); bs["UmVsub"].append(a1 - a2)
    M["ci"] = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in bs.items()}
    M["point_diffs"] = {"EmU": M["auroc_E"] - M["auroc_U"], "VmU": M["auroc_V"] - M["auroc_U"],
                        "UmVne": M["auroc_U_nonempty"] - M["auroc_V_nonempty"],
                        "UmVsub": M["auroc_U_subregion"] - M["auroc_V_subregion"],
                        "capEmU_pp": (M["cap_E_54"] - M["cap_U_54"]) / y.sum() * 100}
    # threshold robustness
    M["thresholds"] = {}
    for thr in (0.4, 0.6):
        yt = (dice < thr).astype(int)
        M["thresholds"][str(thr)] = {"failures": int(yt.sum()), "U": float(auc(yt, u)), "V": float(auc(yt, V)),
                                     "E": float(auc(yt, E)),
                                     "Une": float(roc_auc_score(yt[ne], u[ne])), "Vne": float(roc_auc_score(yt[ne], V[ne])),
                                     "capU": cap(u, yt, 54), "capE": cap(E, yt, 54), "capV": cap(V, yt, 54)}
    R[name] = M

# overlap of failure types across models
for t in ("T1", "T3b"):
    a, b = set(R["Baseline"]["taxonomy"][t]["ids"]), set(R["TopoGraph"]["taxonomy"][t]["ids"])
    R[f"overlap_{t}"] = {"base": len(a), "topo": len(b), "shared": len(a & b), "union": len(a | b)}
for nm in ("Baseline", "TopoGraph"):
    for t in R[nm]["taxonomy"]: R[nm]["taxonomy"][t].pop("ids")
json.dump(R, open(out, "w"), indent=1)

f = lambda x: f"{x:.3f}"; c = lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]"
for nm in ("Baseline", "TopoGraph"):
    M = R[nm]
    print(f"\n==== {nm}: {M['failures']} failures / {M['failure_patients']} patients; mean Dice {M['mean_dice']:.4f}; "
          f"WT/TC/ET {M['dice_WT']:.3f}/{M['dice_TC']:.3f}/{M['dice_ET']:.3f}")
    print(f"rho {f(M['rho_U_dice'])} {c(M['ci']['rho'])}; empty {M['n_empty']} (fail {M['empty_failures']}), ranks {M['empty_U_ranks']}, GT median {M['empty_gt_fg_median']:.0f}")
    for a in "UVE":
        print(f"{a}: AUROC {f(M['auroc_'+a])} {c(M['ci'][a])}  cap " + "/".join(str(M[f'cap_{a}_{n}']) for n in NS)
              + f"  @54 CI {M['ci']['cap'+a][0]:.0f}-{M['ci']['cap'+a][1]:.0f}%  unrev.fail {M['unreviewed_fail_'+a+'_54']}  patients hit {M['patients_hit_'+a+'_54']}")
    p = M["point_diffs"]
    print(f"E-U {p['EmU']:+.3f} {c(M['ci']['EmU'])}; cap E-U {p['capEmU_pp']:+.1f}pp [{M['ci']['capEmU'][0]:+.1f},{M['ci']['capEmU'][1]:+.1f}]; V-U {p['VmU']:+.3f} {c(M['ci']['VmU'])}")
    print(f"non-empty: U {f(M['auroc_U_nonempty'])} {c(M['ci']['Une'])}  V {f(M['auroc_V_nonempty'])} {c(M['ci']['Vne'])}  U-V {p['UmVne']:+.3f} {c(M['ci']['UmVne'])}")
    print(f"sub-region vs non-fail: U {f(M['auroc_U_subregion'])} {c(M['ci']['Usub'])}  V {f(M['auroc_V_subregion'])} {c(M['ci']['Vsub'])}  U-V {p['UmVsub']:+.3f} {c(M['ci']['UmVsub'])}")
    for t, v in M["taxonomy"].items(): print(f"  {t}: {v}")
    for t, v in M["thresholds"].items(): print(f"  thr {t}: {v}")
print("\noverlap:", R["overlap_T1"], R["overlap_T3b"])
print("cohort:", {k: R[k] for k in ("n_timepoints", "n_patients", "gt_fg_empty", "gt_et_absent", "gt_rc_present")})
