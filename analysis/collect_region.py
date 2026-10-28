"""Merge analyse_region.py output with the runner's .verify.json files into region.json.

usage: collect_region.py analysis.json base_test.verify.json topo_test.verify.json out_region.json
"""
import json
import sys

A = json.load(open(sys.argv[1]))
vb, vt = (json.load(open(p)) for p in sys.argv[2:4])
diffs = [v["dice"]["max_abs_diff"] for v in (vb, vt)]
A["repro"] = {
    "max_dice_diff": max(d for d in diffs if d is not None),
    "failure_sets_identical": all(v["dice"]["failure_set_identical"] for v in (vb, vt)),
    "n_matched": [v["dice"]["n_matched"] for v in (vb, vt)],
    "rho_U_base": vb["uncertainty"]["spearman_mean_fgvar_fg"],
    "rho_U_topo": vt["uncertainty"]["spearman_mean_fgvar_fg"],
}
json.dump(A, open(sys.argv[4], "w"), indent=1)
print("wrote", sys.argv[4], A["repro"])
