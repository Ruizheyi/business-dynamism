# 37_ref_period.py
# Script 36 showed pre (k<=-2) beta=+0.366 and mid (0<=k<5) beta=+0.412.
# Nearly identical. That means the apparently marginal mid effect (p=0.057)
# is a level shift relative to the omitted period, not an effect of adoption.
# k=-1 looks like a low point. This script checks how much the results move
# when the reference period changes, and reports the mid-minus-pre contrast
# which is the quantity that is actually invariant to that choice.
# Run: py scripts\37_ref_period.py

import os
import numpy as np
import pandas as pd

exec(open(os.path.join("scripts", "36_event_study_wcb.py")).read()
     .split("def main()")[0])   # reuse loaders, xwalk, absorb, wcb, ols

REFS = [-1, -2, -3, "avg_pre"]


def run(d, yname, ref, rng):
    sub = d.dropna(subset=[yname]).copy()
    e = sub["exposure"].to_numpy(float)
    trt = (sub["k"] > -900).to_numpy()

    if ref == "avg_pre":
        # omit the whole pre window; identify off post vs pre average
        pre = ((sub["k"] <= -2) & trt).to_numpy(float) * 0.0
    cols, names = [], []
    if ref != "avg_pre":
        pre_mask = (sub["k"] <= -2) & trt & (sub["k"] != ref)
    else:
        pre_mask = np.zeros(len(sub), dtype=bool)
    for nm, mask in [("pre", pre_mask),
                     ("mid", (sub["k"] >= 0) & (sub["k"] < 5) & trt),
                     ("post", (sub["k"] >= 5) & trt)]:
        v = np.asarray(mask, dtype=float) * e
        if v.std() > 1e-12:
            cols.append(v); names.append(nm)
    X = np.column_stack(cols)
    y = sub[yname].to_numpy(float)

    cl = []
    for a, b in [("st", "year"), ("sector", "year"), ("st", "sector")]:
        f = pd.factorize(sub[a].astype(str) + "_" + sub[b].astype(str))
        cl.append((f[0], len(f[1])))
    y = absorb(y, cl); X = absorb(X, cl)
    gid = sub["st"].to_numpy()

    out = {}
    for j, nm in enumerate(names):
        bj, sj, tj, pj, lo, hi = wcb(X, y, gid, j, rng)
        out[nm] = (bj, sj, pj, lo, hi)
    return out, names


def main():
    rng = np.random.default_rng(20260805)
    p = load_pdit()
    g, ad, never, always = build(p)
    b = load_bds()
    use = {FIPS[s]: y for s, y in ad.items()}
    ctrl = {FIPS[s] for s in never}
    d = b[b["st"].isin(set(use) | ctrl)].copy()
    d["adopt"] = d["st"].map(use)
    d["k"] = np.where(d["adopt"].notna(), d["year"] - d["adopt"], -999)
    d["exposure"] = d["sector"].map(g)
    d = d.dropna(subset=["exposure"])

    print("=" * 74)
    print("REFERENCE PERIOD SENSITIVITY: entry_rate")
    print("=" * 74)
    print("if 'mid' and 'post' move a lot across rows, the omitted period is")
    print("doing the work, not the policy.\n")
    print("%-10s %-6s %9s %9s %8s   %s" % ("ref", "term", "beta", "se", "WCBp", "95% CI"))
    for ref in REFS:
        out, names = run(d, "entry_rate", ref, rng)
        for nm in names:
            bj, sj, pj, lo, hi = out[nm]
            print("%-10s %-6s %9.4f %9.4f %8.3f   [%+.4f, %+.4f]"
                  % (str(ref), nm, bj, sj, pj, lo, hi))
        print()

    print("=" * 74)
    print("THE INVARIANT QUANTITY: post minus pre")
    print("=" * 74)
    print("this contrast does not depend on which single year is omitted.\n")
    for yname in ["entry_rate", "exit_rate", "jrr"]:
        sub = d.dropna(subset=[yname]).copy()
        e = sub["exposure"].to_numpy(float)
        trt = (sub["k"] > -900).to_numpy()
        post = ((sub["k"] >= 0) & trt).astype(float) * e
        X = np.asarray(post, dtype=float)[:, None]
        y = sub[yname].to_numpy(float)
        cl = []
        for a, bb in [("st", "year"), ("sector", "year"), ("st", "sector")]:
            f = pd.factorize(sub[a].astype(str) + "_" + sub[bb].astype(str))
            cl.append((f[0], len(f[1])))
        y = absorb(y, cl); X = absorb(X, cl)
        bj, sj, tj, pj, lo, hi = wcb(X, y, sub["st"].to_numpy(), 0, rng)
        mean = sub[yname].mean()
        print("%-11s beta=%+.4f  se=%.4f  WCB p=%.3f  95%% CI [%+.4f, %+.4f]"
              % (yname, bj, sj, pj, lo, hi))
        print("            outcome mean %.2f  ->  CI in %% terms [%+.1f%%, %+.1f%%]"
              % (mean, 100 * lo / mean, 100 * hi / mean))
        print("            FGS state-level estimate for reference: +20%\n")


if __name__ == "__main__":
    main()