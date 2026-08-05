# 33_baseline_and_robustness.py
# Phase 2 step 4: DIAGNOSTIC, not results.
#
# Runs naive TWFE to see the sign / magnitude, then asks the two questions
# that decide whether this is a paper:
#   1. does the estimate survive dropping any single state?
#      (script 32: top 5 states = 58% of identifying variation, eff. clusters 11.7)
#   2. is it only coming from manufacturing + information?
#      (script 32: sectors 31-33 and 51 = 84% of identifying variation)
#
# WARNING on inference: TWFE with staggered adoption and continuous treatment
# is biased (Goodman-Bacon 2021, de Chaisemartin-D'Haultfoeuille 2020). The
# p-values below use clustered SE which are unreliable at ~12 effective
# clusters. Nothing here goes in the paper. Point estimates and stability only.
#
# Run: py scripts\33_baseline_and_robustness.py

import os
import numpy as np
import pandas as pd

PANEL = os.path.join("output", "31_pdit_bds_panel.csv")
OUTDIR = "output"

OUTCOMES = ["entry_rate", "jrr", "exit_rate"]
TREATS = ["rd", "itc"]
HIGH_VAR_SECTORS = ["31-33", "51"]


def demean_by(v, codes, ng):
    s = np.bincount(codes, weights=v, minlength=ng)
    c = np.bincount(codes, minlength=ng).astype(float)
    m = np.divide(s, c, out=np.zeros(ng), where=c > 0)
    return v - m[codes]


def absorb(v, cl, tol=1e-10, maxiter=500):
    r = np.asarray(v, dtype=float).copy()
    for _ in range(maxiter):
        prev = r.copy()
        for codes, ng in cl:
            r = demean_by(r, codes, ng)
        if np.max(np.abs(r - prev)) < tol:
            break
    return r


def make_codes(d):
    out = []
    for a, b in [("st", "year"), ("sector", "year"), ("st", "sector")]:
        key = d[a].astype(str) + "_" + d[b].astype(str)
        f = pd.factorize(key)
        out.append((f[0], len(f[1])))
    return out


def fit(d, yname, treats, weights=None):
    """within estimator with 3 FE, cluster on state. returns dict or None."""
    sub = d.dropna(subset=[yname] + treats).copy()
    if len(sub) < 200:
        return None
    cl = make_codes(sub)

    y = absorb(sub[yname].to_numpy(float), cl)
    X = np.column_stack([absorb(sub[t].to_numpy(float), cl) for t in treats])

    if weights is not None:
        w = np.sqrt(sub[weights].to_numpy(float))
        w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
        y = y * w
        X = X * w[:, None]

    XtX = X.T @ X
    if np.linalg.cond(XtX) > 1e12:
        return None
    beta = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ beta

    # cluster-robust on state (unreliable here, reported for reference only)
    inv = np.linalg.inv(XtX)
    meat = np.zeros_like(XtX)
    for g in sub["st"].unique():
        m = (sub["st"] == g).to_numpy()
        Xg, eg = X[m], e[m]
        s = Xg.T @ eg
        meat += np.outer(s, s)
    G = sub["st"].nunique()
    n, k = X.shape
    adj = (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    V = inv @ meat @ inv * adj
    se = np.sqrt(np.diag(V))

    return {"n": n, "beta": beta, "se": se,
            "t": beta / np.where(se > 0, se, np.nan),
            "sd_treat": [sub[t].std() for t in treats]}


def show(tag, r, treats):
    if r is None:
        print("  %-28s  (not estimable)" % tag)
        return
    parts = []
    for i, t in enumerate(treats):
        # effect of a 1sd move in the treatment, in pp of the outcome
        eff = r["beta"][i] * r["sd_treat"][i]
        parts.append("%s b=%9.1f t=%5.2f  1sd=%+.4fpp" %
                     (t, r["beta"][i], r["t"][i], eff))
    print("  %-28s N=%5d  %s" % (tag, r["n"], "  |  ".join(parts)))


def main():
    d = pd.read_csv(PANEL, dtype={"sector": str})
    d["sector"] = d["sector"].str.strip()
    print("panel: %d rows, %d states, %d sectors, %d-%d"
          % (len(d), d.st.nunique(), d.sector.nunique(), d.year.min(), d.year.max()))
    print("\nNOTE: TWFE is biased under staggered adoption. p-values unreliable")
    print("at ~12 effective clusters. Point estimates and STABILITY only.\n")

    # ---------------- 1. baseline ----------------
    print("=" * 78)
    print("1. BASELINE TWFE  (state-year + sector-year + state-sector FE)")
    print("=" * 78)
    print("b is per 1 unit of treatment (= 100pp of value added), so it is huge.")
    print("'1sd' is the readable number: pp change in outcome per 1sd of treatment.\n")

    base = {}
    for y in OUTCOMES:
        print("%s:" % y)
        r1 = fit(d, y, ["rd"])
        show("rd only", r1, ["rd"])
        r2 = fit(d, y, ["rd", "itc"])
        show("rd + itc (horse race)", r2, ["rd", "itc"])
        rw = fit(d, y, ["rd", "itc"], weights="denom")
        show("rd + itc, emp weighted", rw, ["rd", "itc"])
        base[y] = r2
        print()

    # ---------------- 2. leave one state out ----------------
    print("=" * 78)
    print("2. LEAVE-ONE-STATE-OUT  (entry_rate, rd coefficient)")
    print("=" * 78)
    full = base["entry_rate"]
    if full is None:
        print("baseline failed, skipping")
    else:
        b_full = full["beta"][0]
        print("full sample beta_rd = %.1f\n" % b_full)
        rows = []
        for s in sorted(d.st.unique()):
            r = fit(d[d.st != s], "entry_rate", ["rd", "itc"])
            if r is not None:
                rows.append({"dropped_fips": s, "beta_rd": r["beta"][0],
                             "pct_change": (r["beta"][0] - b_full) / abs(b_full) * 100})
        lo = pd.DataFrame(rows).sort_values("pct_change")
        print("largest swings when a state is dropped:")
        print(lo.head(5).to_string(index=False, float_format=lambda x: "%.1f" % x))
        print("  ...")
        print(lo.tail(5).to_string(index=False, float_format=lambda x: "%.1f" % x))
        print("\n  range of beta_rd: %.1f to %.1f" % (lo.beta_rd.min(), lo.beta_rd.max()))
        print("  sign flips in %d of %d drops"
              % ((np.sign(lo.beta_rd) != np.sign(b_full)).sum(), len(lo)))
        lo.to_csv(os.path.join(OUTDIR, "33_leave_one_out.csv"), index=False)

    # ---------------- 3. where does it come from ----------------
    print("\n" + "=" * 78)
    print("3. HIGH-VARIATION SECTORS vs THE REST")
    print("=" * 78)
    print("sectors 31-33 and 51 hold 84%% of identifying variation.")
    print("if the effect only exists there, the paper is about those sectors.\n")
    hi = d[d.sector.isin(HIGH_VAR_SECTORS)]
    lo_s = d[~d.sector.isin(HIGH_VAR_SECTORS)]
    for y in OUTCOMES:
        print("%s:" % y)
        show("manuf + info only", fit(hi, y, ["rd", "itc"]), ["rd", "itc"])
        show("all other sectors", fit(lo_s, y, ["rd", "itc"]), ["rd", "itc"])
        print()

    # ---------------- 4. pre-trend smell test ----------------
    print("=" * 78)
    print("4. LEAD TEST  (entry_rate on rd led 3 years)")
    print("=" * 78)
    print("if the future predicts the present, the design has a pre-trend problem.\n")
    d2 = d.sort_values(["st", "sector", "year"]).copy()
    d2["rd_lead3"] = d2.groupby(["st", "sector"])["rd"].shift(-3)
    d2["itc_lead3"] = d2.groupby(["st", "sector"])["itc"].shift(-3)
    show("rd lead3 + itc lead3",
         fit(d2, "entry_rate", ["rd_lead3", "itc_lead3"]),
         ["rd_lead3", "itc_lead3"])
    print("\n  (a large lead coefficient is bad news, not a result)")


if __name__ == "__main__":
    main()