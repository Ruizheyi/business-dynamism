# 32_treatment_diagnostics.py
# Before any regression: can rd and itc actually be told apart after the FE?
# The tangible-vs-intangible contrast is the paper's main claim. If the two
# residualised treatments are collinear it dies, and we need to know now.
# Also checks how concentrated the identifying variation is across states.
# Run: py scripts\32_treatment_diagnostics.py

import os
import numpy as np
import pandas as pd

PANEL = os.path.join("output", "31_pdit_bds_panel.csv")


def demean_by(v, codes, ng):
    s = np.bincount(codes, weights=v, minlength=ng)
    c = np.bincount(codes, minlength=ng).astype(float)
    m = np.divide(s, c, out=np.zeros(ng), where=c > 0)
    return v - m[codes]


def absorb(v, cl, tol=1e-10, maxiter=500):
    r = v.astype(float).copy()
    for _ in range(maxiter):
        prev = r.copy()
        for codes, ng in cl:
            r = demean_by(r, codes, ng)
        if np.max(np.abs(r - prev)) < tol:
            break
    return r


def main():
    d = pd.read_csv(PANEL)
    d["sy"] = d["st"].astype(str) + "_" + d["year"].astype(str)
    d["jy"] = d["sector"].astype(str) + "_" + d["year"].astype(str)
    d["sj"] = d["st"].astype(str) + "_" + d["sector"].astype(str)

    codes = []
    for c in ["sy", "jy", "sj"]:
        f = pd.factorize(d[c])
        codes.append((f[0], len(f[1])))

    for v in ["rd", "itc"]:
        d["r_" + v] = absorb(d[v].to_numpy(dtype=float), codes)

    a, b = d["r_rd"].to_numpy(), d["r_itc"].to_numpy()
    rho = np.corrcoef(a, b)[0, 1]

    print("=" * 62)
    print("CAN rd AND itc BE SEPARATED?")
    print("=" * 62)
    print("corr(resid rd, resid itc) = %.4f" % rho)
    print("R2 of one on the other    = %.4f" % rho ** 2)
    print("VIF                       = %.2f" % (1 / (1 - rho ** 2)))
    print("SE inflation vs single-treatment model = %.2fx"
          % (1 / np.sqrt(1 - rho ** 2)))
    print()
    if abs(rho) < 0.3:
        print("  VERDICT: fine. horse race is identified.")
    elif abs(rho) < 0.6:
        print("  VERDICT: workable but report both separately AND jointly.")
    else:
        print("  VERDICT: collinear. the tangible/intangible contrast is")
        print("           not separately identified in this panel.")

    # where does the identifying variation actually come from
    print("\n" + "=" * 62)
    print("WHERE DOES THE rd VARIATION COME FROM?")
    print("=" * 62)
    ss = pd.Series(a ** 2, index=d["st"]).groupby(level=0).sum()
    ss = (ss / ss.sum()).sort_values(ascending=False)
    print("top 8 states by share of residual variance:")
    for k, v in ss.head(8).items():
        print("   FIPS %2d   %.3f" % (k, v))
    print("top 5 states account for %.3f" % ss.head(5).sum())
    # effective number of clusters (Carter-Schnepel-Steigerwald style, crude)
    print("effective N clusters ~= %.1f  (nominal 33)" % (1 / (ss ** 2).sum()))

    print("\ntop 5 sectors by share of residual variance:")
    sj = pd.Series(a ** 2, index=d["sector"]).groupby(level=0).sum()
    sj = (sj / sj.sum()).sort_values(ascending=False)
    for k, v in sj.head(5).items():
        print("   sector %-6s %.3f" % (k, v))


if __name__ == "__main__":
    main()