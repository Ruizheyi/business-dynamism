# 34_outlier_check.py
# Script 33 threw one huge significant coefficient:
#   jrr, manuf+info subsample, rd b=2807.6, t=3.60, 1sd=+4.82pp
# Two orders of magnitude larger than everything else, wrong sign, and only
# in a 1716-obs subsample. Almost certainly small-denominator outliers:
# jrr = (JC+JD)/denom*100 explodes when denom is small.
# This script finds the driving cells and re-runs with winsorising.
# Run: py scripts\34_outlier_check.py

import os
import numpy as np
import pandas as pd

PANEL = os.path.join("output", "31_pdit_bds_panel.csv")
OUTDIR = "output"
HIGH = ["31-33", "51"]


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
        f = pd.factorize(d[a].astype(str) + "_" + d[b].astype(str))
        out.append((f[0], len(f[1])))
    return out


def fit(d, yname, treats):
    sub = d.dropna(subset=[yname] + treats).copy()
    if len(sub) < 100:
        return None
    cl = make_codes(sub)
    y = absorb(sub[yname].to_numpy(float), cl)
    X = np.column_stack([absorb(sub[t].to_numpy(float), cl) for t in treats])
    XtX = X.T @ X
    if np.linalg.cond(XtX) > 1e12:
        return None
    beta = np.linalg.solve(XtX, X.T @ y)
    e = y - X @ beta
    inv = np.linalg.inv(XtX)
    meat = np.zeros_like(XtX)
    for g in sub["st"].unique():
        m = (sub["st"] == g).to_numpy()
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    G = sub["st"].nunique(); n, k = X.shape
    V = inv @ meat @ inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    se = np.sqrt(np.diag(V))
    return {"n": n, "b": beta, "se": se, "t": beta / np.where(se > 0, se, np.nan),
            "sd": [sub[t].std() for t in treats], "resid": e, "X": X, "sub": sub}


def main():
    d = pd.read_csv(PANEL, dtype={"sector": str})
    d["sector"] = d["sector"].str.strip()

    print("=" * 74)
    print("1. HOW BAD IS THE jrr TAIL?")
    print("=" * 74)
    for name, s in [("all sectors", d), ("manuf+info", d[d.sector.isin(HIGH)])]:
        v = s["jrr"].dropna()
        q = v.quantile([.001, .01, .05, .5, .95, .99, .999])
        print("\n%s  N=%d  mean=%.2f  sd=%.2f  max=%.1f"
              % (name, len(v), v.mean(), v.std(), v.max()))
        print("  pctiles p0.1/p1/p5/p50/p95/p99/p99.9:")
        print("   ", " ".join("%.1f" % x for x in q.values))
        print("  cells with jrr > 100: %d (%.4f)" % ((v > 100).sum(), (v > 100).mean()))
        print("  cells with jrr > 200: %d" % (v > 200).sum())

    print("\n" + "=" * 74)
    print("2. THE WORST CELLS  (are they small-denom?)")
    print("=" * 74)
    w = d.nlargest(12, "jrr")[["st", "sector", "year", "jrr", "denom", "emp", "rd"]]
    print(w.to_string(index=False, float_format=lambda x: "%.1f" % x))
    print("\npanel median denom = %.0f" % d.denom.median())

    print("\n" + "=" * 74)
    print("3. WHICH CELLS DRIVE THE 2807 COEFFICIENT?")
    print("=" * 74)
    hi = d[d.sector.isin(HIGH)]
    r = fit(hi, "jrr", ["rd", "itc"])
    if r is not None:
        # contribution of each obs to beta_rd via the influence formula
        inv = np.linalg.inv(r["X"].T @ r["X"])
        infl = (inv @ r["X"].T).T[:, 0] * r["resid"]
        s = r["sub"].copy()
        s["influence"] = infl
        s["abs_infl"] = np.abs(infl)
        tot = s["abs_infl"].sum()
        top = s.nlargest(8, "abs_infl")[["st", "sector", "year", "jrr", "denom",
                                         "rd", "influence"]]
        print("baseline beta_rd = %.1f" % r["b"][0])
        print("\ntop 8 cells by absolute influence on beta_rd:")
        print(top.to_string(index=False, float_format=lambda x: "%.3f" % x))
        print("\ntop 8 cells account for %.3f of total |influence|"
              % (s.nlargest(8, "abs_infl")["abs_infl"].sum() / tot))
        print("top 20 cells account for %.3f"
              % (s.nlargest(20, "abs_infl")["abs_infl"].sum() / tot))

    print("\n" + "=" * 74)
    print("4. DOES IT SURVIVE WINSORISING / TRIMMING?")
    print("=" * 74)
    print("if the coefficient dies here, it was never a result.\n")

    def report(tag, dd):
        for y in ["jrr", "entry_rate"]:
            r = fit(dd[dd.sector.isin(HIGH)], y, ["rd", "itc"])
            if r is None:
                print("  %-30s %-11s (not estimable)" % (tag, y))
                continue
            print("  %-30s %-11s N=%5d  b_rd=%9.1f  t=%5.2f  1sd=%+.4fpp"
                  % (tag, y, r["n"], r["b"][0], r["t"][0], r["b"][0] * r["sd"][0]))

    report("raw", d)
    for lo, hi_p in [(1, 99), (5, 95)]:
        dd = d.copy()
        for c in ["jrr", "entry_rate", "exit_rate"]:
            a, b = dd[c].quantile([lo / 100, hi_p / 100])
            dd[c] = dd[c].clip(a, b)
        report("winsorised %d/%d" % (lo, hi_p), dd)
    for thr in [1000, 5000]:
        report("drop denom < %d" % thr, d[d.denom >= thr])

    print("\n" + "=" * 74)
    print("5. SAME CHECKS ON THE FULL SAMPLE (entry_rate, the primary outcome)")
    print("=" * 74)
    for tag, dd in [("raw", d),
                    ("winsorised 1/99", None),
                    ("drop denom < 1000", d[d.denom >= 1000])]:
        if dd is None:
            dd = d.copy()
            a, b = dd["entry_rate"].quantile([.01, .99])
            dd["entry_rate"] = dd["entry_rate"].clip(a, b)
        r = fit(dd, "entry_rate", ["rd", "itc"])
        if r is not None:
            print("  %-22s N=%5d  b_rd=%8.1f  t=%5.2f  1sd=%+.4fpp"
                  % (tag, r["n"], r["b"][0], r["t"][0], r["b"][0] * r["sd"][0]))


if __name__ == "__main__":
    main()