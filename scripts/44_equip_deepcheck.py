# 44_equip_deepcheck.py
# Self-audit of script 43 found three problems with the randomisation test.
#
# 1. NOT ENOUGH PERMUTATIONS. 500 draws put the Monte Carlo standard error at
#    0.0097 around p=0.05, so the 95% interval for the true p runs 0.031 to
#    0.069. That cannot distinguish 0.03 from 0.07. Raised to 5000 here.
#
# 2. THE TEST ANSWERS THE WRONG QUESTION. Permuting exposure across sectors
#    tests "no sector-specific response at all". If one sector responds
#    strongly for reasons unrelated to equipment, and happens to be
#    high-equipment, the test rejects anyway. It cannot establish that
#    equipment intensity is the operative characteristic. Section 3 adds
#    leave-one-sector-out, which can.
#
# 3. The "equip + aggregate post" row in script 43 returned not estimable
#    three times. That was a design error on my part: post is fully absorbed
#    by state-year FE, so that specification cannot exist. Dropped.
#
# The coefficient under examination: equipment intensity gradient on job
# reallocation, +2.17 (p=0.021 clustered), the only significant coefficient
# among the 24 gradient estimates in script 42, and it is the placebo.
#
# Run: py scripts\44_equip_deepcheck.py

import os
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
PDIT = os.path.join(RAW, "export.csv")
BDS = os.path.join(RAW, "bds2023_st_sec.csv")
BEA = os.path.join("output", "42a_bea_exposure.csv")
OUTDIR = "output"

RATIO = "Discounted - 12% Rate"
NPERM = 5000
NBOOT = 999
SEED = 20260805
BAL_LO, BAL_HI = -5, 4

XWALK = {
    "Construction": "23", "Wholesale Trade": "42", "Retail Trade": "44-45",
    "warehousing and storeage": "48-49",
    "Broadcasting and Telecommunications": "51",
    "Publishing industries (includes software)": "51",
    "Information and data processing services": "51",
    "Credit Intermediation": "52",
    "Insurance carriers and related activities": "52",
    "Securities, commodity contracts, other financial investments, and related activities": "52",
    "Rental and leasing services and lessors of intangible assets": "53",
    "Computer systems design and related services": "54",
    "Legal Services": "54",
    "Miscellaneous professional, scientific, and technical services": "54",
    "Management of companies (holding companies)": "55",
    "Administrative and support services": "56",
    "Waste Management and remediation services": "56",
    "Educational Services": "61",
    "Hospitals, nursing, and residential care facilities": "62",
    "Offices of health practioners and outpatient care centers": "62",
    "Miscellaneous health care and social assistance": "62",
    "Amusement, gambling, and recreation industries": "71",
    "Performing arts, spectator sports, museums and entertainment": "71",
    "Accommodation": "72", "Food services and drinking places": "72",
    "Other services": "81",
}
FIPS = {"AL": 1, "AZ": 4, "CA": 6, "CO": 8, "CT": 9, "DC": 11, "FL": 12,
        "GA": 13, "IA": 19, "IL": 17, "IN": 18, "KY": 21, "LA": 22,
        "MA": 25, "MD": 24, "MI": 26, "MN": 27, "MO": 29, "NC": 37,
        "NE": 31, "NJ": 34, "NM": 35, "NV": 32, "NY": 36, "OH": 39,
        "OR": 41, "PA": 42, "SC": 45, "TN": 47, "TX": 48, "VA": 51,
        "WA": 53, "WI": 55}


def demean_by(v, codes, ng):
    s = np.bincount(codes, weights=v, minlength=ng)
    c = np.bincount(codes, minlength=ng).astype(float)
    m = np.divide(s, c, out=np.zeros(ng), where=c > 0)
    return v - m[codes]


def absorb(M, cl, tol=1e-9, maxiter=300):
    A = np.asarray(M, dtype=float).copy()
    one = A.ndim == 1
    if one:
        A = A[:, None]
    for _ in range(maxiter):
        prev = A.copy()
        for codes, ng in cl:
            for j in range(A.shape[1]):
                A[:, j] = demean_by(A[:, j], codes, ng)
        if np.max(np.abs(A - prev)) < tol:
            break
    return A[:, 0] if one else A


def codes_for(sub, specs):
    out = []
    for cols in specs:
        key = sub[list(cols)].astype(str).agg("_".join, axis=1)
        f = pd.factorize(key)
        out.append((f[0], len(f[1])))
    return out


def ols(X, y):
    XtX = X.T @ X
    return np.linalg.solve(XtX, X.T @ y), np.linalg.inv(XtX)


def cluster_vcv(X, e, inv, gid):
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(gid):
        m = gid == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    G = len(np.unique(gid)); n, k = X.shape
    return inv @ meat @ inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))


def wcb(X, y, gid, j, rng, nboot=NBOOT):
    beta, inv = ols(X, y)
    e = y - X @ beta
    se = np.sqrt(np.diag(cluster_vcv(X, e, inv, gid)))
    if not np.isfinite(se[j]) or se[j] <= 0:
        return beta[j], np.nan, np.nan, np.nan
    t0 = beta[j] / se[j]
    if X.shape[1] > 1:
        Xr = np.delete(X, j, axis=1)
        br, _ = ols(Xr, y)
        er = y - Xr @ br
    else:
        Xr = np.zeros((len(y), 1)); br = np.zeros(1); er = y.copy()
    groups = np.unique(gid)
    ts = []
    for _ in range(nboot):
        w = rng.choice([-1.0, 1.0], size=len(groups))
        wmap = dict(zip(groups, w))
        wv = np.array([wmap[g] for g in gid])
        yb = Xr @ br + er * wv
        try:
            bb, invb = ols(X, yb)
        except np.linalg.LinAlgError:
            continue
        eb = yb - X @ bb
        sb = np.sqrt(np.diag(cluster_vcv(X, eb, invb, gid)))
        if np.isfinite(sb[j]) and sb[j] > 0:
            ts.append(bb[j] / sb[j])
    ts = np.array(ts)
    if len(ts) < 50:
        return beta[j], se[j], np.nan, np.nan
    p = (np.abs(ts) >= abs(t0)).mean()
    lo_t, hi_t = np.quantile(ts, [0.025, 0.975])
    return beta[j], p, beta[j] - hi_t * se[j], beta[j] - lo_t * se[j]


def load_pdit():
    p = pd.read_csv(PDIT)
    p["Industry"] = p["Industry"].str.strip()
    p = p[p["Ratio Type"] == RATIO]
    p = p[~p["Industry"].isin(["All Export", "All Non-Export"])].copy()
    xw = dict(XWALK)
    for m in [i for i in p["Industry"].unique() if i not in xw]:
        xw[m] = "31-33"
    p["sector"] = p["Industry"].map(xw)
    return p


def build(p):
    rd, va = "Research and Development Credit", "Industry Value-Added"
    live = p[p.groupby(["State", "Base Year"])[rd].transform("max") > 0]
    g = ((live[rd] * live[va]).groupby(live["sector"]).sum()
         / live[va].groupby(live["sector"]).sum())
    g = (g / g.max()).rename("pdit_exp")
    sy = p.groupby(["State", "Base Year"])[rd].max().reset_index()
    sy["on"] = (sy[rd] > 0).astype(int)
    first = sy[sy.on == 1].groupby("State")["Base Year"].min()
    return g, first[first > 1990], sorted(set(p["State"]) - set(first.index)), sy


def load_bds(keep):
    b = pd.read_csv(BDS, dtype={"st": str, "sector": str}, low_memory=False)
    b.columns = [c.strip().lower() for c in b.columns]
    for c in b.columns:
        if c not in ("st", "sector"):
            b[c] = pd.to_numeric(b[c], errors="coerce")
    b["st"] = pd.to_numeric(b["st"], errors="coerce")
    b["sector"] = b["sector"].str.strip()
    b = b.sort_values(["st", "sector", "year"])
    b["estabs_lag"] = b.groupby(["st", "sector"])["estabs"].shift(1)
    b["den"] = (b["estabs"] + b["estabs_lag"]) / 2
    b["entry_rate"] = b["estabs_entry"] / b["den"] * 100
    b["jrr"] = (b["job_creation"] + b["job_destruction"]) / b["denom"] * 100
    b["jc_rate"] = b["job_creation"] / b["denom"] * 100
    b["jd_rate"] = b["job_destruction"] / b["denom"] * 100
    return b[b["sector"].isin(keep)]


FE_GRAD = [("st", "year"), ("sector", "year"), ("st", "sector")]


def point_est(sub, yname, xcol, cl=None):
    """residualised OLS point estimate only, for speed inside loops"""
    if cl is None:
        cl = codes_for(sub, FE_GRAD)
    y = absorb(sub[yname].to_numpy(float), cl)
    x = absorb(sub[xcol].to_numpy(float), cl)
    if x.std() < 1e-12:
        return np.nan
    b, _ = ols(x[:, None], y)
    return b[0]


def main():
    rng = np.random.default_rng(SEED)
    p = load_pdit()
    g, adopters, never, sy = build(p)
    bea = pd.read_csv(BEA, dtype={"sector": str}).set_index("sector")
    ex = pd.DataFrame({"pdit_exp": g, "bea_exp": bea["bea_norm"],
                       "equip_exp": bea["equip_norm"]}).dropna()

    adopt_fips = {FIPS[s]: y for s, y in adopters.items()}
    ctrl = {FIPS[s] for s in never}
    on = sy.copy(); on["st"] = on["State"].map(FIPS)
    on = on[["st", "Base Year", "on"]].rename(columns={"Base Year": "year"})

    b = load_bds(set(ex.index))
    b = b[b["st"].isin(set(adopt_fips) | ctrl)]
    d = b.merge(on, on=["st", "year"], how="left").join(ex, on="sector")
    d["adopt"] = d["st"].map(adopt_fips)
    d["k"] = np.where(d["adopt"].notna(), d["year"] - d["adopt"], -999)
    d["post"] = d["on"].fillna(0).astype(float)
    dv = d[(d.year >= 1990) & (d.year <= 2015)]
    trt = dv["k"] > -900
    dd = dv[(~trt) | dv["k"].between(BAL_LO, BAL_HI)].copy()
    for nm in ["pdit", "bea", "equip"]:
        dd["post_" + nm] = dd["post"] * dd[nm + "_exp"]

    # ---------- 1 ----------
    print("=" * 78)
    print("1. RANDOMISATION WITH %d PERMUTATIONS" % NPERM)
    print("=" * 78)
    print("script 43 used 500, giving a Monte Carlo SE of 0.0097 around")
    print("p=0.05, so its 95%% interval for the true p was [0.031, 0.069].")
    print("At %d draws the SE falls to about %.4f.\n"
          % (NPERM, np.sqrt(0.05 * 0.95 / NPERM)))

    secs = list(ex.index)
    vals = ex["equip_exp"].to_numpy(float)
    results = {}
    for yn in ["jrr", "entry_rate", "jc_rate", "jd_rate"]:
        sub = dd.dropna(subset=[yn]).copy()
        sub = sub[np.isfinite(sub[yn].to_numpy(float))]
        cl = codes_for(sub, FE_GRAD)
        yv = absorb(sub[yn].to_numpy(float), cl)
        postv = sub["post"].to_numpy(float)
        secv = sub["sector"].to_numpy()
        b_real = point_est(sub, yn, "post_equip", cl)

        draws = np.empty(NPERM)
        for i in range(NPERM):
            perm = dict(zip(secs, rng.permutation(vals)))
            xx = postv * np.array([perm[s] for s in secv])
            xx = absorb(xx, cl)
            if xx.std() < 1e-12:
                draws[i] = np.nan
                continue
            bb, _ = ols(xx[:, None], yv)
            draws[i] = bb[0]
        draws = draws[np.isfinite(draws)]
        pv = (np.abs(draws) >= abs(b_real)).mean()
        mcse = np.sqrt(pv * (1 - pv) / len(draws))
        results[yn] = (b_real, pv, mcse)
        print("%-12s real=%+.4f  placebo sd=%.4f  p=%.4f (MC SE %.4f, CI [%.3f, %.3f])"
              % (yn, b_real, draws.std(), pv, mcse,
                 max(pv - 1.96 * mcse, 0), pv + 1.96 * mcse))

    # ---------- 2 ----------
    print("\n" + "=" * 78)
    print("2. IS IT ONE SECTOR? LEAVE-ONE-SECTOR-OUT")
    print("=" * 78)
    print("permutation cannot tell whether equipment intensity is the")
    print("operative characteristic or whether one sector drives everything.")
    print("this can. jrr, equipment gradient.\n")
    full = point_est(dd, "jrr", "post_equip")
    print("  full sample beta = %+.4f\n" % full)
    rows = []
    for s in secs:
        sub = dd[dd.sector != s]
        bb = point_est(sub, "jrr", "post_equip")
        rows.append({"dropped": s, "beta": bb,
                     "pct": 100 * (bb - full) / abs(full),
                     "equip": ex.loc[s, "equip_exp"]})
    o = pd.DataFrame(rows).sort_values("beta")
    print("%-8s %8s %10s %10s" % ("dropped", "equip", "beta", "change"))
    for _, r in o.iterrows():
        flag = "  <-" if abs(r["pct"]) > 30 else ""
        print("%-8s %8.3f %10.4f %9.0f%%%s"
              % (r["dropped"], r["equip"], r["beta"], r["pct"], flag))
    print("\n  range %+.4f to %+.4f, sign flips %d of %d"
          % (o.beta.min(), o.beta.max(),
             (np.sign(o.beta) != np.sign(full)).sum(), len(o)))

    # ---------- 3 ----------
    print("\n" + "=" * 78)
    print("3. WHICH SIDE OF JRR MOVES?")
    print("=" * 78)
    print("jrr = job creation + job destruction, over the DHS denominator.")
    print("A credit that raised entry would show up in creation. Destruction")
    print("moving instead would be harder to attribute to the policy.\n")
    for yn in ["jrr", "jc_rate", "jd_rate"]:
        sub = dd.dropna(subset=[yn]).copy()
        sub = sub[np.isfinite(sub[yn].to_numpy(float))]
        cl = codes_for(sub, FE_GRAD)
        y = absorb(sub[yn].to_numpy(float), cl)
        X = absorb(sub[["post_equip"]].to_numpy(float), cl)
        bj, pj, lo, hi = wcb(X, y, sub["st"].to_numpy(), 0, rng)
        mean = sub[yn].mean()
        print("  %-10s (mean %5.2f)  beta=%+.4f  WCB p=%.3f  CI [%+.4f,%+.4f]  (%+.1f%%,%+.1f%%)"
              % (yn, mean, bj, pj, lo, hi, 100 * lo / mean, 100 * hi / mean))

    # ---------- 4 ----------
    print("\n" + "=" * 78)
    print("4. VERDICT")
    print("=" * 78)
    bj, pv, mcse = results["jrr"]
    print("  randomisation p for the jrr equipment gradient: %.4f" % pv)
    if pv > 0.10:
        print("  -> inside the placebo distribution. Report as noise.")
    elif pv > 0.05:
        print("  -> borderline. Report the coefficient, do not interpret it.")
    else:
        print("  -> outside the placebo range. Section 2 decides whether that")
        print("     is equipment intensity or one sector.")
    print("\n  whatever the number, the randomisation test cannot attribute")
    print("  the effect to equipment intensity specifically. Leave-one-sector-out")
    print("  in section 2 is the check that can.")

    os.makedirs(OUTDIR, exist_ok=True)
    o.to_csv(os.path.join(OUTDIR, "44_loso_equip.csv"), index=False)
    print("\nwrote output\\44_loso_equip.csv")


if __name__ == "__main__":
    main()