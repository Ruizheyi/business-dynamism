# 50_final_robustness.py
# Cleans up the loose ends found in self-audit, then runs the two checks that
# matter for how the headline is written.
#
# WHAT THE AUDIT FOUND. The window called "balanced" in scripts 41-49 is not
# balanced. The sample runs 1990-2015 but MA and IL adopt in 1991, so k=-5 for
# them is 1986, outside the sample. Coverage by event time:
#     k=-5  14 of 20 states   (missing MA IL AZ CT NJ WA)
#     k=-4  16 of 20
#     k=-3  17 of 20
#     k=-2  18 of 20
#     k=-1 to +3  20 of 20
#     k=+4  19 of 20          (missing FL)
# So leads are estimated off fewer states than lags, and the six states absent
# from k=-5 are exactly the earliest adopters. The post-vs-pre estimate is a
# weighted average whose weights come from calendar position: MA contributes
# one pre year and five post years, FL contributes five pre and four post.
# It should be called a truncated window, not a balanced one.
#
# FIXES APPLIED HERE
#   - treatment is actual credit status everywhere, never 1{k>=0}
#     (scripts 47 and 48 reintroduced the absorbing-treatment error)
#   - the "share of swing before k=0" statistic from script 47 is dropped;
#     it was invented, and it explodes when the pre mean is near zero
#   - window terminology corrected throughout
#
# TWO NEW CHECKS
#   Section 2: genuinely balanced windows. k in [-3,2] keeps 18 of 20 states
#     complete; k in [-2,1] keeps 20 of 20. If the estimate holds as the
#     window tightens toward full balance, the imbalance was not driving it.
#   Section 3: adoption cohorts. Early adopters (1991-1998) versus late
#     (2000-2012). Goodman-Bacon decomposition problems bite hardest when
#     effects differ across cohorts; if the two agree, TWFE bias is less of a
#     worry here.
#
# Run: py scripts\50_final_robustness.py

import math
import os
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
PDIT = os.path.join(RAW, "export.csv")
BDS = os.path.join(RAW, "bds2023_st_sec.csv")
OUTDIR = "output"

RATIO = "Discounted - 12% Rate"
NBOOT = 999
SEED = 20260805

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
        return beta[j], np.nan, np.nan, np.nan, np.nan
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
        return beta[j], se[j], np.nan, np.nan, np.nan
    pv = (np.abs(ts) >= abs(t0)).mean()
    lo_t, hi_t = np.quantile(ts, [0.025, 0.975])
    return beta[j], se[j], pv, beta[j] - hi_t * se[j], beta[j] - lo_t * se[j]


def chi2_p(stat, df):
    if not np.isfinite(stat) or df < 1:
        return np.nan
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / np.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / np.sqrt(2))


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
    b["exit_rate"] = b["estabs_exit"] / b["den"] * 100
    b["jc_rate"] = b["job_creation"] / b["denom"] * 100
    return b[b["sector"].isin(keep)]


FE = [("st", "sector"), ("sector", "year")]


def est(d, yname, rng, lo, hi, states=None):
    """post-vs-pre on a chosen event window, treatment = actual status"""
    s = d.copy()
    if states is not None:
        s = s[s["st"].isin(states) | (s["k"] <= -900)]
    trt = s["k"] > -900
    s = s[(~trt) | s["k"].between(lo, hi)]
    s = s.dropna(subset=[yname])
    s = s[np.isfinite(s[yname].to_numpy(float))]
    if len(s) < 300 or s["post_on"].std() < 1e-12:
        return None
    cl = codes_for(s, FE)
    y = absorb(s[yname].to_numpy(float), cl)
    X = absorb(s[["post_on"]].to_numpy(float), cl)
    bj, sj, pj, clo, chi = wcb(X, y, s["st"].to_numpy(), 0, rng)
    return {"n": len(s), "states": s["st"].nunique(), "beta": bj, "p": pj,
            "lo": clo, "hi": chi, "mean": s[yname].mean()}


def show(tag, r):
    if r is None:
        print("  %-40s (not estimable)" % tag); return
    star = " *" if np.isfinite(r["p"]) and r["p"] < 0.05 else ""
    print("  %-40s N=%5d beta=%+.4f p=%.3f CI [%+.4f,%+.4f]%s"
          % (tag, r["n"], r["beta"], r["p"], r["lo"], r["hi"], star))


def main():
    rng = np.random.default_rng(SEED)
    p = load_pdit()
    rd = "Research and Development Credit"
    sy = p.groupby(["State", "Base Year"])[rd].max().reset_index()
    sy["on"] = (sy[rd] > 0).astype(int)
    first = sy[sy.on == 1].groupby("State")["Base Year"].min()
    adopters = first[first > 1990]
    never = sorted(set(p["State"]) - set(first.index))
    adopt_fips = {FIPS[s]: y for s, y in adopters.items()}
    ctrl = {FIPS[s] for s in never}
    on = sy.copy(); on["st"] = on["State"].map(FIPS)
    on = on[["st", "Base Year", "on"]].rename(
        columns={"Base Year": "year", "on": "on_status"})

    b = load_bds(set(p["sector"].unique()))
    b = b[b["st"].isin(set(adopt_fips) | ctrl)]
    d = b.merge(on, on=["st", "year"], how="left")
    d["on_status"] = d["on_status"].fillna(0).astype(int)
    d["adopt"] = d["st"].map(adopt_fips)
    d["k"] = np.where(d["adopt"].notna(), d["year"] - d["adopt"], -999)
    d["post_on"] = d["on_status"].astype(float)
    dv = d[(d.year >= 1990) & (d.year <= 2015)].copy()

    # ---------- 1 ----------
    print("=" * 78)
    print("1. HOW UNBALANCED IS THE WINDOW USED SO FAR?")
    print("=" * 78)
    print("sample 1990-2015. A state contributes event year k only if")
    print("adopt+k falls inside the sample.\n")
    print("  %3s  %10s  %s" % ("k", "states", "missing"))
    for k in range(-5, 5):
        miss = [s for s, a in adopters.items() if not (1990 <= a + k <= 2015)]
        print("  %+3d  %2d of %d    %s"
              % (k, len(adopters) - len(miss), len(adopters),
                 ",".join(miss) if miss else "-"))
    print("\n  the states absent at k=-5 are the earliest adopters, so leads")
    print("  and lags are estimated off different sets of states. This window")
    print("  is truncated, not balanced.")

    # ---------- 2 ----------
    print("\n" + "=" * 78)
    print("2. TIGHTENING THE WINDOW TOWARD FULL BALANCE")
    print("=" * 78)
    print("entry_rate. As the window narrows, coverage becomes complete but")
    print("the sample shrinks. If the estimate holds, imbalance was not")
    print("driving it.\n")
    for lo, hi in [(-5, 4), (-4, 3), (-3, 2), (-2, 1), (-1, 1)]:
        miss = set()
        for k in range(lo, hi + 1):
            miss |= {s for s, a in adopters.items() if not (1990 <= a + k <= 2015)}
        r = est(dv, "entry_rate", rng, lo, hi)
        tag = "k in [%+d,%+d]  %d/%d states complete" % (
            lo, hi, len(adopters) - len(miss), len(adopters))
        show(tag, r)
    print("\n  k in [-2,+1] is the first fully balanced window.")

    print("\n  same for the other outcomes, k in [-3,+2]:")
    for yn in ["exit_rate", "jc_rate"]:
        show(yn, est(dv, yn, rng, -3, 2))

    # ---------- 3 ----------
    print("\n" + "=" * 78)
    print("3. ADOPTION COHORTS")
    print("=" * 78)
    print("TWFE with staggered timing is most misleading when effects differ")
    print("across cohorts. Early adopters also spend more of the sample")
    print("treated, so they carry more weight in the pooled estimate.\n")
    early = {FIPS[s]: a for s, a in adopters.items() if a <= 1998}
    late = {FIPS[s]: a for s, a in adopters.items() if a >= 2000}
    print("  early cohort (1991-1998): %d states  %s"
          % (len(early), ",".join(s for s, a in adopters.items() if a <= 1998)))
    print("  late  cohort (2000-2012): %d states  %s\n"
          % (len(late), ",".join(s for s, a in adopters.items() if a >= 2000)))
    for lo, hi in [(-5, 4), (-3, 2)]:
        print("  window k in [%+d,%+d]:" % (lo, hi))
        show("    all adopters", est(dv, "entry_rate", rng, lo, hi))
        show("    early only", est(dv, "entry_rate", rng, lo, hi,
                                   states=set(early) | ctrl))
        show("    late only", est(dv, "entry_rate", rng, lo, hi,
                                  states=set(late) | ctrl))
        print()

    # ---------- 4 ----------
    print("=" * 78)
    print("4. EVENT STUDY ON THE FULLY BALANCED WINDOW")
    print("=" * 78)
    print("k in [-3,+2], 18 of 20 states complete, treatment = actual status,")
    print("reference k=-1.\n")
    s = dv.copy()
    trt = s["k"] > -900
    s = s[(~trt) | s["k"].between(-3, 2)]
    s = s.dropna(subset=["entry_rate"])
    s = s[np.isfinite(s["entry_rate"].to_numpy(float))]
    ks = [k for k in range(-3, 3) if k != -1]
    cols, kept = [], []
    for k in ks:
        ind = (s["k"] == k) & (s["k"] > -900)
        if k >= 0:
            ind = ind & (s["on_status"] == 1)
        v = ind.to_numpy(float)
        if v.std() > 1e-12:
            cols.append(v); kept.append(k)
    X = np.column_stack(cols)
    cl = codes_for(s, FE)
    y = absorb(s["entry_rate"].to_numpy(float), cl)
    X = absorb(X, cl)
    beta, inv = ols(X, y)
    V = cluster_vcv(X, y - X @ beta, inv, s["st"].to_numpy())
    se = np.sqrt(np.diag(V))
    print("%5s %10s %10s %8s" % ("k", "beta", "se", "t"))
    for k, bb, ss in zip(kept, beta, se):
        print("%5d %10.4f %10.4f %8.2f%s"
              % (k, bb, ss, bb / ss if ss > 0 else np.nan,
                 "  <- pre" if k < -1 else ""))
    pre = [i for i, k in enumerate(kept) if k < -1]
    post = [i for i, k in enumerate(kept) if k >= 0]
    if len(pre) >= 2:
        bp = beta[pre]; Vp = V[np.ix_(pre, pre)]
        try:
            stat = float(bp @ np.linalg.solve(Vp, bp))
            print("\n  lead Wald chi2(%d) = %.2f, p = %.3f"
                  % (len(pre), stat, chi2_p(stat, len(pre))))
        except np.linalg.LinAlgError:
            pass
    w = np.ones(len(post)) / len(post)
    eff = float(w @ beta[post])
    seff = float(np.sqrt(w @ V[np.ix_(post, post)] @ w))
    print("  mean post effect = %+.4f (se %.4f)  CI [%+.4f,%+.4f]"
          % (eff, seff, eff - 1.96 * seff, eff + 1.96 * seff))
    if pre:
        print("  mean lead        = %+.4f" % beta[pre].mean())
        print("  net of lead mean = %+.4f" % (eff - beta[pre].mean()))

    os.makedirs(OUTDIR, exist_ok=True)


if __name__ == "__main__":
    main()