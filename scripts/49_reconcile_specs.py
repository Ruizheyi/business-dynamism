# 49_reconcile_specs.py
# TWO SPECIFICATIONS ON THE SAME DATA DISAGREE BY A FACTOR OF 3.3.
#
#   post-vs-pre  (script 41/47)  +0.3005  WCB p=0.020  CI [+0.057, +0.555]
#   event study  (script 47)     +0.0906  se 0.1245    CI [-0.154, +0.335]
#
# Same outcome, same panel, same fixed effects. Before concluding anything
# about the headline result, that gap has to be explained.
#
# THE BUG. Scripts 47 and 48 defined treatment in the event study as
# 1{k >= 0} & treated -- that is, "the k-th year after adoption". But script
# 41 established that treatment is NOT absorbing: MO had no credit for 11 of
# its 15 post years, TX for 6, MI for 4, WA for 1. The post-vs-pre spec uses
# actual on/off status. The event study reintroduced exactly the coding error
# script 41 fixed. That is my error, and it invalidates the claim in the last
# session that the headline result fails its lead test.
#
# THREE DIFFERENCES, opened one at a time:
#   A. treatment definition: actual on/off vs 1{k>=0}
#   B. sample window: balanced k in [-5,4] vs full 1990-2015
#   C. reference point: mean of all pre periods vs the single year k=-1
#      (arithmetic already says C can account for at most 0.168 of 0.3005)
#
# Section 4 then reruns the lead test with treatment defined correctly.
#
# Run: py scripts\49_reconcile_specs.py

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
BAL_LO, BAL_HI = -5, 4
KLO, KHI = -5, 7

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


def simple(sub, yname, postcol, rng):
    s = sub.dropna(subset=[yname, postcol]).copy()
    s = s[np.isfinite(s[yname].to_numpy(float))]
    cl = codes_for(s, FE)
    y = absorb(s[yname].to_numpy(float), cl)
    X = absorb(s[[postcol]].to_numpy(float), cl)
    bj, pj, lo, hi = wcb(X, y, s["st"].to_numpy(), 0, rng)
    return {"n": len(s), "beta": bj, "p": pj, "lo": lo, "hi": hi}


def show(tag, r):
    if r is None:
        print("  %-46s (n/a)" % tag); return
    star = " *" if np.isfinite(r["p"]) and r["p"] < 0.05 else ""
    print("  %-46s N=%5d beta=%+.4f p=%.3f CI [%+.4f,%+.4f]%s"
          % (tag, r["n"], r["beta"], r["p"], r["lo"], r["hi"], star))


def event_fit(sub, yname, postmode, ref=-1):
    """postmode 'on' uses actual credit status inside each event year,
    'k' uses 1{k>=0} which is what scripts 47/48 wrongly did"""
    trt = sub["k"] > -900
    ks = [k for k in range(KLO, KHI + 1) if k != ref]
    cols, kept = [], []
    for k in ks:
        ind = (sub["k"] == k) & trt
        if postmode == "on" and k >= 0:
            ind = ind & (sub["on_status"] == 1)
        v = ind.to_numpy(float)
        if v.std() > 1e-12:
            cols.append(v); kept.append(k)
    X = np.column_stack(cols)
    cl = codes_for(sub, FE)
    y = absorb(sub[yname].to_numpy(float), cl)
    X = absorb(X, cl)
    beta, inv = ols(X, y)
    V = cluster_vcv(X, y - X @ beta, inv, sub["st"].to_numpy())
    return np.array(kept), beta, V


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
    d["post_on"] = d["on_status"].astype(float)           # correct
    d["post_k"] = ((d["k"] >= 0) & (d["k"] > -900)).astype(float)   # buggy
    dv = d[(d.year >= 1990) & (d.year <= 2015)].copy()
    trt = dv["k"] > -900
    dbal = dv[(~trt) | dv["k"].between(BAL_LO, BAL_HI)].copy()

    print("=" * 78)
    print("0. HOW MUCH DO THE TWO TREATMENT DEFINITIONS DIFFER?")
    print("=" * 78)
    tt = dv[trt]
    dis = (tt["post_on"] != tt["post_k"]).mean()
    print("  treated-state cells where the two disagree: %.3f" % dis)
    print("  cells coded post by event time but with no credit in force: %d"
          % ((tt["post_k"] == 1) & (tt["post_on"] == 0)).sum())
    print("\n  by state:")
    for st, gg in tt.groupby("st"):
        n = ((gg["post_k"] == 1) & (gg["post_on"] == 0)).sum()
        if n:
            nm = [k for k, v in FIPS.items() if v == st][0]
            print("    %s: %d cell-years mis-coded by the event-time definition"
                  % (nm, n))

    print("\n" + "=" * 78)
    print("1. OPENING THE THREE DIFFERENCES ONE AT A TIME")
    print("=" * 78)
    print("outcome: entry_rate. Start from the headline spec and change one")
    print("thing at a time until it becomes the event-study sample.\n")
    show("A balanced window, post = actual on/off  [HEADLINE]",
         simple(dbal, "entry_rate", "post_on", rng))
    show("B balanced window, post = 1{k>=0}  [the bug]",
         simple(dbal, "entry_rate", "post_k", rng))
    show("C full window,     post = actual on/off",
         simple(dv, "entry_rate", "post_on", rng))
    show("D full window,     post = 1{k>=0}  [the bug]",
         simple(dv, "entry_rate", "post_k", rng))
    print("\n  A vs B isolates the treatment definition.")
    print("  A vs C isolates the sample window.")
    print("  D is what scripts 47 and 48 were implicitly estimating.")

    print("\n" + "=" * 78)
    print("2. EVENT STUDY, BOTH TREATMENT DEFINITIONS")
    print("=" * 78)
    for mode, lab in [("k", "1{k>=0}  [the bug]"),
                      ("on", "actual on/off  [correct]")]:
        sub = dv.dropna(subset=["entry_rate"]).copy()
        sub = sub[np.isfinite(sub["entry_rate"].to_numpy(float))]
        kept, beta, V = event_fit(sub, "entry_rate", mode)
        se = np.sqrt(np.diag(V))
        print("\n--- treatment = %s ---" % lab)
        print("%5s %10s %10s %8s" % ("k", "beta", "se", "t"))
        for k, bb, ss in zip(kept, beta, se):
            print("%5d %10.4f %10.4f %8.2f%s"
                  % (k, bb, ss, bb / ss if ss > 0 else np.nan,
                     "  <- pre" if k < -1 else ""))
        pre = [i for i, k in enumerate(kept) if k < -1]
        post = [i for i, k in enumerate(kept) if 0 <= k <= 4]
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
        print("  mean k=0..4 effect = %+.4f (se %.4f), CI [%+.4f, %+.4f]"
              % (eff, seff, eff - 1.96 * seff, eff + 1.96 * seff))
        print("  mean lead          = %+.4f" % beta[pre].mean())
        print("  effect net of lead mean = %+.4f" % (eff - beta[pre].mean()))

    print("\n" + "=" * 78)
    print("3. EVENT STUDY ON THE BALANCED WINDOW, CORRECT TREATMENT")
    print("=" * 78)
    print("matches the headline sample exactly, so the only remaining")
    print("difference is the reference point.\n")
    sub = dbal.dropna(subset=["entry_rate"]).copy()
    sub = sub[np.isfinite(sub["entry_rate"].to_numpy(float))]
    kept, beta, V = event_fit(sub, "entry_rate", "on")
    se = np.sqrt(np.diag(V))
    print("%5s %10s %10s %8s" % ("k", "beta", "se", "t"))
    for k, bb, ss in zip(kept, beta, se):
        print("%5d %10.4f %10.4f %8.2f%s"
              % (k, bb, ss, bb / ss if ss > 0 else np.nan,
                 "  <- pre" if k < -1 else ""))
    pre = [i for i, k in enumerate(kept) if k < -1]
    post = [i for i, k in enumerate(kept) if k >= 0]
    if pre:
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
    print("  mean post effect        = %+.4f (se %.4f)" % (eff, seff))
    print("  mean lead               = %+.4f" % (beta[pre].mean() if pre else 0))
    print("  effect net of lead mean = %+.4f"
          % (eff - (beta[pre].mean() if pre else 0)))
    print("  headline post-vs-pre    = +0.3005")
    print("\n  if 'effect net of lead mean' now lands near +0.30, the three")
    print("  differences are fully accounted for and the headline stands.")

    print("\n" + "=" * 78)
    print("4. THE OTHER OUTCOMES, CORRECT TREATMENT, BALANCED WINDOW")
    print("=" * 78)
    for yn in ["exit_rate", "jc_rate"]:
        show(yn, simple(dbal, yn, "post_on", rng))

    os.makedirs(OUTDIR, exist_ok=True)


if __name__ == "__main__":
    main()