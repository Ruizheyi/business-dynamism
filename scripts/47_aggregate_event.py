# 47_aggregate_event.py
# The equipment gradient just failed. Post-vs-pre gave +1.70 with WCB p=0.012,
# it passed randomisation (p=0.009), leave-one-sector-out (0 flips in 16), and
# fake adoption timing (p=0.041). Then the dynamic specification showed the
# gradient was already at -1.59 three years before adoption (t=-2.65), joint
# Wald on the leads chi2(3)=10.97 against a 5% critical value of 7.81. More
# than half the movement happened before treatment.
#
# The lesson is that none of those four tests can see a pre-trend, because
# they all operate on the same post-minus-pre mean difference.
#
# THE HEADLINE RESULT IS ALSO A POST-MINUS-PRE MEAN DIFFERENCE.
# Aggregate effect on entry, balanced window: +0.301, WCB p=0.018. Its leads
# have never been inspected. Script 42 ran leads for the two exposure
# GRADIENTS, not for the aggregate. This script does that, plus the same
# checks the equipment gradient passed, so the two can be compared directly.
#
# Run: py scripts\47_aggregate_event.py

import os
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
PDIT = os.path.join(RAW, "export.csv")
BDS = os.path.join(RAW, "bds2023_st_sec.csv")
OUTDIR = "output"

RATIO = "Discounted - 12% Rate"
NBOOT = 999
NFAKE = 1000
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
    """Wilson-Hilferty, adequate for reporting"""
    import math
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


def build(p):
    rd = "Research and Development Credit"
    sy = p.groupby(["State", "Base Year"])[rd].max().reset_index()
    sy["on"] = (sy[rd] > 0).astype(int)
    first = sy[sy.on == 1].groupby("State")["Base Year"].min()
    sectors = sorted(p["sector"].unique())
    return sectors, first[first > 1990], sorted(set(p["State"]) - set(first.index)), sy


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
    b["jrr"] = (b["job_creation"] + b["job_destruction"]) / b["denom"] * 100
    return b[b["sector"].isin(keep)]


# aggregate spec: no state-year FE, or post would be absorbed
FE_AGG = [("st", "sector"), ("sector", "year")]


def event_study(sub, yname, fes, kcol="k", trtcol=None):
    trt = sub[kcol] > -900 if trtcol is None else sub[trtcol]
    ks = [k for k in range(KLO, KHI + 1) if k != -1]
    cols, kept = [], []
    for k in ks:
        v = ((sub[kcol] == k) & trt).to_numpy(float)
        if v.std() > 1e-12:
            cols.append(v); kept.append(k)
    if not cols:
        return None
    X = np.column_stack(cols)
    cl = codes_for(sub, fes)
    y = absorb(sub[yname].to_numpy(float), cl)
    X = absorb(X, cl)
    beta, inv = ols(X, y)
    V = cluster_vcv(X, y - X @ beta, inv, sub["st"].to_numpy())
    return kept, beta, np.sqrt(np.diag(V)), V


def main():
    rng = np.random.default_rng(SEED)
    p = load_pdit()
    sectors, adopters, never, sy = build(p)
    adopt_fips = {FIPS[s]: y for s, y in adopters.items()}
    ctrl = {FIPS[s] for s in never}
    on = sy.copy(); on["st"] = on["State"].map(FIPS)
    on = on[["st", "Base Year", "on"]].rename(columns={"Base Year": "year"})

    b = load_bds(set(sectors))
    b = b[b["st"].isin(set(adopt_fips) | ctrl)]
    d = b.merge(on, on=["st", "year"], how="left")
    d["adopt"] = d["st"].map(adopt_fips)
    d["k"] = np.where(d["adopt"].notna(), d["year"] - d["adopt"], -999)
    d["post"] = d["on"].fillna(0).astype(float)
    dv = d[(d.year >= 1990) & (d.year <= 2015)].copy()
    trt = dv["k"] > -900
    dd = dv[(~trt) | dv["k"].between(BAL_LO, BAL_HI)].copy()

    print("=" * 78)
    print("1. THE HEADLINE ESTIMATE, RESTATED")
    print("=" * 78)
    print("aggregate effect, balanced window, treatment = actual on/off.\n")
    for yn in ["entry_rate", "exit_rate", "jc_rate", "jrr"]:
        sub = dd.dropna(subset=[yn]).copy()
        sub = sub[np.isfinite(sub[yn].to_numpy(float))]
        cl = codes_for(sub, FE_AGG)
        y = absorb(sub[yn].to_numpy(float), cl)
        X = absorb(sub[["post"]].to_numpy(float), cl)
        bj, pj, lo, hi = wcb(X, y, sub["st"].to_numpy(), 0, rng)
        m = sub[yn].mean()
        print("  %-12s (mean %5.2f)  beta=%+.4f  WCB p=%.3f  CI [%+.4f,%+.4f]  (%+.1f%%,%+.1f%%)"
              % (yn, m, bj, pj, lo, hi, 100 * lo / m, 100 * hi / m))

    print("\n" + "=" * 78)
    print("2. THE CHECK THAT KILLED THE EQUIPMENT GRADIENT")
    print("=" * 78)
    print("leads have never been inspected for the aggregate. Reference k=-1.")
    print("5%% critical values: chi2(3)=7.81, chi2(4)=9.49.\n")
    for yn in ["entry_rate", "jc_rate", "exit_rate"]:
        sub = dv.dropna(subset=[yn]).copy()
        sub = sub[np.isfinite(sub[yn].to_numpy(float))]
        res = event_study(sub, yn, FE_AGG)
        if res is None:
            continue
        kept, beta, se, V = res
        print("--- %s ---" % yn)
        print("%5s %10s %10s %8s" % ("k", "beta", "se", "t"))
        for k, bb, ss in zip(kept, beta, se):
            t = bb / ss if ss > 0 else np.nan
            bar = "#" * min(int(abs(bb) / 0.08), 26)
            print("%5d %10.4f %10.4f %8.2f  %s%s"
                  % (k, bb, ss, t, bar, "  <- pre" if k < -1 else ""))
        pre = [i for i, k in enumerate(kept) if k < -1]
        if pre:
            bp = beta[pre]; Vp = V[np.ix_(pre, pre)]
            try:
                stat = float(bp @ np.linalg.solve(Vp, bp))
                print("\n  joint Wald on %d leads: chi2 = %.2f, p = %.3f  %s"
                      % (len(pre), stat, chi2_p(stat, len(pre)),
                         "<- REJECTS parallel trends" if chi2_p(stat, len(pre)) < 0.05
                         else "<- does not reject"))
            except np.linalg.LinAlgError:
                print("\n  Wald not computable")
        post = [i for i, k in enumerate(kept) if k >= 0]
        if post:
            print("  mean post coefficient: %+.4f" % beta[post].mean())
            print("  mean pre  coefficient: %+.4f" % beta[pre].mean())
            print("  share of the pre-to-post swing occurring before k=0: %.0f%%"
                  % (100 * abs(beta[pre].mean()) /
                     max(abs(beta[post].mean() - beta[pre].mean()), 1e-9))
                  if len(pre) else "")
        print()

    print("=" * 78)
    print("3. FAKE ADOPTION TIMING, AGGREGATE")
    print("=" * 78)
    print("same test the equipment gradient passed. %d draws.\n" % NFAKE)
    real_years = sorted(adopters.values)
    all_states = sorted(set(dv["st"].unique()))
    for yn in ["entry_rate", "jc_rate"]:
        sub = dd.dropna(subset=[yn]).copy()
        sub = sub[np.isfinite(sub[yn].to_numpy(float))]
        cl = codes_for(sub, FE_AGG)
        y = absorb(sub[yn].to_numpy(float), cl)
        X = absorb(sub[["post"]].to_numpy(float), cl)
        breal, _ = ols(X, y)
        breal = breal[0]

        base = dv.dropna(subset=[yn]).copy()
        base = base[np.isfinite(base[yn].to_numpy(float))]
        yrv = base["year"].to_numpy(); stv = base["st"].to_numpy()
        draws = []
        for _ in range(NFAKE):
            fake = {s: rng.choice(real_years) for s in all_states}
            kk = np.array([yrv[i] - fake[stv[i]] for i in range(len(base))])
            keep = (kk >= BAL_LO) & (kk <= BAL_HI)
            if keep.sum() < 1000:
                continue
            s2 = base[keep]
            postf = (kk[keep] >= 0).astype(float)
            cl2 = codes_for(s2, FE_AGG)
            y2 = absorb(s2[yn].to_numpy(float), cl2)
            x2 = absorb(postf, cl2)
            if x2.std() < 1e-12:
                continue
            bb, _ = ols(x2[:, None], y2)
            draws.append(bb[0])
        draws = np.array(draws)
        pv = (np.abs(draws) >= abs(breal)).mean()
        print("%s:" % yn)
        print("  real  = %+.4f | fake mean %+.4f, sd %.4f | p = %.4f (n=%d)"
              % (breal, draws.mean(), draws.std(), pv, len(draws)))

    print("\n" + "=" * 78)
    print("4. HOW TO READ THIS")
    print("=" * 78)
    print("  leads flat, Wald does not reject -> the aggregate survives the")
    print("     check that killed the equipment gradient, and is the paper's")
    print("     one estimate that has passed a dynamic specification.")
    print("  leads trending, Wald rejects -> the aggregate is also a pre-trend")
    print("     and the paper has no causal estimate at all. That is a")
    print("     legitimate result but it changes what the paper says.")

    os.makedirs(OUTDIR, exist_ok=True)


if __name__ == "__main__":
    main()