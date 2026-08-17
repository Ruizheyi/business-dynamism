# 48_pretrend_deep.py
# The lead test in script 47 has three weaknesses.
#
# 1. It hangs on a single omitted year, k=-1. Script 37 already showed how
#    much that choice can matter: a gradient p-value moved from 0.057 to
#    0.586 depending on which year was dropped. The lead test inherits that
#    fragility.
#
# 2. The "share of the swing occurring before k=0" statistic in script 47 was
#    something I made up: |mean pre| / |mean post - mean pre|. It blows up
#    when the pre mean is near zero, which is why exit rate reported 95%
#    despite having no effect at all. It should not be used and is dropped
#    here.
#
# 3. A joint Wald that fails to reject says there is not enough power to
#    detect a pre-trend. It does not say the pre-trend is absent. With 20
#    adopting states that distinction matters.
#
# This script does three things instead.
#   Section 1: the lead Wald across four reference choices.
#   Section 2: a linear pre-trend test. The worry is not that leads are noisy,
#              it is that they slope. Regressing lead coefficients on event
#              time by GLS answers that directly.
#   Section 3: Rambachan-Roth style sensitivity. Rather than assuming parallel
#              trends holds, ask how large a post-treatment violation the
#              estimate can survive, expressed as a multiple M of the largest
#              pre-treatment violation observed. M=1 means "post-treatment
#              deviation no worse than what is visible pre-treatment".
#
# Run: py scripts\48_pretrend_deep.py

import math
import os
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
PDIT = os.path.join(RAW, "export.csv")
BDS = os.path.join(RAW, "bds2023_st_sec.csv")
OUTDIR = "output"

RATIO = "Discounted - 12% Rate"
SEED = 20260805
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


def chi2_p(stat, df):
    if not np.isfinite(stat) or df < 1:
        return np.nan
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / np.sqrt(2 / (9 * df))
    return 0.5 * math.erfc(z / np.sqrt(2))


def norm_p(z):
    return math.erfc(abs(z) / np.sqrt(2))


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


FE_AGG = [("st", "sector"), ("sector", "year")]


def event_fit(sub, yname, ref):
    """returns kept event times, coefficients, covariance"""
    trt = sub["k"] > -900
    ks = [k for k in range(KLO, KHI + 1) if k != ref]
    cols, kept = [], []
    for k in ks:
        v = ((sub["k"] == k) & trt).to_numpy(float)
        if v.std() > 1e-12:
            cols.append(v); kept.append(k)
    X = np.column_stack(cols)
    cl = codes_for(sub, FE_AGG)
    y = absorb(sub[yname].to_numpy(float), cl)
    X = absorb(X, cl)
    beta, inv = ols(X, y)
    V = cluster_vcv(X, y - X @ beta, inv, sub["st"].to_numpy())
    return np.array(kept), beta, V


def main():
    p = load_pdit()
    rd = "Research and Development Credit"
    sy = p.groupby(["State", "Base Year"])[rd].max().reset_index()
    sy["on"] = (sy[rd] > 0).astype(int)
    first = sy[sy.on == 1].groupby("State")["Base Year"].min()
    adopters = first[first > 1990]
    never = sorted(set(p["State"]) - set(first.index))
    adopt_fips = {FIPS[s]: y for s, y in adopters.items()}
    ctrl = {FIPS[s] for s in never}

    b = load_bds(set(p["sector"].unique()))
    b = b[b["st"].isin(set(adopt_fips) | ctrl)]
    b["adopt"] = b["st"].map(adopt_fips)
    b["k"] = np.where(b["adopt"].notna(), b["year"] - b["adopt"], -999)
    dv = b[(b.year >= 1990) & (b.year <= 2015)].copy()

    for yname in ["entry_rate", "jc_rate", "exit_rate"]:
        sub = dv.dropna(subset=[yname]).copy()
        sub = sub[np.isfinite(sub[yname].to_numpy(float))]

        print("=" * 78)
        print("OUTCOME: %s   (mean %.2f)" % (yname, sub[yname].mean()))
        print("=" * 78)

        # ---- 1. reference period sensitivity ----
        print("\n1. LEAD WALD ACROSS REFERENCE CHOICES")
        print("   script 37 showed a single omitted year can move a p-value")
        print("   from 0.057 to 0.586. The lead test inherits that fragility.\n")
        print("   %-6s %8s %6s %8s   %s" % ("ref", "chi2", "df", "p", "leads"))
        for ref in [-1, -2, -3, -4]:
            kept, beta, V = event_fit(sub, yname, ref)
            pre = np.where(kept < ref)[0] if ref > KLO else np.where(kept < -1)[0]
            pre = np.array([i for i, k in enumerate(kept) if k < 0 and k != ref])
            if len(pre) < 2:
                print("   %-6d (too few leads)" % ref)
                continue
            bp = beta[pre]; Vp = V[np.ix_(pre, pre)]
            try:
                stat = float(bp @ np.linalg.solve(Vp, bp))
            except np.linalg.LinAlgError:
                print("   %-6d (singular)" % ref); continue
            print("   %-6d %8.2f %6d %8.3f   %s"
                  % (ref, stat, len(pre), chi2_p(stat, len(pre)),
                     ",".join(str(kept[i]) for i in pre)))

        # ---- 2. linear pre-trend ----
        print("\n2. LINEAR PRE-TREND")
        print("   the concern is not noisy leads, it is sloping leads.")
        print("   GLS of lead coefficients on event time, using the full")
        print("   covariance matrix.\n")
        kept, beta, V = event_fit(sub, yname, -1)
        pre = np.array([i for i, k in enumerate(kept) if k < -1])
        kp = kept[pre].astype(float)
        bp = beta[pre]
        Vp = V[np.ix_(pre, pre)]
        Z = np.column_stack([np.ones(len(kp)), kp])
        try:
            Vinv = np.linalg.inv(Vp)
            A = np.linalg.inv(Z.T @ Vinv @ Z)
            g = A @ Z.T @ Vinv @ bp
            se_g = np.sqrt(np.diag(A))
            print("   slope per event year = %+.4f (se %.4f, t %.2f, p %.3f)"
                  % (g[1], se_g[1], g[1] / se_g[1], norm_p(g[1] / se_g[1])))
            print("   intercept at k=0     = %+.4f (se %.4f)" % (g[0], se_g[0]))
            print("   extrapolated pre-trend contribution to a 5-year post")
            print("   average, if the slope continued: %+.4f" % (g[1] * 2.5))
        except np.linalg.LinAlgError:
            print("   not computable")

        # ---- 3. sensitivity ----
        print("\n3. SENSITIVITY TO PARALLEL-TRENDS VIOLATION")
        print("   Rather than assume parallel trends, allow the post-treatment")
        print("   deviation to be up to M times the largest deviation visible")
        print("   before treatment, and widen the interval accordingly.")
        print("   M=1 means 'no worse after than before'.\n")
        post = np.array([i for i, k in enumerate(kept) if 0 <= k <= 4])
        if len(post) == 0:
            print("   no post periods\n"); continue
        w = np.ones(len(post)) / len(post)
        eff = float(w @ beta[post])
        se_eff = float(np.sqrt(w @ V[np.ix_(post, post)] @ w))
        # largest pre-treatment deviation, in absolute value
        maxpre = float(np.max(np.abs(beta[pre])))
        print("   post-period average effect (k=0..4) = %+.4f (se %.4f)"
              % (eff, se_eff))
        print("   conventional 95%% CI                 = [%+.4f, %+.4f]"
              % (eff - 1.96 * se_eff, eff + 1.96 * se_eff))
        print("   largest pre-treatment lead, abs     = %.4f\n" % maxpre)
        print("   %-6s %22s %s" % ("M", "95% CI", "excludes zero?"))
        for M in [0.0, 0.5, 1.0, 1.5, 2.0]:
            lo = eff - 1.96 * se_eff - M * maxpre
            hi = eff + 1.96 * se_eff + M * maxpre
            print("   %-6.1f [%+.4f, %+.4f]      %s"
                  % (M, lo, hi, "yes" if lo > 0 or hi < 0 else "no"))
        if maxpre > 0:
            bp_M = (eff - 1.96 * se_eff) / maxpre
            print("\n   breakdown M (interval first covers zero): %.2f" % max(bp_M, 0))
            print("   read as: the estimate survives a post-treatment violation")
            print("   up to %.2f times the largest pre-treatment deviation."
                  % max(bp_M, 0))
        print()


if __name__ == "__main__":
    main()