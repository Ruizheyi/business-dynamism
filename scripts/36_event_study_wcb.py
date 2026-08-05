# 36_event_study_wcb.py
# Corrected event study with wild cluster bootstrap CIs.
#
# Fixes carried over from the self-audit:
#  - no 2-sector subsample split. With only 2 sectors, state-year FE leaves
#    just the manuf-minus-info difference and the estimator degenerates.
#    Exposure enters continuously instead.
#  - the jrr=2807 coefficient was NOT an outlier problem. It was the
#    degenerate split above. Dropped, not winsorised.
#  - export_share is mechanically collinear with sector-year FE because PDIT
#    value-added is national and state-invariant. Removed entirely.
#
# WHAT THIS ESTIMATES, stated precisely because it matters:
#   state-year FE absorbs ALL state-level policy effects, which is exactly
#   the parameter Fazio-Guzman-Stern (2020) estimate. This regression can
#   only detect whether the credit's effect DIFFERS across sectors by R&D
#   exposure. A zero here does not contradict FGS.
#
# Inference: 11.7 effective clusters (script 32). Wild cluster bootstrap-t,
# Rademacher weights, null imposed. Cameron-Gelbach-Miller (2008).
# Run: py scripts\36_event_study_wcb.py

import os
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
PDIT = os.path.join(RAW, "export.csv")
BDS = os.path.join(RAW, "bds2023_st_sec.csv")
OUTDIR = "output"
RATIO = "Discounted - 12% Rate"
KMIN, KMAX, REF = -6, 11, -1
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


def load_pdit():
    p = pd.read_csv(PDIT)
    p["Industry"] = p["Industry"].str.strip()
    p = p[p["Ratio Type"] == RATIO]
    p = p[~p["Industry"].isin(["All Export", "All Non-Export"])].copy()
    mfg = [i for i in p["Industry"].unique() if i not in XWALK]
    xw = dict(XWALK)
    for m in mfg:
        xw[m] = "31-33"
    p["sector"] = p["Industry"].map(xw)
    return p


def build(p):
    rd, va = "Research and Development Credit", "Industry Value-Added"
    live = p[p.groupby(["State", "Base Year"])[rd].transform("max") > 0]
    g = ((live[rd] * live[va]).groupby(live["sector"]).sum()
         / live[va].groupby(live["sector"]).sum())
    g = (g / g.max()).rename("exposure")
    first = p[p[rd] > 0].groupby("State")["Base Year"].min()
    adopters = first[first > 1990]
    never = sorted(set(p["State"].unique()) - set(first.index))
    always = sorted(first[first <= 1990].index)
    return g, adopters, never, always


def load_bds():
    b = pd.read_csv(BDS, dtype={"st": str, "sector": str}, low_memory=False)
    b.columns = [c.strip().lower() for c in b.columns]
    for c in b.columns:
        if c not in ("st", "sector"):
            b[c] = pd.to_numeric(b[c], errors="coerce")
    b["st"] = pd.to_numeric(b["st"], errors="coerce")
    b["sector"] = b["sector"].str.strip()
    b["jrr"] = (b["job_creation"] + b["job_destruction"]) / b["denom"] * 100
    b["entry_rate"] = b["estabs_entry_rate"]
    b["exit_rate"] = b["estabs_exit_rate"]
    return b[["year", "st", "sector", "entry_rate", "jrr", "exit_rate", "denom"]]


def design(sub):
    ks = [k for k in range(KMIN, KMAX + 1) if k != REF]
    cols, keep_k = [], []
    e = sub["exposure"].to_numpy(float)
    for k in ks:
        if k == KMIN:
            ind = (sub["k"] <= KMIN)
        elif k == KMAX:
            ind = (sub["k"] >= KMAX) & (sub["k"] > -900)
        else:
            ind = (sub["k"] == k)
        v = ind.to_numpy(float) * e
        if v.std() > 1e-12:
            cols.append(v)
            keep_k.append(k)
    return np.column_stack(cols), keep_k


def ols(X, y):
    XtX = X.T @ X
    beta = np.linalg.solve(XtX, X.T @ y)
    return beta, y - X @ beta, np.linalg.inv(XtX)


def cluster_se(X, e, inv, gid):
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(gid):
        m = gid == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    G = len(np.unique(gid))
    n, k = X.shape
    V = inv @ meat @ inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    return np.sqrt(np.diag(V))


def wcb(X, y, gid, j, rng, nboot=NBOOT):
    """wild cluster bootstrap-t for coefficient j, null imposed"""
    beta, e, inv = ols(X, y)
    se = cluster_se(X, e, inv, gid)
    t0 = beta[j] / se[j]
    # restricted model: drop column j
    Xr = np.delete(X, j, axis=1)
    br, er, _ = ols(Xr, y)
    groups = np.unique(gid)
    ts = []
    for _ in range(nboot):
        w = rng.choice([-1.0, 1.0], size=len(groups))
        wmap = dict(zip(groups, w))
        wv = np.array([wmap[g] for g in gid])
        yb = Xr @ br + er * wv
        bb, eb, invb = ols(X, yb)
        sb = cluster_se(X, eb, invb, gid)
        if sb[j] > 0:
            ts.append(bb[j] / sb[j])
    ts = np.array(ts)
    p = (np.abs(ts) >= abs(t0)).mean()
    lo_t, hi_t = np.quantile(ts, [0.025, 0.975])
    return beta[j], se[j], t0, p, beta[j] - hi_t * se[j], beta[j] - lo_t * se[j]


def main():
    rng = np.random.default_rng(SEED)
    p = load_pdit()
    g, ad, never, always = build(p)

    print("=" * 72)
    print("SAMPLE")
    print("=" * 72)
    print("clean adopters (post-1990): %d" % len(ad))
    print(ad.sort_values().to_string())
    print("\nnever treated: %d  %s" % (len(never), never))
    print("dropped (treated by 1990): %d  %s" % (len(always), always))
    print("\nfixed sector exposure g(j), max normalised to 1:")
    print(g.sort_values(ascending=False).round(3).to_string())

    b = load_bds()
    use = {FIPS[s]: y for s, y in ad.items()}
    ctrl = {FIPS[s] for s in never}
    d = b[b["st"].isin(set(use) | ctrl)].copy()
    d["adopt"] = d["st"].map(use)
    d["k"] = np.where(d["adopt"].notna(), d["year"] - d["adopt"], -999)
    d["exposure"] = d["sector"].map(g)
    d = d.dropna(subset=["exposure"])
    print("\nsample: %d rows, %d states, %d sectors, %d-%d"
          % (len(d), d.st.nunique(), d.sector.nunique(), d.year.min(), d.year.max()))
    print("NOTE: outcome window runs to 2023 even though PDIT stops in 2015,")
    print("      because the event study only needs the adoption year.")

    for yname in ["entry_rate", "jrr", "exit_rate"]:
        sub = d.dropna(subset=[yname]).copy()
        X, ks = design(sub)
        y = sub[yname].to_numpy(float)
        cl = []
        for a2, b2 in [("st", "year"), ("sector", "year"), ("st", "sector")]:
            f = pd.factorize(sub[a2].astype(str) + "_" + sub[b2].astype(str))
            cl.append((f[0], len(f[1])))
        y = absorb(y, cl)
        X = absorb(X, cl)
        gid = sub["st"].to_numpy()

        beta, e, inv = ols(X, y)
        se = cluster_se(X, e, inv, gid)

        print("\n" + "=" * 72)
        print("EVENT STUDY: %s   N=%d  ref k=%d" % (yname, len(y), REF))
        print("=" * 72)
        print("%4s %9s %9s %7s   %s" % ("k", "beta", "se", "t", ""))
        for k, bb, ss in zip(ks, beta, se):
            t = bb / ss if ss > 0 else np.nan
            print("%4d %9.4f %9.4f %7.2f   %s%s"
                  % (k, bb, ss, t, "#" * min(int(abs(bb) / 0.05), 30),
                     "  <- pre" if k < 0 else ""))

        pre = [(bb / ss) ** 2 for k, bb, ss in zip(ks, beta, se) if k < REF and ss > 0]
        print("\n  pre-period joint stat (sum t^2, df=%d) = %.1f, 5%% crit ~ %.1f"
              % (len(pre), sum(pre), len(pre) + 2 * np.sqrt(2 * len(pre))))

        # post-period average effect, with WCB
        post_idx = [i for i, k in enumerate(ks) if k >= 5]
        if post_idx:
            R = np.zeros(X.shape[1])
            R[post_idx] = 1.0 / len(post_idx)
            Xa = np.column_stack([X @ R, X - np.outer(X @ R, R) / (R @ R)])
            # simpler: refit with a single post5 dummy
        d2 = sub.copy()
        d2["post5"] = ((d2["k"] >= 5) & (d2["k"] > -900)).astype(float) * d2["exposure"]
        d2["pre"] = ((d2["k"] <= -2) & (d2["k"] > -900)).astype(float) * d2["exposure"]
        d2["mid"] = ((d2["k"] >= 0) & (d2["k"] < 5)).astype(float) * d2["exposure"]
        Xs = absorb(d2[["pre", "mid", "post5"]].to_numpy(float), cl)
        for j, nm in [(0, "pre  (k<=-2)"), (1, "mid  (0<=k<5)"), (2, "post (k>=5)")]:
            bj, sj, tj, pj, lo, hi = wcb(Xs, y, gid, j, rng)
            print("  %-14s beta=%+.4f  se=%.4f  t=%5.2f  WCB p=%.3f  "
                  "95%% CI [%+.4f, %+.4f]" % (nm, bj, sj, tj, pj, lo, hi))

        pd.DataFrame({"k": ks, "beta": beta, "se": se}).to_csv(
            os.path.join(OUTDIR, "36_event_%s.csv" % yname), index=False)

    print("\nwrote output\\36_event_*.csv")


if __name__ == "__main__":
    main()