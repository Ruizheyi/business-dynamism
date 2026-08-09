# 40_binary_gradient.py
# Section 5 of script 39 threw up something that needs resolving:
#   all 16 sectors     beta=+0.2977  p=0.069
#   drop 31-33 and 51  beta=+0.2572  p=0.116
#   only 31-33 and 51  beta=+0.5814  p=0.017
# The high-exposure sectors respond about twice as strongly, but the other
# fourteen do not lose their effect. That is a gradient, and it looks steadier
# than the continuous-exposure version in script 39, which flipped sign across
# reference periods (+0.25 at ref=-1, -0.05 at avg_pre).
#
# Possible reason: exposure values come out of Bartik's hypothetical-firm
# simulation and carry measurement noise, while "is this manufacturing or
# information" is clean.
#
# THE FIX to script 33's error. That script estimated the split by running the
# regression WITHIN each group. With two sectors in the high group, state-year
# FE leaves nothing but the manufacturing-minus-information difference and the
# estimator degenerates (that is where jrr=2807 came from). Here 'high' is a
# dummy interacted in the FULL sample, so the other fourteen sectors still
# supply the within-state-year comparison.
#
# Self-contained on purpose: chaining exec across scripts broke twice.
# Run: py scripts\40_binary_gradient.py

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
HIGH = ["31-33", "51"]
REFS = [-1, -2, -3, "avg_pre"]

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


# ---------------- FE machinery ----------------

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
    return np.linalg.solve(XtX, X.T @ y), None, np.linalg.inv(XtX)


def cluster_se(X, e, inv, gid):
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(gid):
        m = gid == g
        s = X[m].T @ e[m]
        meat += np.outer(s, s)
    G = len(np.unique(gid)); n, k = X.shape
    V = inv @ meat @ inv * (G / max(G - 1, 1)) * ((n - 1) / max(n - k, 1))
    return np.sqrt(np.diag(V))


def wcb(X, y, gid, j, rng, nboot=NBOOT):
    """wild cluster bootstrap-t, Rademacher weights, null imposed.
    handles the single-column case and skips singular draws."""
    beta, _, inv = ols(X, y)
    e = y - X @ beta
    se = cluster_se(X, e, inv, gid)
    if not np.isfinite(se[j]) or se[j] <= 0:
        return beta[j], np.nan, np.nan, np.nan, np.nan, np.nan
    t0 = beta[j] / se[j]
    if X.shape[1] > 1:
        Xr = np.delete(X, j, axis=1)
        br, _, _ = ols(Xr, y)
        er = y - Xr @ br
    else:
        Xr = np.zeros((len(y), 1))
        br = np.zeros(1)
        er = y.copy()
    groups = np.unique(gid)
    ts = []
    for _ in range(nboot):
        w = rng.choice([-1.0, 1.0], size=len(groups))
        wmap = dict(zip(groups, w))
        wv = np.array([wmap[g] for g in gid])
        yb = Xr @ br + er * wv
        try:
            bb, _, invb = ols(X, yb)
        except np.linalg.LinAlgError:
            continue
        eb = yb - X @ bb
        sb = cluster_se(X, eb, invb, gid)
        if np.isfinite(sb[j]) and sb[j] > 0:
            ts.append(bb[j] / sb[j])
    ts = np.array(ts)
    if len(ts) < 50:
        return beta[j], se[j], t0, np.nan, np.nan, np.nan
    p = (np.abs(ts) >= abs(t0)).mean()
    lo_t, hi_t = np.quantile(ts, [0.025, 0.975])
    return beta[j], se[j], t0, p, beta[j] - hi_t * se[j], beta[j] - lo_t * se[j]


# ---------------- data ----------------

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


def treatment_calendar(p):
    rd, va = "Research and Development Credit", "Industry Value-Added"
    live = p[p.groupby(["State", "Base Year"])[rd].transform("max") > 0]
    g = ((live[rd] * live[va]).groupby(live["sector"]).sum()
         / live[va].groupby(live["sector"]).sum())
    g = (g / g.max()).rename("exposure")
    first = p[p[rd] > 0].groupby("State")["Base Year"].min()
    return g, first[first > 1990], sorted(set(p["State"]) - set(first.index))


def load_bds(pdit_sectors):
    b = pd.read_csv(BDS, dtype={"st": str, "sector": str}, low_memory=False)
    b.columns = [c.strip().lower() for c in b.columns]
    for c in b.columns:
        if c not in ("st", "sector"):
            b[c] = pd.to_numeric(b[c], errors="coerce")
    b["st"] = pd.to_numeric(b["st"], errors="coerce")
    b["sector"] = b["sector"].str.strip()
    b = b.sort_values(["st", "sector", "year"])
    # DHS denominator rebuilt from counts; averaging published rates would
    # weight a tiny sector the same as manufacturing
    b["estabs_lag"] = b.groupby(["st", "sector"])["estabs"].shift(1)
    b["estabs_denom"] = (b["estabs"] + b["estabs_lag"]) / 2
    b["in_pdit"] = b["sector"].isin(pdit_sectors)
    return b


def sector_outcomes(b):
    d = b.copy()
    d["entry_rate"] = d["estabs_entry"] / d["estabs_denom"] * 100
    d["exit_rate"] = d["estabs_exit"] / d["estabs_denom"] * 100
    d["jrr"] = (d["job_creation"] + d["job_destruction"]) / d["denom"] * 100
    return d


def add_event_time(d, adopt_fips):
    d = d.copy()
    d["adopt"] = d["st"].map(adopt_fips)
    d["k"] = np.where(d["adopt"].notna(), d["year"] - d["adopt"], -999)
    return d


def event_terms(d, ref):
    d = d.copy()
    trt = d["k"] > -900
    if ref == "avg_pre":
        d["pre"] = 0.0                      # whole pre window is the base
    else:
        d["pre"] = ((d["k"] <= -2) & trt & (d["k"] != ref)).astype(float)
    d["post"] = ((d["k"] >= 0) & trt).astype(float)
    return d


def prep(d, ref):
    d = event_terms(d, ref)
    d["high"] = d["sector"].isin(HIGH).astype(float)
    d["post_high"] = d["post"] * d["high"]
    d["pre_high"] = d["pre"] * d["high"]
    d["post_x"] = d["post"] * d["exposure"]
    d["pre_x"] = d["pre"] * d["exposure"]
    return d


# ---------------- estimation ----------------

def fit_named(d, yname, fe_specs, treat_cols, target, rng):
    sub = d.dropna(subset=[yname] + treat_cols).copy()
    sub = sub[np.isfinite(sub[yname].to_numpy(float))]
    if len(sub) < 100:
        return None
    y = sub[yname].to_numpy(float)
    X = sub[treat_cols].to_numpy(float)
    keep = X.std(axis=0) > 1e-12
    names = [c for c, m in zip(treat_cols, keep) if m]
    if target not in names:
        return None
    X = X[:, keep]
    cl = codes_for(sub, fe_specs)
    y = absorb(y, cl)
    X = absorb(X, cl)
    try:
        bj, sj, tj, pj, lo, hi = wcb(X, y, sub["st"].to_numpy(),
                                     names.index(target), rng)
    except np.linalg.LinAlgError:
        return None
    return {"n": len(sub), "beta": bj, "se": sj, "p": pj, "lo": lo,
            "hi": hi, "mean": sub[yname].mean()}


def line(tag, r, mean=None):
    if r is None:
        print("  %-34s (not estimable)" % tag)
        return
    m = mean if mean is not None else r["mean"]
    if not np.isfinite(r["lo"]):
        print("  %-34s beta=%+.4f   (bootstrap failed)" % (tag, r["beta"]))
        return
    print("  %-34s beta=%+.4f  p=%.3f  CI [%+.4f, %+.4f]  (%+.1f%%, %+.1f%%)"
          % (tag, r["beta"], r["p"], r["lo"], r["hi"],
             100 * r["lo"] / m, 100 * r["hi"] / m))


FE_AGG = [("st", "sector"), ("sector", "year")]
FE_GRAD = [("st", "year"), ("sector", "year"), ("st", "sector")]


def main():
    rng = np.random.default_rng(SEED)
    p = load_pdit()
    g, adopters, never = treatment_calendar(p)
    adopt_fips = {FIPS[s]: y for s, y in adopters.items()}
    ctrl = {FIPS[s] for s in never}
    b = load_bds(set(g.index))
    b = b[b["st"].isin(set(adopt_fips) | ctrl)]
    d = sector_outcomes(b[b["in_pdit"]])
    d = add_event_time(d, adopt_fips)
    d["exposure"] = d["sector"].map(g)
    d = d.dropna(subset=["exposure"])

    print("=" * 78)
    print("1. THE SPLIT, DONE CORRECTLY THIS TIME")
    print("=" * 78)
    print("script 33 ran the split as separate subsample regressions. With two")
    print("sectors in the high group, state-year FE leaves only the")
    print("manufacturing-minus-information difference and the estimator")
    print("degenerates. Here 'high' is a dummy interacted in the full sample,")
    print("so the other fourteen sectors still identify the state-year terms.\n")
    hi_emp = d[d.sector.isin(HIGH)]["denom"].sum() / d["denom"].sum()
    print("  high group: %s" % ", ".join(HIGH))
    print("  employment share of high group: %.3f" % hi_emp)
    print("  exposure: 31-33 = 1.000, 51 = 0.725, all others <= 0.224\n")

    dd = prep(d, -1)
    for yname in ["entry_rate", "exit_rate", "jrr"]:
        mean = d[yname].mean()
        print("%s (mean %.2f):" % (yname, mean))
        line("aggregate, no gradient",
             fit_named(dd, yname, FE_AGG, ["pre", "post"], "post", rng))
        line("binary gradient (high vs rest)",
             fit_named(dd, yname, FE_GRAD, ["pre_high", "post_high"],
                       "post_high", rng), mean)
        line("continuous gradient",
             fit_named(dd, yname, FE_GRAD, ["pre_x", "post_x"],
                       "post_x", rng), mean)
        line("binary, controlling continuous",
             fit_named(dd, yname, FE_GRAD,
                       ["pre_high", "post_high", "pre_x", "post_x"],
                       "post_high", rng), mean)
        print()

    print("=" * 78)
    print("2. IS THE BINARY GRADIENT STABLE ACROSS REFERENCE PERIODS?")
    print("=" * 78)
    print("this is the question. the continuous version swung from +0.25 to")
    print("-0.05 in script 39. if the binary version holds steady, the")
    print("heterogeneity is real and the continuous measure was just noisy.")
    print("if it swings too, there is no robust heterogeneity.\n")
    print("%-10s %-30s %10s %8s   %s" % ("ref", "term", "beta", "p", "95% CI"))
    for ref in REFS:
        dd = prep(d, ref)
        for tag, cols, target, fes in [
                ("aggregate (no gradient)", ["pre", "post"], "post", FE_AGG),
                ("binary gradient", ["pre_high", "post_high"],
                 "post_high", FE_GRAD),
                ("continuous gradient", ["pre_x", "post_x"],
                 "post_x", FE_GRAD)]:
            use = [c for c in cols
                   if not (ref == "avg_pre" and c.startswith("pre"))]
            r = fit_named(dd, "entry_rate", fes, use, target, rng)
            if r is None:
                print("%-10s %-30s (not estimable)" % (str(ref), tag))
            elif not np.isfinite(r["lo"]):
                print("%-10s %-30s %+10.4f      n/a"
                      % (str(ref), tag, r["beta"]))
            else:
                print("%-10s %-30s %+10.4f %8.3f   [%+.4f, %+.4f]"
                      % (str(ref), tag, r["beta"], r["p"], r["lo"], r["hi"]))
        print()

    print("=" * 78)
    print("3. WHICH OF THE TWO SECTORS DRIVES IT?")
    print("=" * 78)
    print("manufacturing is much larger than information. if the result is")
    print("entirely one sector it is much weaker than it looks.\n")
    dd = prep(d, -1)
    mean = d["entry_rate"].mean()
    for lab, sec in [("manufacturing only (31-33)", ["31-33"]),
                     ("information only (51)", ["51"]),
                     ("professional svc (54, placebo)", ["54"]),
                     ("accommodation (72, placebo)", ["72"])]:
        dd["g"] = dd["sector"].isin(sec).astype(float)
        dd["post_g"] = dd["post"] * dd["g"]
        dd["pre_g"] = dd["pre"] * dd["g"]
        line(lab, fit_named(dd, "entry_rate", FE_GRAD,
                            ["pre_g", "post_g"], "post_g", rng), mean)
    print("\n  sector 54 has exposure 0.224 and 72 has 0.007, both well below")
    print("  31-33 and 51. if they jump too, the split is not picking up")
    print("  R&D exposure but something else about those sectors.")

    print("\n" + "=" * 78)
    print("4. LEAVE-ONE-STATE-OUT ON THE BINARY GRADIENT")
    print("=" * 78)
    dd = prep(d, -1)
    full = fit_named(dd, "entry_rate", FE_GRAD,
                     ["pre_high", "post_high"], "post_high", rng)
    if full:
        bf = full["beta"]
        print("  full sample beta = %+.4f\n" % bf)
        out = []
        for st in sorted(dd["st"].unique()):
            r = fit_named(dd[dd.st != st], "entry_rate", FE_GRAD,
                          ["pre_high", "post_high"], "post_high", rng)
            if r:
                out.append({"dropped_fips": st, "beta": r["beta"]})
        o = pd.DataFrame(out).sort_values("beta")
        print("  range: %+.4f to %+.4f" % (o.beta.min(), o.beta.max()))
        print("  sign flips: %d of %d"
              % ((np.sign(o.beta) != np.sign(bf)).sum(), len(o)))
        print("  most influential drops:")
        for _, r in pd.concat([o.head(3), o.tail(3)]).iterrows():
            print("    FIPS %2d -> %+.4f (%+.0f%%)"
                  % (r["dropped_fips"], r["beta"],
                     100 * (r["beta"] - bf) / abs(bf)))
        os.makedirs(OUTDIR, exist_ok=True)
        o.to_csv(os.path.join(OUTDIR, "40_loo_binary.csv"), index=False)
        print("\n  wrote output\\40_loo_binary.csv")

    print("\n" + "=" * 78)
    print("READING GUIDE")
    print("=" * 78)
    print("  binary stable + placebo sectors null + few sign flips")
    print("    -> real heterogeneity; the continuous measure was noisy")
    print("  binary swings like the continuous one did")
    print("    -> no robust heterogeneity; report the aggregate only")
    print("  binary stable but sector 54 or 72 also jumps")
    print("    -> the split is picking up something other than R&D exposure")


if __name__ == "__main__":
    main()