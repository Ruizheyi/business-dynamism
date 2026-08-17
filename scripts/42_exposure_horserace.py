# 42_exposure_horserace.py
# The gradient test so far sorted sectors on PDIT exposure, which measures how
# much R&D credit money a sector can claim. The hypothesis is about intangible
# capital. These are not the same thing: the two measures correlate 0.573
# (Spearman 0.506) and rank sectors very differently.
#
#     sector                 PDIT      BEA intangible
#     54 professional        0.224      0.546  (1st)
#     71 arts, entertainment 0.006      0.515  (2nd)
#     31-33 manufacturing    1.000      0.410  (3rd)
#     51 information         0.725      0.366  (4th)
#
# So the earlier finding that "sector coefficients bear no relation to
# exposure" may have used the wrong sorting variable. This script puts both
# measures in the same regression:
#
#     y = b1*(credit x PDIT) + b2*(credit x BEA) + FE
#
# b1 picks up the mechanical channel (who gets the money). b2 picks up the
# hypothesis (is the effect larger where intangibles matter). r = 0.573 gives
# VIF around 1.5, so they separate.
#
# Also fixes the pre-trend test. Script 36 summed squared t-statistics, which
# ignores the covariance between lead coefficients and is not a Wald test.
# Worse, script 36 used the absorbing treatment coding and the outcome window
# extended to 2023, both of which script 41 overturned. The README claim that
# pre-trends are not rejected currently has no valid basis. Redone here on the
# corrected sample with the full covariance matrix.
#
# Third: adds job_creation_births. Phase 1 found entry COUNTS moved while
# entry-driven EMPLOYMENT did not. If the credit raises entry but not
# entry-driven employment, the two halves of the project tell one story.
#
# Run: py scripts\42_exposure_horserace.py

import os
import numpy as np
import pandas as pd
import math

RAW = os.path.join("data", "raw")
PDIT = os.path.join(RAW, "export.csv")
BDS = os.path.join(RAW, "bds2023_st_sec.csv")
BEA = os.path.join("output", "42a_bea_exposure.csv")
OUTDIR = "output"

RATIO = "Discounted - 12% Rate"
NBOOT = 999
SEED = 20260805
BAL_LO, BAL_HI = -5, 4
KMIN, KMAX, REF = -5, 8, -1

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
    return np.linalg.solve(XtX, X.T @ y), np.linalg.inv(XtX)


def cluster_vcv(X, e, inv, gid):
    """full cluster-robust covariance matrix, not just the diagonal"""
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
    V = cluster_vcv(X, e, inv, gid)
    se = np.sqrt(np.diag(V))
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
    p = (np.abs(ts) >= abs(t0)).mean()
    lo_t, hi_t = np.quantile(ts, [0.025, 0.975])
    return beta[j], se[j], p, beta[j] - hi_t * se[j], beta[j] - lo_t * se[j]


def wald(beta, V, idx):
    """proper joint test: b' V^-1 b over the selected coefficients"""
    b = beta[idx]
    Vs = V[np.ix_(idx, idx)]
    try:
        stat = float(b @ np.linalg.solve(Vs, b))
    except np.linalg.LinAlgError:
        return np.nan, len(idx), np.nan
    df = len(idx)
    # chi2 survival without scipy: Wilson-Hilferty approximation
    z = ((stat / df) ** (1 / 3) - (1 - 2 / (9 * df))) / np.sqrt(2 / (9 * df))
    p = 0.5 * math.erfc(z / np.sqrt(2))
    return stat, df, p


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


def load_bds(keep_sectors):
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
    b["jrr"] = (b["job_creation"] + b["job_destruction"]) / b["denom"] * 100
    b["birth_emp_rate"] = b["job_creation_births"] / b["denom"] * 100
    b["death_emp_rate"] = b["job_destruction_deaths"] / b["denom"] * 100
    return b[b["sector"].isin(keep_sectors)]


def fit(d, yname, fes, cols, target, rng):
    sub = d.dropna(subset=[yname] + cols).copy()
    sub = sub[np.isfinite(sub[yname].to_numpy(float))]
    if len(sub) < 100:
        return None
    y = sub[yname].to_numpy(float)
    X = sub[cols].to_numpy(float)
    keep = X.std(axis=0) > 1e-12
    names = [c for c, m in zip(cols, keep) if m]
    if target not in names:
        return None
    X = X[:, keep]
    cl = codes_for(sub, fes)
    y = absorb(y, cl); X = absorb(X, cl)
    try:
        bj, sj, pj, lo, hi = wcb(X, y, sub["st"].to_numpy(),
                                 names.index(target), rng)
    except np.linalg.LinAlgError:
        return None
    # VIF of the target against the other regressors
    vif = np.nan
    if X.shape[1] > 1:
        j = names.index(target)
        Xo = np.delete(X, j, axis=1)
        try:
            bb, _ = ols(Xo, X[:, j])
            r = X[:, j] - Xo @ bb
            r2 = 1 - r.var() / X[:, j].var()
            vif = 1 / max(1 - r2, 1e-9)
        except np.linalg.LinAlgError:
            pass
    return {"n": len(sub), "beta": bj, "se": sj, "p": pj, "lo": lo, "hi": hi,
            "mean": sub[yname].mean(), "vif": vif}


def line(tag, r, mean=None):
    if r is None:
        print("  %-34s (not estimable)" % tag); return
    m = mean if mean is not None else r["mean"]
    if not np.isfinite(r["lo"]):
        print("  %-34s beta=%+.4f  (bootstrap failed)" % (tag, r["beta"])); return
    v = "" if not np.isfinite(r["vif"]) else "  VIF=%.2f" % r["vif"]
    star = " *" if r["p"] < 0.05 else ""
    print("  %-34s N=%5d beta=%+.4f p=%.3f CI [%+.4f,%+.4f] (%+.1f%%,%+.1f%%)%s%s"
          % (tag, r["n"], r["beta"], r["p"], r["lo"], r["hi"],
             100 * r["lo"] / m, 100 * r["hi"] / m, v, star))


FE_AGG = [("st", "sector"), ("sector", "year")]
FE_GRAD = [("st", "year"), ("sector", "year"), ("st", "sector")]


def main():
    rng = np.random.default_rng(SEED)
    p = load_pdit()
    g, adopters, never, sy = build(p)
    bea = pd.read_csv(BEA, dtype={"sector": str}).set_index("sector")

    ex = pd.DataFrame({"pdit_exp": g, "bea_exp": bea["bea_norm"],
                       "equip_exp": bea["equip_norm"]}).dropna()

    print("=" * 78)
    print("1. THE TWO EXPOSURE MEASURES")
    print("=" * 78)
    print("PDIT: how much R&D credit money the sector can claim.")
    print("BEA:  IPP / total fixed investment, 1985-89, pre-treatment.")
    print("The hypothesis is about the second. The gradient test used the first.\n")
    print("  Pearson  r = %.3f" % ex.pdit_exp.corr(ex.bea_exp))
    print("  Spearman r = %.3f" % ex.pdit_exp.corr(ex.bea_exp, method="spearman"))
    print("\n%-8s %10s %10s %10s" % ("sector", "PDIT", "BEA", "equip"))
    for s, r in ex.sort_values("bea_exp", ascending=False).iterrows():
        print("%-8s %10.3f %10.3f %10.3f"
              % (s, r["pdit_exp"], r["bea_exp"], r["equip_exp"]))

    adopt_fips = {FIPS[s]: y for s, y in adopters.items()}
    ctrl = {FIPS[s] for s in never}
    on = sy.copy(); on["st"] = on["State"].map(FIPS)
    on = on[["st", "Base Year", "on"]].rename(columns={"Base Year": "year"})

    b = load_bds(set(ex.index))
    b = b[b["st"].isin(set(adopt_fips) | ctrl)]
    d = b.merge(on, on=["st", "year"], how="left")
    d = d.join(ex, on="sector")
    d["adopt"] = d["st"].map(adopt_fips)
    d["k"] = np.where(d["adopt"].notna(), d["year"] - d["adopt"], -999)
    d["post"] = d["on"].fillna(0).astype(float)
    for nm in ["pdit", "bea", "equip"]:
        d["post_" + nm] = d["post"] * d[nm + "_exp"]

    dv = d[(d.year >= 1990) & (d.year <= 2015)].copy()
    trt = dv["k"] > -900
    dbal = dv[(~trt) | dv["k"].between(BAL_LO, BAL_HI)].copy()

    print("\n" + "=" * 78)
    print("2. HORSE RACE: WHICH EXPOSURE, IF EITHER, PREDICTS THE EFFECT?")
    print("=" * 78)
    print("balanced window k in [%d,%d], 1990-2015, treatment = actual on/off.\n"
          % (BAL_LO, BAL_HI))
    outcomes = [("entry_rate", "establishment entry"),
                ("birth_emp_rate", "employment from births"),
                ("exit_rate", "establishment exit"),
                ("jrr", "job reallocation")]
    for yn, lab in outcomes:
        mean = dbal[yn].mean()
        print("%s  (%s, mean %.2f):" % (yn, lab, mean))
        line("aggregate (no gradient)",
             fit(dbal, yn, FE_AGG, ["post"], "post", rng))
        line("PDIT exposure alone",
             fit(dbal, yn, FE_GRAD, ["post_pdit"], "post_pdit", rng), mean)
        line("BEA exposure alone",
             fit(dbal, yn, FE_GRAD, ["post_bea"], "post_bea", rng), mean)
        line("both: PDIT coefficient",
             fit(dbal, yn, FE_GRAD, ["post_pdit", "post_bea"], "post_pdit", rng), mean)
        line("both: BEA coefficient",
             fit(dbal, yn, FE_GRAD, ["post_pdit", "post_bea"], "post_bea", rng), mean)
        line("equipment (placebo)",
             fit(dbal, yn, FE_GRAD, ["post_equip"], "post_equip", rng), mean)
        print()

    print("=" * 78)
    print("3. DROPPING SECTOR 71 (ARTS AND ENTERTAINMENT)")
    print("=" * 78)
    print("BEA IPP bundles R&D, software AND entertainment originals. Sector 71")
    print("scores 0.944 on the BEA measure almost entirely on film and music")
    print("copyrights, which is not the intangible capital in the story.\n")
    d71 = dbal[dbal.sector != "71"]
    for yn, lab in [("entry_rate", "entry"), ("birth_emp_rate", "birth employment")]:
        mean = d71[yn].mean()
        print("%s:" % yn)
        line("BEA exposure, all sectors",
             fit(dbal, yn, FE_GRAD, ["post_bea"], "post_bea", rng), dbal[yn].mean())
        line("BEA exposure, drop 71",
             fit(d71, yn, FE_GRAD, ["post_bea"], "post_bea", rng), mean)
        print()

    print("=" * 78)
    print("4. PROPER PRE-TREND WALD TEST")
    print("=" * 78)
    print("script 36 summed squared t-stats, ignoring covariance between leads,")
    print("and did it on the absorbing-treatment sample that script 41 overturned.")
    print("redone here with the full cluster covariance matrix.\n")
    for yn, expcol in [("entry_rate", "post_bea"), ("entry_rate", "post_pdit")]:
        sub = dv.dropna(subset=[yn]).copy()
        sub = sub[np.isfinite(sub[yn].to_numpy(float))]
        e = sub[expcol.replace("post_", "") + "_exp"].to_numpy(float)
        ks = [k for k in range(KMIN, KMAX + 1) if k != REF]
        cols, kept = [], []
        for k in ks:
            if k == KMIN:
                ind = (sub["k"] <= KMIN) & (sub["k"] > -900)
            elif k == KMAX:
                ind = (sub["k"] >= KMAX)
            else:
                ind = (sub["k"] == k)
            v = ind.to_numpy(float) * e
            if v.std() > 1e-12:
                cols.append(v); kept.append(k)
        X = np.column_stack(cols)
        y = sub[yn].to_numpy(float)
        cl = codes_for(sub, FE_GRAD)
        y = absorb(y, cl); X = absorb(X, cl)
        beta, inv = ols(X, y)
        res = y - X @ beta
        V = cluster_vcv(X, y - X @ beta, inv, sub["st"].to_numpy())
        se = np.sqrt(np.diag(V))
        pre_idx = [i for i, k in enumerate(kept) if k < REF]
        stat, df, pv = wald(beta, V, pre_idx)

        print("--- %s, exposure = %s ---" % (yn, expcol))
        print("%5s %10s %10s %8s" % ("k", "beta", "se", "t"))
        for k, bb, ss in zip(kept, beta, se):
            print("%5d %10.4f %10.4f %8.2f%s"
                  % (k, bb, ss, bb / ss if ss > 0 else np.nan,
                     "  <- pre" if k < REF else ""))
        print("\n  joint Wald on %d pre-period leads: chi2 = %.2f, p = %.3f"
              % (df, stat, pv))
        print("  (script 36 reported sum of t^2 = 7.0 vs a made-up critical")
        print("   value; that number should not be used)\n")

    print("=" * 78)
    print("5. WHAT THIS CHANGES")
    print("=" * 78)
    print("  BEA coefficient zero too -> the null is now robust to TWO")
    print("     independent exposure measures, which is a stronger claim")
    print("     than the one the README currently makes.")
    print("  BEA coefficient nonzero  -> the earlier null came from sorting")
    print("     sectors on the wrong variable. That is a real finding and the")
    print("     gradient section has to be rewritten around it.")
    print("  equipment placebo nonzero -> whatever is going on is not specific")
    print("     to intangibles.")

    os.makedirs(OUTDIR, exist_ok=True)
    ex.to_csv(os.path.join(OUTDIR, "42_exposure_measures.csv"))
    dbal.to_csv(os.path.join(OUTDIR, "42_panel.csv"), index=False)
    print("\nwrote output\\42_exposure_measures.csv and 42_panel.csv")


if __name__ == "__main__":
    main()