# 43_equip_check.py
# Script 42 produced exactly one significant gradient, and it was the placebo:
# equipment intensity on job reallocation, +1.894, p=0.034. Entry and birth
# employment were close behind (p=0.081, p=0.059). Both intangible measures
# were flat everywhere.
#
# Before this counts as a finding, three things need ruling out.
#
# 1. WHAT DOES EQUIPMENT INTENSITY ACTUALLY MEASURE? Its top sectors are
#    construction 0.962, transportation 0.909, wholesale 0.902, agriculture
#    0.871; its bottom is real estate 0.092. That ranking reads like
#    "physical operations vs finance and property" rather than capital
#    deepening. Section 1 checks what it correlates with.
#
# 2. IS IT INDEPENDENT OF THE INTANGIBLE MEASURES? Equipment and intangible
#    intensity are negatively related across sectors, so post x equip may be
#    picking up how the AGGREGATE effect distributes rather than a separate
#    gradient. Section 2 runs them together.
#
# 3. IS IT MULTIPLE COMPARISONS? Script 42 estimated 24 coefficients. One at
#    p=0.034 is what 24 draws produce. Section 3 runs a randomisation test:
#    reassign the exposure vector across sectors many times and see how often
#    a coefficient this large appears by chance.
#
# Run: py scripts\43_equip_check.py

import csv
import math
import os
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
PDIT = os.path.join(RAW, "export.csv")
BDS = os.path.join(RAW, "bds2023_st_sec.csv")
BEA = os.path.join("output", "42a_bea_exposure.csv")
OUTDIR = "output"

RATIO = "Discounted - 12% Rate"
NBOOT = 999
NPERM = 500
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
    b["exit_rate"] = b["estabs_exit"] / b["den"] * 100
    b["jrr"] = (b["job_creation"] + b["job_destruction"]) / b["denom"] * 100
    b["birth_emp_rate"] = b["job_creation_births"] / b["denom"] * 100
    return b[b["sector"].isin(keep)]


FE_GRAD = [("st", "year"), ("sector", "year"), ("st", "sector")]


def fit(d, yname, cols, target, rng, fes=FE_GRAD):
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
    return {"n": len(sub), "beta": bj, "p": pj, "lo": lo, "hi": hi,
            "mean": sub[yname].mean()}


def line(tag, r, mean=None):
    if r is None:
        print("  %-38s (not estimable)" % tag); return
    m = mean if mean is not None else r["mean"]
    if not np.isfinite(r["lo"]):
        print("  %-38s beta=%+.4f (bootstrap failed)" % (tag, r["beta"])); return
    star = " *" if r["p"] < 0.05 else ""
    print("  %-38s beta=%+.4f p=%.3f CI [%+.4f,%+.4f] (%+.1f%%,%+.1f%%)%s"
          % (tag, r["beta"], r["p"], r["lo"], r["hi"],
             100 * r["lo"] / m, 100 * r["hi"] / m, star))


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
    dbal = dv[(~trt) | dv["k"].between(BAL_LO, BAL_HI)].copy()
    for nm in ["pdit", "bea", "equip"]:
        dbal["post_" + nm] = dbal["post"] * dbal[nm + "_exp"]

    # ---------- 1 ----------
    print("=" * 78)
    print("1. WHAT DOES EQUIPMENT INTENSITY ACTUALLY MEASURE?")
    print("=" * 78)
    print("correlations across the 16 sectors in the sample:\n")
    print("  equip vs BEA intangible : %+.3f" % ex.equip_exp.corr(ex.bea_exp))
    print("  equip vs PDIT exposure  : %+.3f" % ex.equip_exp.corr(ex.pdit_exp))
    print("\nsector characteristics, sorted by equipment intensity:")
    # sector size and baseline dynamism, to see what equip is proxying for
    base = dbal[dbal.post == 0].groupby("sector").agg(
        emp_share=("denom", "sum"), entry=("entry_rate", "mean"),
        jrr=("jrr", "mean"))
    base["emp_share"] /= base["emp_share"].sum()
    tab = ex.join(base).sort_values("equip_exp", ascending=False)
    print("%-8s %8s %8s %8s %10s %8s %8s"
          % ("sector", "equip", "BEA", "PDIT", "emp share", "entry", "JRR"))
    for s, r in tab.iterrows():
        print("%-8s %8.3f %8.3f %8.3f %10.3f %8.2f %8.2f"
              % (s, r["equip_exp"], r["bea_exp"], r["pdit_exp"],
                 r["emp_share"], r["entry"], r["jrr"]))
    print("\n  equip vs baseline JRR   : %+.3f" % tab.equip_exp.corr(tab.jrr))
    print("  equip vs baseline entry : %+.3f" % tab.equip_exp.corr(tab.entry))
    print("\n  if equipment intensity correlates strongly with baseline dynamism,")
    print("  then post x equip partly reproduces the aggregate effect scaled by")
    print("  each sector's own level, which is not an independent gradient.")

    # ---------- 2 ----------
    print("\n" + "=" * 78)
    print("2. DOES IT SURVIVE CONTROLLING FOR THE INTANGIBLE MEASURES?")
    print("=" * 78)
    for yn in ["jrr", "entry_rate", "birth_emp_rate"]:
        mean = dbal[yn].mean()
        print("\n%s (mean %.2f):" % (yn, mean))
        line("equip alone",
             fit(dbal, yn, ["post_equip"], "post_equip", rng), mean)
        line("equip + BEA",
             fit(dbal, yn, ["post_equip", "post_bea"], "post_equip", rng), mean)
        line("equip + BEA + PDIT",
             fit(dbal, yn, ["post_equip", "post_bea", "post_pdit"],
                 "post_equip", rng), mean)
        line("equip + aggregate post",
             fit(dbal, yn, ["post_equip", "post"], "post_equip", rng), mean)

    # ---------- 3 ----------
    print("\n" + "=" * 78)
    print("3. RANDOMISATION TEST")
    print("=" * 78)
    print("reassign the equipment exposure vector across the 16 sectors at")
    print("random, %d times, and re-estimate. If the real coefficient sits" % NPERM)
    print("well inside the placebo distribution, it is multiple comparisons.\n")
    secs = list(ex.index)
    for yn in ["jrr", "entry_rate"]:
        sub = dbal.dropna(subset=[yn]).copy()
        sub = sub[np.isfinite(sub[yn].to_numpy(float))]
        cl = codes_for(sub, FE_GRAD)
        yv = absorb(sub[yn].to_numpy(float), cl)
        gid = sub["st"].to_numpy()
        real_x = absorb((sub["post"] * sub["equip_exp"]).to_numpy(float), cl)
        b0, inv0 = ols(real_x[:, None], yv)
        b_real = b0[0]

        vals = ex["equip_exp"].to_numpy(float)
        draws = []
        for _ in range(NPERM):
            perm = dict(zip(secs, rng.permutation(vals)))
            xx = sub["post"].to_numpy(float) * sub["sector"].map(perm).to_numpy(float)
            xx = absorb(xx, cl)
            if xx.std() < 1e-12:
                continue
            bb, _ = ols(xx[:, None], yv)
            draws.append(bb[0])
        draws = np.array(draws)
        pct = (np.abs(draws) >= abs(b_real)).mean()
        print("%s:" % yn)
        print("  real beta                = %+.4f" % b_real)
        print("  placebo mean             = %+.4f" % draws.mean())
        print("  placebo sd               = %.4f" % draws.std())
        print("  placebo 2.5 / 97.5 pct   = %+.4f / %+.4f"
              % tuple(np.quantile(draws, [0.025, 0.975])))
        print("  share of |placebo| >= |real| = %.3f" % pct)
        print("  -> %s\n"
              % ("real coefficient is NOT unusual; treat as noise" if pct > 0.1
                 else "real coefficient sits outside the placebo range"))

    # ---------- 4 ----------
    print("=" * 78)
    print("4. MULTIPLE COMPARISONS ACROSS SCRIPT 42")
    print("=" * 78)
    print("script 42 reported 24 gradient coefficients (4 outcomes x 6 specs).")
    print("under the null, the chance of at least one p < 0.05 among 24")
    print("independent draws is %.2f. The coefficients are not independent, so"
          % (1 - 0.95 ** 24))
    print("this overstates it, but the order of magnitude is the point: one")
    print("hit at p=0.034 out of 24 is what noise looks like.")
    print("\nBonferroni-adjusted threshold for 24 tests: p < %.4f" % (0.05 / 24))
    print("the equipment-JRR coefficient at p=0.034 does not clear it.")


if __name__ == "__main__":
    main()