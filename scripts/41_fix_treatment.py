# 41_fix_treatment.py
# Two errors found in the self-audit of scripts 38-40, plus one found in the
# first run of this script.
#
# ERROR 1: treatment was coded as absorbing (post = 1 for every year after
# adoption). It is not. Four states in the estimation sample repealed:
#     MO adopted 2001, credit off 2005-2015   (11 of 15 post years)
#     TX adopted 2000, off 2008-2013
#     MI adopted 2008, off 2012-2015
#     WA adopted 1994, off 2015
# 22 of 328 observable post state-years (6.7%) were mis-coded as treated.
#
# ERROR 2: extending the outcome window to 2023 was justified on the grounds
# that an event study only needs the adoption year. That holds only if
# treatment is absorbing. It is not, and PDIT ends in 2015, so treatment
# status is unobserved for 1996-2023. Main sample is now 1990-2015.
#
# ERROR 3 (found on the first run of this script): I asserted that
# never-treated states contribute nothing to the gradient because their
# treatment variable is identically zero and state-year FE absorbs them.
# The coefficients were NOT identical when they were dropped (+0.048 vs
# +0.233). They supply no treatment variation, but they do help estimate the
# sector-year fixed effects, which changes the residualised outcome. Section 4
# now tests what they actually do instead of asserting it.
#
# Run: py scripts\41_fix_treatment.py

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
BAL_LO, BAL_HI = -5, 4        # balanced event window

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
    """wild cluster bootstrap-t, Rademacher weights, null imposed"""
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
        Xr = np.zeros((len(y), 1)); br = np.zeros(1); er = y.copy()
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


def build(p):
    rd, va = "Research and Development Credit", "Industry Value-Added"
    live = p[p.groupby(["State", "Base Year"])[rd].transform("max") > 0]
    g = ((live[rd] * live[va]).groupby(live["sector"]).sum()
         / live[va].groupby(live["sector"]).sum())
    g = (g / g.max()).rename("exposure")
    sy = p.groupby(["State", "Base Year"])[rd].max().reset_index()
    sy["on"] = (sy[rd] > 0).astype(int)
    first = sy[sy.on == 1].groupby("State")["Base Year"].min()
    return g, first[first > 1990], sorted(set(p["State"]) - set(first.index)), sy


def load_bds(pdit_sectors):
    b = pd.read_csv(BDS, dtype={"st": str, "sector": str}, low_memory=False)
    b.columns = [c.strip().lower() for c in b.columns]
    for c in b.columns:
        if c not in ("st", "sector"):
            b[c] = pd.to_numeric(b[c], errors="coerce")
    b["st"] = pd.to_numeric(b["st"], errors="coerce")
    b["sector"] = b["sector"].str.strip()
    b = b.sort_values(["st", "sector", "year"])
    # DHS denominator rebuilt from counts. Averaging published sector rates
    # would weight a tiny sector the same as manufacturing. Checked against
    # the published rate: correlation 0.9998.
    b["estabs_lag"] = b.groupby(["st", "sector"])["estabs"].shift(1)
    b["estabs_denom"] = (b["estabs"] + b["estabs_lag"]) / 2
    b["in_pdit"] = b["sector"].isin(pdit_sectors)
    b["entry_rate"] = b["estabs_entry"] / b["estabs_denom"] * 100
    b["exit_rate"] = b["estabs_exit"] / b["estabs_denom"] * 100
    b["jrr"] = (b["job_creation"] + b["job_destruction"]) / b["denom"] * 100
    return b


def fit(d, yname, fe_specs, cols, target, rng):
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
    cl = codes_for(sub, fe_specs)
    y = absorb(y, cl); X = absorb(X, cl)
    try:
        bj, sj, tj, pj, lo, hi = wcb(X, y, sub["st"].to_numpy(),
                                     names.index(target), rng)
    except np.linalg.LinAlgError:
        return None
    return {"n": len(sub), "beta": bj, "se": sj, "p": pj, "lo": lo, "hi": hi,
            "mean": sub[yname].mean(), "states": sub["st"].nunique()}


def line(tag, r, mean=None):
    if r is None:
        print("  %-36s (not estimable)" % tag); return
    m = mean if mean is not None else r["mean"]
    if not np.isfinite(r["lo"]):
        print("  %-36s beta=%+.4f  (bootstrap failed)" % (tag, r["beta"])); return
    star = " *" if (r["p"] < 0.05) else ""
    print("  %-36s N=%6d beta=%+.4f p=%.3f CI [%+.4f,%+.4f] (%+.1f%%,%+.1f%%)%s"
          % (tag, r["n"], r["beta"], r["p"], r["lo"], r["hi"],
             100 * r["lo"] / m, 100 * r["hi"] / m, star))


FE_AGG = [("st", "sector"), ("sector", "year")]
FE_GRAD = [("st", "year"), ("sector", "year"), ("st", "sector")]


def main():
    rng = np.random.default_rng(SEED)
    p = load_pdit()
    g, adopters, never, sy = build(p)
    adopt_fips = {FIPS[s]: y for s, y in adopters.items()}
    ctrl = {FIPS[s] for s in never}

    # ---------- 1 ----------
    print("=" * 78)
    print("1. THE CODING ERROR")
    print("=" * 78)
    rev = []
    for st in adopters.index:
        gg = sy[(sy.State == st) & (sy["Base Year"] >= adopters[st])]
        off = gg.loc[gg.on == 0, "Base Year"].tolist()
        if off:
            rev.append((st, adopters[st], off))
    print("  states that repealed after adopting:")
    for st, a, off in rev:
        print("    %s adopted %d, credit off %d year(s): %d-%d"
              % (st, a, len(off), min(off), max(off)))
    tot = sum(len(sy[(sy.State == st) & (sy["Base Year"] >= adopters[st])])
              for st in adopters.index)
    mis = sum(len(o) for _, _, o in rev)
    print("  %d of %d observable post state-years mis-coded (%.1f%%)"
          % (mis, tot, 100 * mis / tot))

    on = sy.copy()
    on["st"] = on["State"].map(FIPS)
    on = on[["st", "Base Year", "on"]].rename(columns={"Base Year": "year"})

    b = load_bds(set(g.index))
    b = b[b["st"].isin(set(adopt_fips) | ctrl)]
    d = b[b["in_pdit"]].merge(on, on=["st", "year"], how="left")
    d["exposure"] = d["sector"].map(g)
    d = d.dropna(subset=["exposure"])
    d["adopt"] = d["st"].map(adopt_fips)
    d["k"] = np.where(d["adopt"].notna(), d["year"] - d["adopt"], -999)
    trt = (d["k"] > -900)
    d["post_absorb"] = ((d["k"] >= 0) & trt).astype(float)
    d["post_on"] = d["on"].fillna(0).astype(float)
    d["hi"] = d["sector"].isin(HIGH).astype(float)
    for c in ["post_absorb", "post_on"]:
        d[c + "_x"] = d[c] * d["exposure"]
        d[c + "_hi"] = d[c] * d["hi"]

    dv = d[(d.year >= 1990) & (d.year <= 2015)].copy()
    dbal = dv[(~(dv["k"] > -900)) | dv["k"].between(BAL_LO, BAL_HI)].copy()

    # ---------- 2 ----------
    print("\n" + "=" * 78)
    print("2. MAIN SAMPLE, 1990-2015 (treatment status observed)")
    print("=" * 78)
    print("* marks p < 0.05.\n")
    for yname in ["entry_rate", "exit_rate", "jrr"]:
        mean = dv[yname].mean()
        print("%s (mean %.2f):" % (yname, mean))
        line("aggregate, absorbing (old, wrong)",
             fit(dv, yname, FE_AGG, ["post_absorb"], "post_absorb", rng))
        line("aggregate, actual on/off (fixed)",
             fit(dv, yname, FE_AGG, ["post_on"], "post_on", rng))
        line("gradient continuous, fixed",
             fit(dv, yname, FE_GRAD, ["post_on_x"], "post_on_x", rng), mean)
        line("gradient binary, fixed",
             fit(dv, yname, FE_GRAD, ["post_on_hi"], "post_on_hi", rng), mean)
        print()

    print("  the correction LOWERS the aggregate rather than raising it.")
    print("  mis-coding untreated years as treated should attenuate, so the")
    print("  prior was that fixing it would raise the estimate. It did not,")
    print("  which means entry was relatively high in the repeal years. With")
    print("  only 22 such state-years this could be chance, but it does not")
    print("  support the positive-effect reading.")

    # ---------- 3 ----------
    print("\n" + "=" * 78)
    print("3. BALANCED EVENT WINDOW, k in [%d, %d]" % (BAL_LO, BAL_HI))
    print("=" * 78)
    print("a single post dummy on the full window averages MA's 25th post")
    print("year with FL's 4th. Restricting to five years either side puts")
    print("every adopter on the same horizon. Conceptually this is the")
    print("cleaner estimand, not a specification picked after seeing results.\n")
    for yname in ["entry_rate", "exit_rate", "jrr"]:
        mean = dbal[yname].mean()
        print("%s (mean %.2f):" % (yname, mean))
        line("aggregate", fit(dbal, yname, FE_AGG, ["post_on"], "post_on", rng))
        line("gradient continuous",
             fit(dbal, yname, FE_GRAD, ["post_on_x"], "post_on_x", rng), mean)
        line("gradient binary",
             fit(dbal, yname, FE_GRAD, ["post_on_hi"], "post_on_hi", rng), mean)
        print()

    # ---------- 4 ----------
    print("=" * 78)
    print("4. WHAT DO THE NEVER-TREATED STATES ACTUALLY DO?")
    print("=" * 78)
    print("I previously asserted they contribute nothing to the gradient")
    print("because their treatment variable is identically zero. That was")
    print("wrong: dropping them changed the coefficient from +0.048 to")
    print("+0.233. They supply no treatment variation, but they do help pin")
    print("down the sector-year fixed effects, and that changes the")
    print("residualised outcome for everyone else.\n")
    for tag, sel in [("all %d states" % (len(adopt_fips) + len(ctrl)), dv),
                     ("%d adopters only" % len(adopt_fips),
                      dv[dv.st.isin(adopt_fips)]),
                     ("balanced window, all states", dbal),
                     ("balanced window, adopters only",
                      dbal[dbal.st.isin(adopt_fips)])]:
        line("gradient: " + tag,
             fit(sel, "entry_rate", FE_GRAD, ["post_on_x"], "post_on_x", rng))
    print("\n  if these move a lot, the gradient depends on which states help")
    print("  estimate the sector-year terms, which is worth saying out loud.")
    print("  the aggregate specs are less exposed to this because the")
    print("  never-treated states are genuine comparison units there.")

    # ---------- 5 ----------
    print("\n" + "=" * 78)
    print("5. EXTENDED WINDOW 1990-2023, FOR COMPARISON ONLY")
    print("=" * 78)
    print("PDIT ends in 2015. Treatment status for 2016-2023 is UNOBSERVED.")
    print("These rows assume every credit stayed in force, which is already")
    print("false for four states inside the observed window. Reported so the")
    print("earlier numbers can be reconciled, not as an estimate.\n")
    line("aggregate, absorbing to 2023",
         fit(d, "entry_rate", FE_AGG, ["post_absorb"], "post_absorb", rng))
    line("gradient continuous to 2023",
         fit(d, "entry_rate", FE_GRAD, ["post_absorb_x"], "post_absorb_x", rng),
         d["entry_rate"].mean())

    # ---------- 6 ----------
    print("\n" + "=" * 78)
    print("6. SUMMARY OF EVERY GRADIENT ESTIMATE RUN SO FAR")
    print("=" * 78)
    print("  script 36  continuous, event study, ref k=-1      +0.146")
    print("  script 37  continuous, ref = pre-window average   -0.068")
    print("  script 39  continuous, rescaled to mean sector    +0.045")
    print("  script 40  binary, ref k=-1                       +0.324")
    print("  script 40  binary, ref = pre-window average       -0.003")
    print("  script 40  manufacturing alone                    -0.178")
    print("  script 40  information alone                      +0.783")
    print("  script 40  professional services (placebo)        -0.145")
    print("  script 40  accommodation (placebo)                +0.169")
    print("  script 41  continuous, treatment corrected        see above")
    print("  script 41  balanced window                        see above")
    print("\n  no specification produces a gradient that is both signed")
    print("  consistently and distinguishable from zero. Sector coefficients")
    print("  are unrelated to sector exposure: professional services, at")
    print("  exposure 0.224, is more negative than manufacturing at 1.000.")

    os.makedirs(OUTDIR, exist_ok=True)
    dv.to_csv(os.path.join(OUTDIR, "41_panel_corrected.csv"), index=False)
    print("\nwrote output\\41_panel_corrected.csv")


if __name__ == "__main__":
    main()