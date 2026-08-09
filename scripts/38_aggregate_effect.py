# 38_aggregate_effect.py
# Lucia's point: the three-way FE coefficient estimates only whether adoption
# moves dynamism MORE in high-exposure sectors. To say anything about the
# channel, the aggregate effect has to be estimated on the same BDS outcomes
# and the same sample, then placed next to the sector gradient.
#
# Also per her note: FGS is NOT a comparable benchmark. They use business
# registration data and report a cumulative ten-year effect. Nothing here is
# scaled against their 20%.
#
# Three estimates, same states, same window, same inference:
#   A  aggregate, state-level collapse       state FE + year FE
#   B  aggregate, panel form                 state-sector FE + sector-year FE
#   C  sector gradient (from script 36)      + state-year FE, exposure interacted
#
# A and B answer "did adoption move dynamism at all". C answers "did it move
# more where R&D exposure is higher". The pair is the object of interest.
#
# Run: py scripts\38_aggregate_effect.py

import os
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
PDIT = os.path.join(RAW, "export.csv")
BDS = os.path.join(RAW, "bds2023_st_sec.csv")
OUTDIR = "output"

RATIO = "Discounted - 12% Rate"
KMIN, KMAX = -6, 11
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


# ---------- FE machinery ----------

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
    """specs is a list of column-name tuples, e.g. [('st',),('year',)]"""
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
    beta, _, inv = ols(X, y)
    e = y - X @ beta
    se = cluster_se(X, e, inv, gid)
    t0 = beta[j] / se[j] if se[j] > 0 else np.nan
    Xr = np.delete(X, j, axis=1) if X.shape[1] > 1 else np.zeros((len(y), 1))
    br, _, _ = ols(Xr, y)
    er = y - Xr @ br
    groups = np.unique(gid)
    ts = []
    for _ in range(nboot):
        w = rng.choice([-1.0, 1.0], size=len(groups))
        wv = np.array([dict(zip(groups, w))[g] for g in gid])
        yb = Xr @ br + er * wv
        bb, _, invb = ols(X, yb)
        eb = yb - X @ bb
        sb = cluster_se(X, eb, invb, gid)
        if sb[j] > 0:
            ts.append(bb[j] / sb[j])
    ts = np.array(ts)
    if len(ts) < 50:
        # bootstrap failed, usually collinearity or non-finite data upstream.
        # return the point estimate but no interval, rather than crashing.
        return beta[j], se[j], t0, np.nan, np.nan, np.nan
    p = (np.abs(ts) >= abs(t0)).mean()
    lo_t, hi_t = np.quantile(ts, [0.025, 0.975])
    return beta[j], se[j], t0, p, beta[j] - hi_t * se[j], beta[j] - lo_t * se[j]


# ---------- data ----------

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
    # DHS denominator for establishment rates, rebuilt from counts so that
    # aggregation across sectors is correct (averaging published rates is not)
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


def state_outcomes(b, restrict_to_pdit):
    """collapse to state-year by summing counts, then form rates"""
    d = b[b["in_pdit"]] if restrict_to_pdit else b
    g = d.groupby(["st", "year"], as_index=False).agg(
        estabs_entry=("estabs_entry", "sum"),
        estabs_exit=("estabs_exit", "sum"),
        estabs_denom=("estabs_denom", "sum"),
        job_creation=("job_creation", "sum"),
        job_destruction=("job_destruction", "sum"),
        denom=("denom", "sum"))
    g["entry_rate"] = g["estabs_entry"] / g["estabs_denom"] * 100
    g["exit_rate"] = g["estabs_exit"] / g["estabs_denom"] * 100
    g["jrr"] = (g["job_creation"] + g["job_destruction"]) / g["denom"] * 100
    return g


def add_event_time(d, adopt_fips):
    d = d.copy()
    d["adopt"] = d["st"].map(adopt_fips)
    d["k"] = np.where(d["adopt"].notna(), d["year"] - d["adopt"], -999)
    trt = d["k"] > -900
    d["pre"] = ((d["k"] <= -2) & trt).astype(float)
    d["post"] = ((d["k"] >= 0) & trt).astype(float)
    return d


# ---------- estimation ----------

def run(d, yname, fe_specs, treat_cols, rng, label):
    sub = d.dropna(subset=[yname] + treat_cols).copy()
    # inf shows up when the DHS denominator sums to zero: the first year has
    # no lag, and groupby.sum skips NaN so it collapses to 0 rather than NaN.
    # dropna does not catch inf, so filter explicitly.
    sub = sub[np.isfinite(sub[yname].to_numpy(float))].copy()
    if len(sub) < 100:
        print("  %-34s (too few usable rows)" % label)
        return None
    y = sub[yname].to_numpy(float)
    X = sub[treat_cols].to_numpy(float)
    keep = X.std(axis=0) > 1e-12
    if not keep.any():
        print("  %-34s (no variation)" % label)
        return None
    X, names = X[:, keep], [c for c, m in zip(treat_cols, keep) if m]
    cl = codes_for(sub, fe_specs)
    y = absorb(y, cl)
    X = absorb(X, cl)
    gid = sub["st"].to_numpy()
    j = names.index("post") if "post" in names else 0
    bj, sj, tj, pj, lo, hi = wcb(X, y, gid, j, rng)
    mean = sub[yname].mean()
    print("  %-34s N=%6d  beta=%+.4f  se=%.4f  p=%.3f  CI [%+.4f, %+.4f]"
          % (label, len(sub), bj, sj, pj, lo, hi))
    print("  %-34s mean=%.2f  ->  CI as %% of mean [%+.1f%%, %+.1f%%]"
          % ("", mean, 100 * lo / mean, 100 * hi / mean))
    return {"label": label, "outcome": yname, "n": len(sub), "beta": bj,
            "se": sj, "p": pj, "lo": lo, "hi": hi, "mean": mean}


def main():
    rng = np.random.default_rng(SEED)
    p = load_pdit()
    g, adopters, never = treatment_calendar(p)
    adopt_fips = {FIPS[s]: y for s, y in adopters.items()}
    ctrl = {FIPS[s] for s in never}
    keep_states = set(adopt_fips) | ctrl
    pdit_sectors = set(g.index)

    print("=" * 78)
    print("SAMPLE (identical across all three estimates)")
    print("=" * 78)
    print("adopters %d, never-treated %d, total states %d"
          % (len(adopt_fips), len(ctrl), len(keep_states)))
    print("PDIT-covered sectors: %d of 19" % len(pdit_sectors))
    print("\nNOTE: entry and exit rates are rebuilt from counts using the DHS")
    print("denominator, because averaging published sector rates would weight")
    print("a small sector the same as manufacturing.")

    b = load_bds(pdit_sectors)
    b = b[b["st"].isin(keep_states)]

    rows = []

    # ---- A: aggregate, state-level collapse ----
    print("\n" + "=" * 78)
    print("A. AGGREGATE EFFECT, state-year panel")
    print("=" * 78)
    print("state FE + year FE. treatment is adoption itself, no exposure.")
    print("this is the closest thing in BDS to the FGS estimand.\n")
    for restrict, tag in [(False, "all 19 sectors"), (True, "PDIT 16 sectors")]:
        s = add_event_time(state_outcomes(b, restrict), adopt_fips)
        for yname in ["entry_rate", "exit_rate", "jrr"]:
            r = run(s, yname, [("st",), ("year",)], ["pre", "post"], rng,
                    "%s / %s" % (yname, tag))
            if r:
                r["spec"] = "A aggregate state-year"
                rows.append(r)
        print()

    # ---- B: aggregate, panel form ----
    print("=" * 78)
    print("B. AGGREGATE EFFECT, state-sector-year panel")
    print("=" * 78)
    print("state-sector FE + sector-year FE. same cells as C below, so the")
    print("comparison with the gradient is like for like.\n")
    d = sector_outcomes(b[b["in_pdit"]])
    d = add_event_time(d, adopt_fips)
    for yname in ["entry_rate", "exit_rate", "jrr"]:
        r = run(d, yname, [("st", "sector"), ("sector", "year")],
                ["pre", "post"], rng, yname)
        if r:
            r["spec"] = "B aggregate panel"
            rows.append(r)

    # ---- C: sector gradient ----
    print("\n" + "=" * 78)
    print("C. SECTOR GRADIENT, state-sector-year panel")
    print("=" * 78)
    print("adds state-year FE and interacts treatment with exposure.")
    print("this absorbs the aggregate effect by construction.\n")
    d["exposure"] = d["sector"].map(g)
    d = d.dropna(subset=["exposure"])
    d["pre_x"] = d["pre"] * d["exposure"]
    d["post_x"] = d["post"] * d["exposure"]
    for yname in ["entry_rate", "exit_rate", "jrr"]:
        sub = d.dropna(subset=[yname]).copy()
        sub = sub.rename(columns={"pre_x": "pre_i", "post_x": "post_i"})
        sub["pre"], sub["post"] = sub["pre_i"], sub["post_i"]
        r = run(sub, yname,
                [("st", "year"), ("sector", "year"), ("st", "sector")],
                ["pre", "post"], rng, yname)
        if r:
            r["spec"] = "C sector gradient"
            rows.append(r)

    # ---- side by side ----
    print("\n" + "=" * 78)
    print("SIDE BY SIDE  (this is the table Lucia asked for)")
    print("=" * 78)
    out = pd.DataFrame(rows)
    for yname in ["entry_rate", "exit_rate", "jrr"]:
        sel = out[(out.outcome == yname) & (out.label.str.contains("all 19|PDIT 16") == False)]
        agg_a = out[(out.outcome == yname) & (out.label.str.contains("all 19"))]
        print("\n%s:" % yname)
        for _, r in pd.concat([agg_a, sel]).iterrows():
            print("  %-26s beta=%+.4f  p=%.3f  CI [%+.4f, %+.4f]  (%+.1f%%, %+.1f%%)"
                  % (r["spec"], r["beta"], r["p"], r["lo"], r["hi"],
                     100 * r["lo"] / r["mean"], 100 * r["hi"] / r["mean"]))

    print("\nreading guide:")
    print("  A or B positive, C zero  -> policy works, not via R&D exposure")
    print("  A and B zero,   C zero   -> no BDS effect at all; the gap with FGS")
    print("                              is about outcome data, not channels")
    print("  A or B positive, C positive -> gradient exists after all")
    print("nothing here is scaled against the FGS 20%: different outcome data,")
    print("and theirs is a cumulative ten-year figure.")

    os.makedirs(OUTDIR, exist_ok=True)
    out.to_csv(os.path.join(OUTDIR, "38_aggregate_vs_gradient.csv"), index=False)
    print("\nwrote output\\38_aggregate_vs_gradient.csv")


if __name__ == "__main__":
    main()