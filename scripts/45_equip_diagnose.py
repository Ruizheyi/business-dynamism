# 45_equip_diagnose.py
# Script 44 established the equipment gradient is not noise:
#   job creation  +1.697, randomisation p=0.0074, WCB p=0.010
#   job destruction +0.197, p=0.717   (nothing)
#   entry rate    +0.283, p=0.377     (nothing)
#   leave-one-sector-out: 0 sign flips in 16, range +1.57 to +2.25
#
# So R&D credit adoption is followed by higher job creation in
# equipment-intensive sectors, with no movement in destruction or entry.
# That is the opposite of the paper's hypothesis and has no obvious mechanism.
#
# Three explanations, each testable with data already on disk.
#
# A CYCLE. The top equipment sectors are construction 0.962, transportation
#   0.909, wholesale 0.902 -- all cyclically sensitive. If states adopt credits
#   at particular points in their own business cycle, the coefficient picks up
#   the cycle rather than the policy.
#
# B SCALE. Equipment intensity may proxy for sectors that were growing anyway.
#
# C REAL. Tangible capital intensity is genuinely the operative characteristic.
#
# Section 1 measures each sector's cyclical sensitivity from pre-1990 data and
# correlates it with equipment intensity. Section 2 does the same for
# pre-period growth. Section 3 swaps in structures intensity, which is also
# tangible but distributed differently across sectors. Section 4 checks whether
# equipment-intensive sectors are concentrated in generous states. Section 5
# controls for cyclical sensitivity directly.
#
# Run: py scripts\45_equip_diagnose.py

import csv
import os
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
PDIT = os.path.join(RAW, "export.csv")
BDS = os.path.join(RAW, "bds2023_st_sec.csv")
BEA = os.path.join("output", "42a_bea_exposure.csv")
OUTDIR = "output"

RATIO = "Discounted - 12% Rate"
NPERM = 2000
NBOOT = 999
SEED = 20260805
BAL_LO, BAL_HI = -5, 4
PRE_LO, PRE_HI = 1978, 1989          # pre-treatment window for sector traits

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
BEA2BDS = {
    "Agriculture, forestry, fishing, and hunting": "11", "Mining": "21",
    "Utilities": "22", "Construction": "23", "Manufacturing": "31-33",
    "Wholesale trade": "42", "Retail trade": "44-45",
    "Transportation and warehousing": "48-49", "Information": "51",
    "Finance and insurance": "52", "Real estate and rental and leasing": "53",
    "Professional, scientific, and technical services": "54",
    "Management of companies and enterprises 5": "55",
    "Administrative and waste management services": "56",
    "Educational services": "61", "Health and social assistance": "62",
    "Arts, entertainment, and recreation": "71",
    "Accommodation and food services": "72",
    "Other services, except government": "81",
}


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


def load_bea_raw(fname):
    path = os.path.join(RAW, fname)
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    hdr = [i for i, l in enumerate(lines) if l.startswith("Line,")][0]
    yrs = [y.strip() for y in list(csv.reader([lines[hdr]]))[0][2:] if y.strip()]
    out = {}
    for r in csv.reader(lines[hdr + 1:]):
        if not r or not r[0].strip().isdigit():
            continue
        lab = r[1]
        if len(lab) - len(lab.lstrip()) != 0:
            continue
        vals = []
        for v in r[2:2 + len(yrs)]:
            try:
                vals.append(float(v.replace(",", "")))
            except ValueError:
                vals.append(np.nan)
        out[lab.strip()] = pd.Series(vals, index=[int(y) for y in yrs])
    return out


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


def load_bds_full():
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
    b["jrr"] = (b["job_creation"] + b["job_destruction"]) / b["denom"] * 100
    b["jc_rate"] = b["job_creation"] / b["denom"] * 100
    b["jd_rate"] = b["job_destruction"] / b["denom"] * 100
    return b


FE_GRAD = [("st", "year"), ("sector", "year"), ("st", "sector")]


def point_est(sub, yname, xcols, target, cl=None):
    if cl is None:
        cl = codes_for(sub, FE_GRAD)
    y = absorb(sub[yname].to_numpy(float), cl)
    X = absorb(sub[xcols].to_numpy(float), cl)
    if X[:, xcols.index(target)].std() < 1e-12:
        return np.nan
    b, _ = ols(X, y)
    return b[xcols.index(target)]


def main():
    rng = np.random.default_rng(SEED)
    p = load_pdit()
    g, adopters, never, sy = build(p)
    bea_tab = pd.read_csv(BEA, dtype={"sector": str}).set_index("sector")

    # structures intensity, pre-treatment, as a second tangible measure
    st_raw = load_bea_raw("bea_struct_by_industry.csv")
    pfa_raw = load_bea_raw("bea_pfa_by_industry.csv")
    rows = []
    for lab, sec in BEA2BDS.items():
        s = st_raw[lab].loc[1985:1989].sum()
        f = pfa_raw[lab].loc[1985:1989].sum()
        rows.append({"sector": sec, "struct_int": s / f})
    stru = pd.DataFrame(rows).set_index("sector")["struct_int"]
    stru = (stru / stru.max()).rename("struct_exp")

    ex = pd.DataFrame({"pdit_exp": g, "bea_exp": bea_tab["bea_norm"],
                       "equip_exp": bea_tab["equip_norm"]}).join(stru).dropna()

    # ---------- sector traits from pre-1990 BDS ----------
    ball = load_bds_full()
    pre = ball[(ball.year >= PRE_LO) & (ball.year <= PRE_HI)]
    nat = pre.groupby("year").apply(
        lambda x: x["job_creation"].sum() / x["denom"].sum() * 100,
        include_groups=False).rename("nat_jc")
    traits = []
    for s, gsec in pre.groupby("sector"):
        agg = gsec.groupby("year").apply(
            lambda x: pd.Series({
                "jc": x["job_creation"].sum() / x["denom"].sum() * 100,
                "emp": x["emp"].sum()}), include_groups=False)
        agg = agg.join(nat)
        if len(agg) < 8 or agg["jc"].std() == 0:
            continue
        # cyclical sensitivity: elasticity of sector jc to national jc
        X = np.column_stack([np.ones(len(agg)), agg["nat_jc"].to_numpy()])
        b, _ = ols(X, agg["jc"].to_numpy())
        yrs = agg.index.to_numpy(float)
        lg = np.log(agg["emp"].to_numpy())
        gb, _ = ols(np.column_stack([np.ones(len(yrs)), yrs - yrs.mean()]), lg)
        traits.append({"sector": s, "cyc_beta": b[1], "pre_growth": gb[1] * 100,
                       "jc_sd": agg["jc"].std()})
    tr = pd.DataFrame(traits).set_index("sector")
    ex = ex.join(tr).dropna()

    print("=" * 78)
    print("1. IS EQUIPMENT INTENSITY A PROXY FOR CYCLICAL SENSITIVITY?")
    print("=" * 78)
    print("cyc_beta: elasticity of the sector's job creation rate to the")
    print("national rate, estimated on %d-%d, before any adoption in the"
          % (PRE_LO, PRE_HI))
    print("estimation sample (earliest is MA 1991).\n")
    print("%-8s %8s %8s %9s %10s %8s"
          % ("sector", "equip", "struct", "cyc_beta", "pre_growth", "jc_sd"))
    for s, r in ex.sort_values("equip_exp", ascending=False).iterrows():
        print("%-8s %8.3f %8.3f %9.3f %9.2f%% %8.2f"
              % (s, r["equip_exp"], r["struct_exp"], r["cyc_beta"],
                 r["pre_growth"], r["jc_sd"]))
    print("\n  corr(equip, cyclical sensitivity) = %+.3f"
          % ex.equip_exp.corr(ex.cyc_beta))
    print("  corr(equip, pre-period growth)    = %+.3f"
          % ex.equip_exp.corr(ex.pre_growth))
    print("  corr(equip, jc volatility)        = %+.3f"
          % ex.equip_exp.corr(ex.jc_sd))
    print("  corr(struct, cyclical sensitivity)= %+.3f"
          % ex.struct_exp.corr(ex.cyc_beta))
    print("  corr(equip, struct)               = %+.3f"
          % ex.equip_exp.corr(ex.struct_exp))
    print("\n  a high first correlation supports explanation A (cycle).")

    # ---------- build panel ----------
    adopt_fips = {FIPS[s]: y for s, y in adopters.items()}
    ctrl = {FIPS[s] for s in never}
    on = sy.copy(); on["st"] = on["State"].map(FIPS)
    on = on[["st", "Base Year", "on"]].rename(columns={"Base Year": "year"})
    b = ball[ball["sector"].isin(ex.index) & ball["st"].isin(set(adopt_fips) | ctrl)]
    d = b.merge(on, on=["st", "year"], how="left").join(ex, on="sector")
    d["adopt"] = d["st"].map(adopt_fips)
    d["k"] = np.where(d["adopt"].notna(), d["year"] - d["adopt"], -999)
    d["post"] = d["on"].fillna(0).astype(float)
    dv = d[(d.year >= 1990) & (d.year <= 2015)]
    trt = dv["k"] > -900
    dd = dv[(~trt) | dv["k"].between(BAL_LO, BAL_HI)].copy()
    for nm in ["pdit", "bea", "equip", "struct"]:
        dd["post_" + nm] = dd["post"] * dd[nm + "_exp"]
    dd["post_cyc"] = dd["post"] * dd["cyc_beta"]
    dd["post_grow"] = dd["post"] * dd["pre_growth"]

    # ---------- 2 ----------
    print("\n" + "=" * 78)
    print("2. DOES STRUCTURES INTENSITY DO THE SAME THING?")
    print("=" * 78)
    print("structures is also tangible capital but sits differently across")
    print("sectors. If tangible capital is the operative characteristic, both")
    print("should show it. If only equipment does, equipment is proxying.\n")
    for yn in ["jc_rate", "jd_rate", "entry_rate"]:
        sub = dd.dropna(subset=[yn]).copy()
        sub = sub[np.isfinite(sub[yn].to_numpy(float))]
        cl = codes_for(sub, FE_GRAD)
        print("%s (mean %.2f):" % (yn, sub[yn].mean()))
        for nm in ["equip", "struct"]:
            y = absorb(sub[yn].to_numpy(float), cl)
            X = absorb(sub[["post_" + nm]].to_numpy(float), cl)
            bj, pj, lo, hi = wcb(X, y, sub["st"].to_numpy(), 0, rng)
            print("  %-10s beta=%+.4f  WCB p=%.3f  CI [%+.4f,%+.4f]"
                  % (nm, bj, pj, lo, hi))
        print()

    # ---------- 3 ----------
    print("=" * 78)
    print("3. DOES THE EQUIPMENT GRADIENT SURVIVE CONTROLLING FOR CYCLE?")
    print("=" * 78)
    print("adding post x cyclical_sensitivity and post x pre_growth.\n")
    for yn in ["jc_rate"]:
        sub = dd.dropna(subset=[yn]).copy()
        sub = sub[np.isfinite(sub[yn].to_numpy(float))]
        cl = codes_for(sub, FE_GRAD)
        specs = [(["post_equip"], "equip alone"),
                 (["post_equip", "post_cyc"], "+ cyclical sensitivity"),
                 (["post_equip", "post_grow"], "+ pre-period growth"),
                 (["post_equip", "post_cyc", "post_grow"], "+ both"),
                 (["post_equip", "post_struct"], "+ structures"),
                 (["post_equip", "post_bea", "post_cyc"], "+ intangible + cycle")]
        for cols, lab in specs:
            y = absorb(sub[yn].to_numpy(float), cl)
            X = absorb(sub[cols].to_numpy(float), cl)
            if X[:, 0].std() < 1e-12:
                print("  %-26s (not estimable)" % lab); continue
            bj, pj, lo, hi = wcb(X, y, sub["st"].to_numpy(), 0, rng)
            print("  %-26s beta=%+.4f  WCB p=%.3f  CI [%+.4f,%+.4f]"
                  % (lab, bj, pj, lo, hi))
        # and the cycle term on its own
        y = absorb(sub[yn].to_numpy(float), cl)
        X = absorb(sub[["post_cyc"]].to_numpy(float), cl)
        bj, pj, lo, hi = wcb(X, y, sub["st"].to_numpy(), 0, rng)
        print("  %-26s beta=%+.4f  WCB p=%.3f  CI [%+.4f,%+.4f]"
              % ("cycle alone", bj, pj, lo, hi))

    # ---------- 4 ----------
    print("\n" + "=" * 78)
    print("4. ARE EQUIPMENT-INTENSIVE SECTORS BIGGER IN GENEROUS STATES?")
    print("=" * 78)
    print("if so, the gradient could reflect where the credit is generous")
    print("rather than what equipment intensity does.\n")
    pre_emp = dv[dv.post == 0].groupby(["st", "sector"])["emp"].mean()
    shares = (pre_emp / pre_emp.groupby("st").sum()).rename("share").reset_index()
    shares = shares.join(ex["equip_exp"], on="sector")
    gen = p.groupby("State")["Research and Development Credit"].max()
    gen.index = [FIPS[s] for s in gen.index]
    st_equip = shares.groupby("st").apply(
        lambda x: (x["share"] * x["equip_exp"]).sum(), include_groups=False)
    comb = pd.DataFrame({"equip_weight": st_equip, "max_credit": gen}).dropna()
    print("  corr(state equipment weight, max credit generosity) = %+.3f"
          % comb.equip_weight.corr(comb.max_credit))
    print("  n states = %d" % len(comb))

    # ---------- 5 ----------
    print("\n" + "=" * 78)
    print("5. RANDOMISATION AGAIN, NOW FOR STRUCTURES AND CYCLE")
    print("=" * 78)
    secs = list(ex.index)
    for xname in ["equip_exp", "struct_exp", "cyc_beta"]:
        sub = dd.dropna(subset=["jc_rate"]).copy()
        sub = sub[np.isfinite(sub["jc_rate"].to_numpy(float))]
        cl = codes_for(sub, FE_GRAD)
        yv = absorb(sub["jc_rate"].to_numpy(float), cl)
        postv = sub["post"].to_numpy(float)
        secv = sub["sector"].to_numpy()
        real_x = absorb(postv * sub[xname].to_numpy(float), cl)
        breal, _ = ols(real_x[:, None], yv)
        breal = breal[0]
        vals = ex[xname].to_numpy(float)
        draws = []
        for _ in range(NPERM):
            perm = dict(zip(secs, rng.permutation(vals)))
            xx = absorb(postv * np.array([perm[s] for s in secv]), cl)
            if xx.std() < 1e-12:
                continue
            bb, _ = ols(xx[:, None], yv)
            draws.append(bb[0])
        draws = np.array(draws)
        pv = (np.abs(draws) >= abs(breal)).mean()
        print("  %-12s real=%+.4f  placebo sd=%.4f  p=%.4f"
              % (xname, breal, draws.std(), pv))

    print("\n" + "=" * 78)
    print("READING")
    print("=" * 78)
    print("  equipment survives all controls, structures does nothing")
    print("    -> equipment intensity specifically, and the mechanism is open")
    print("  equipment dies once cycle enters, or cycle alone does the same job")
    print("    -> explanation A, the coefficient was the business cycle")
    print("  both equipment and structures work")
    print("    -> tangible capital broadly, not an equipment story")

    os.makedirs(OUTDIR, exist_ok=True)
    ex.to_csv(os.path.join(OUTDIR, "45_sector_traits.csv"))
    print("\nwrote output\\45_sector_traits.csv")


if __name__ == "__main__":
    main()