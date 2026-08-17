# 46_equip_final.py
# Final check on the equipment gradient before writing it up.
#
# WHERE THINGS STAND. Script 45 killed the cycle explanation: equipment
# intensity correlates only +0.219 with pre-1990 cyclical sensitivity, -0.285
# with pre-period growth, and +0.157 with state credit generosity. Controlling
# for all of them leaves the coefficient at +1.65 with p=0.005.
#
# But it also broke the "tangible capital" reading. Structures intensity, also
# tangible, comes out with the OPPOSITE sign (-1.19, p=0.030), and equip and
# struct correlate -0.635 across sectors. Real estate is struct 1.000 /
# equip 0.092; construction is struct 0.166 / equip 0.962. So these are two
# ends of one dimension, and that dimension is capital COMPOSITION, not
# capital intensity.
#
# TWO THINGS THIS SCRIPT DOES.
#
# 1. Tests composition directly: post x equip_share, where
#    equip_share = equip / (equip + struct). Script 45 ran equip and struct
#    separately and once together, but never their ratio, which is the
#    variable the pattern implies. That was an oversight in its design.
#
# 2. THE TEST THAT SHOULD HAVE COME FIRST. Assign fake adoption years to the
#    never-treated states and to shuffled versions of the real ones, then
#    re-estimate. If the same gradient appears under fake treatment, it is a
#    pre-existing differential trend across sectors, not a policy effect.
#    Everything so far has conditioned on the real timing being meaningful.
#
# Run: py scripts\46_equip_final.py

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
NFAKE = 1000
NBOOT = 999
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
    lines = open(os.path.join(RAW, fname), encoding="utf-8",
                 errors="replace").read().split("\n")
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


def load_bds():
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
    b["jc_rate"] = b["job_creation"] / b["denom"] * 100
    b["jd_rate"] = b["job_destruction"] / b["denom"] * 100
    return b


FE = [("st", "year"), ("sector", "year"), ("st", "sector")]


def main():
    rng = np.random.default_rng(SEED)
    p = load_pdit()
    g, adopters, never, sy = build(p)
    bea_tab = pd.read_csv(BEA, dtype={"sector": str}).set_index("sector")

    st_raw = load_bea_raw("bea_struct_by_industry.csv")
    pfa_raw = load_bea_raw("bea_pfa_by_industry.csv")
    eq_raw = load_bea_raw("bea_equip_by_industry.csv")
    rows = []
    for lab, sec in BEA2BDS.items():
        s = st_raw[lab].loc[1985:1989].sum()
        e = eq_raw[lab].loc[1985:1989].sum()
        f = pfa_raw[lab].loc[1985:1989].sum()
        rows.append({"sector": sec, "struct_raw": s / f, "equip_raw": e / f,
                     "equip_share": e / (e + s)})
    tan = pd.DataFrame(rows).set_index("sector")

    ex = pd.DataFrame({"pdit_exp": g, "bea_exp": bea_tab["bea_norm"],
                       "equip_exp": bea_tab["equip_norm"]}).join(tan).dropna()
    ex["struct_exp"] = ex["struct_raw"] / ex["struct_raw"].max()
    ex["eqshare_n"] = ex["equip_share"] / ex["equip_share"].max()

    print("=" * 78)
    print("1. CAPITAL COMPOSITION")
    print("=" * 78)
    print("equip and struct correlate %+.3f across sectors, so they are two"
          % ex.equip_exp.corr(ex.struct_exp))
    print("ends of one dimension. equip_share = equip/(equip+struct) states")
    print("that dimension directly.\n")
    print("%-8s %8s %8s %11s" % ("sector", "equip", "struct", "equip_share"))
    for s, r in ex.sort_values("equip_share", ascending=False).iterrows():
        print("%-8s %8.3f %8.3f %11.3f"
              % (s, r["equip_exp"], r["struct_exp"], r["equip_share"]))

    adopt_fips = {FIPS[s]: y for s, y in adopters.items()}
    ctrl = {FIPS[s] for s in never}
    on = sy.copy(); on["st"] = on["State"].map(FIPS)
    on = on[["st", "Base Year", "on"]].rename(columns={"Base Year": "year"})

    b = load_bds()
    b = b[b["sector"].isin(ex.index) & b["st"].isin(set(adopt_fips) | ctrl)]
    d = b.merge(on, on=["st", "year"], how="left").join(ex, on="sector")
    d["adopt"] = d["st"].map(adopt_fips)
    d["k"] = np.where(d["adopt"].notna(), d["year"] - d["adopt"], -999)
    d["post"] = d["on"].fillna(0).astype(float)
    dv = d[(d.year >= 1990) & (d.year <= 2015)].copy()
    trt = dv["k"] > -900
    dd = dv[(~trt) | dv["k"].between(BAL_LO, BAL_HI)].copy()
    for nm in ["equip_exp", "struct_exp", "eqshare_n", "bea_exp", "pdit_exp"]:
        dd["p_" + nm] = dd["post"] * dd[nm]

    print("\n" + "=" * 78)
    print("2. DOES COMPOSITION BEAT LEVEL?")
    print("=" * 78)
    print("if composition is the operative variable, equip_share should be at")
    print("least as strong as equip alone.\n")
    for yn in ["jc_rate", "jd_rate", "entry_rate"]:
        sub = dd.dropna(subset=[yn]).copy()
        sub = sub[np.isfinite(sub[yn].to_numpy(float))]
        cl = codes_for(sub, FE)
        y = absorb(sub[yn].to_numpy(float), cl)
        print("%s (mean %.2f):" % (yn, sub[yn].mean()))
        for nm, lab in [("p_equip_exp", "equipment level"),
                        ("p_struct_exp", "structures level"),
                        ("p_eqshare_n", "equipment SHARE")]:
            X = absorb(sub[[nm]].to_numpy(float), cl)
            bj, pj, lo, hi = wcb(X, y, sub["st"].to_numpy(), 0, rng)
            print("  %-20s beta=%+.4f  WCB p=%.3f  CI [%+.4f,%+.4f]"
                  % (lab, bj, pj, lo, hi))
        print()

    print("=" * 78)
    print("3. THE TEST THAT SHOULD HAVE COME FIRST: FAKE ADOPTION YEARS")
    print("=" * 78)
    print("everything so far conditioned on the real adoption timing being")
    print("meaningful. Here every state, treated or not, gets a random")
    print("adoption year drawn from the observed 1991-2012 range, and the")
    print("gradient is re-estimated. %d draws.\n" % NFAKE)
    real_years = sorted(adopters.values)
    all_states = sorted(set(dv["st"].unique()))

    for yn, xcol in [("jc_rate", "eqshare_n"), ("jc_rate", "equip_exp")]:
        base = dv.dropna(subset=[yn]).copy()
        base = base[np.isfinite(base[yn].to_numpy(float))]
        # real estimate on the same balanced-window construction
        sub = dd.dropna(subset=[yn]).copy()
        sub = sub[np.isfinite(sub[yn].to_numpy(float))]
        cl = codes_for(sub, FE)
        yv = absorb(sub[yn].to_numpy(float), cl)
        X = absorb(sub[["p_" + xcol]].to_numpy(float), cl)
        breal, _ = ols(X, yv)
        breal = breal[0]

        exv = base[xcol].to_numpy(float)
        yrv = base["year"].to_numpy()
        stv = base["st"].to_numpy()
        draws = []
        for _ in range(NFAKE):
            fake = {s: rng.choice(real_years) for s in all_states}
            kk = np.array([yrv[i] - fake[stv[i]] for i in range(len(base))])
            keep = (kk >= BAL_LO) & (kk <= BAL_HI)
            if keep.sum() < 1000:
                continue
            s2 = base[keep].copy()
            postf = (kk[keep] >= 0).astype(float)
            cl2 = codes_for(s2, FE)
            y2 = absorb(s2[yn].to_numpy(float), cl2)
            x2 = absorb(postf * exv[keep], cl2)
            if x2.std() < 1e-12:
                continue
            bb, _ = ols(x2[:, None], y2)
            draws.append(bb[0])
        draws = np.array(draws)
        pv = (np.abs(draws) >= abs(breal)).mean()
        print("%s, exposure = %s:" % (yn, xcol))
        print("  real (true timing)     = %+.4f" % breal)
        print("  fake-timing mean       = %+.4f" % draws.mean())
        print("  fake-timing sd         = %.4f" % draws.std())
        print("  fake 2.5 / 97.5 pct    = %+.4f / %+.4f"
              % tuple(np.quantile(draws, [0.025, 0.975])))
        print("  share |fake| >= |real| = %.4f  (n draws %d)" % (pv, len(draws)))
        if pv > 0.10:
            print("  -> the gradient appears under fake timing too. It is a")
            print("     pre-existing differential trend, not a policy effect.")
        else:
            print("  -> does not appear under fake timing. The pattern is tied")
            print("     to the actual adoption dates.")
        print()

    print("=" * 78)
    print("4. EVENT-TIME SHAPE")
    print("=" * 78)
    print("if adjustment costs drive this (equipment is quick to buy, buildings")
    print("are slow), the equipment gradient should peak early and fade.\n")
    sub = dv.dropna(subset=["jc_rate"]).copy()
    sub = sub[np.isfinite(sub["jc_rate"].to_numpy(float))]
    trt2 = sub["k"] > -900
    ks = [k for k in range(-4, 7) if k != -1]
    cols = []
    kept = []
    e = sub["eqshare_n"].to_numpy(float)
    for k in ks:
        ind = ((sub["k"] == k) & trt2).to_numpy(float)
        v = ind * e
        if v.std() > 1e-12:
            cols.append(v); kept.append(k)
    X = np.column_stack(cols)
    cl = codes_for(sub, FE)
    y = absorb(sub["jc_rate"].to_numpy(float), cl)
    X = absorb(X, cl)
    beta, inv = ols(X, y)
    V = cluster_vcv(X, y - X @ beta, inv, sub["st"].to_numpy())
    se = np.sqrt(np.diag(V))
    print("%5s %10s %10s %8s" % ("k", "beta", "se", "t"))
    for k, bb, ss in zip(kept, beta, se):
        bar = "#" * min(int(abs(bb) / 0.3), 28)
        print("%5d %10.4f %10.4f %8.2f  %s%s"
              % (k, bb, ss, bb / ss if ss > 0 else np.nan, bar,
                 "  <- pre" if k < -1 else ""))
    pre_idx = [i for i, k in enumerate(kept) if k < -1]
    if pre_idx:
        bp = beta[pre_idx]; Vp = V[np.ix_(pre_idx, pre_idx)]
        try:
            stat = float(bp @ np.linalg.solve(Vp, bp))
            print("\n  joint Wald on %d leads: chi2 = %.2f" % (len(pre_idx), stat))
        except np.linalg.LinAlgError:
            pass

    os.makedirs(OUTDIR, exist_ok=True)
    ex.to_csv(os.path.join(OUTDIR, "46_capital_composition.csv"))
    print("\nwrote output\\46_capital_composition.csv")


if __name__ == "__main__":
    main()