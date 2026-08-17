# 51_audit_and_window.py
# External review raised five points. Three are accepted, one is overstated,
# one needs a different fix than proposed.
#
# ERROR #15 (accepted, and the reason this script exists). The preferred
# window k in [-3,+2] was justified by "17 of 20 adopters with complete
# event-time coverage". But k=+3 has 20 of 20 coverage, so k in [-3,+3] has
# the SAME 17 of 20 complete adopters. The coverage criterion does not imply
# the +2 endpoint. Script 50 stepped through windows symmetrically --
# (-5,4), (-4,3), (-3,2), (-2,1) -- and never tried an asymmetric one. Since
# the window was chosen after seeing the full-window result, this matters.
# Section 1 runs every window consistent with the stated criterion.
#
# 5.1 DECOMPOSITION (accepted, being removed). The handoff wrote
# beta_B = gamma * E[g] + common, with E[g] = 0.178 the employment-weighted
# mean exposure, giving "15% / 85%". Spec B is unweighted FE OLS, so its
# implicit weights come from FWL-residualised treatment variation, not from
# employment. B and C also use different projections (B omits state-year FE).
# The identity 0.253 = 0.298 - 0.045 holds by construction and carries no
# independent meaning. Section 3 computes the implicit weights so the correct
# statement can be made.
#
# CALLAWAY-SANT'ANNA (accepted). C&S assumes treatment is absorbing. Four
# states repeal. The handoff's claim that C&S would resolve the cohort split
# is wrong; the relevant framework is de Chaisemartin-D'Haultfoeuille for
# non-absorbing treatment. No code change, a documentation fix.
#
# BOOTSTRAP DRAWS (partly rejected). 999 draws give a Monte Carlo SE of about
# 0.0058 at p=0.035, so p sits 1.5 MC standard errors from 0.05. More draws
# improve precision but will not flip the conclusion. Section 4 raises the
# headline to 9999 anyway, since it is the final number.
#
# ABSORB CONVERGENCE (accepted, different fix). The review asked for a
# convergence warning. The stronger check is orthogonality: after absorption,
# every fixed-effect group mean of the residual should be ~0. Section 2 tests
# that directly, which is sufficient regardless of iteration count.
#
# COHORT LANGUAGE (accepted). "I lean toward power" is not supportable without
# a direct test of beta_early = beta_late. Section 5 runs it.
#
# Run: py scripts\51_audit_and_window.py

import math
import os
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
PDIT = os.path.join(RAW, "export.csv")
BDS = os.path.join(RAW, "bds2023_st_sec.csv")
OUTDIR = "output"

RATIO = "Discounted - 12% Rate"
NBOOT = 999
NBOOT_FINAL = 9999
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


def absorb(M, cl, tol=1e-9, maxiter=300, report=False):
    A = np.asarray(M, dtype=float).copy()
    one = A.ndim == 1
    if one:
        A = A[:, None]
    it, chg = 0, np.nan
    for it in range(1, maxiter + 1):
        prev = A.copy()
        for codes, ng in cl:
            for j in range(A.shape[1]):
                A[:, j] = demean_by(A[:, j], codes, ng)
        chg = np.max(np.abs(A - prev))
        if chg < tol:
            break
    out = A[:, 0] if one else A
    if report:
        # orthogonality check: group means of the residual should be ~0
        worst = []
        for codes, ng in cl:
            col = A[:, 0]
            s = np.bincount(codes, weights=col, minlength=ng)
            c = np.bincount(codes, minlength=ng).astype(float)
            m = np.divide(s, c, out=np.zeros(ng), where=c > 0)
            worst.append(np.max(np.abs(m)))
        return out, {"iters": it, "last_change": chg, "max_group_mean": worst,
                     "converged": chg < tol}
    return out


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
        return beta[j], np.nan, np.nan, np.nan, np.nan, np.nan
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
        return beta[j], se[j], np.nan, np.nan, np.nan, len(ts)
    pv = (np.abs(ts) >= abs(t0)).mean()
    lo_t, hi_t = np.quantile(ts, [0.025, 0.975])
    mcse = np.sqrt(pv * (1 - pv) / len(ts))
    return beta[j], se[j], pv, beta[j] - hi_t * se[j], beta[j] - lo_t * se[j], mcse


def chi2_p(stat, df):
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


FE = [("st", "sector"), ("sector", "year")]


def est(d, yname, rng, lo, hi, nboot=NBOOT, states=None):
    s = d.copy()
    if states is not None:
        s = s[s["st"].isin(states) | (s["k"] <= -900)]
    trt = s["k"] > -900
    s = s[(~trt) | s["k"].between(lo, hi)]
    s = s.dropna(subset=[yname])
    s = s[np.isfinite(s[yname].to_numpy(float))]
    if len(s) < 300 or s["post_on"].std() < 1e-12:
        return None
    cl = codes_for(s, FE)
    y = absorb(s[yname].to_numpy(float), cl)
    X = absorb(s[["post_on"]].to_numpy(float), cl)
    b, se, pv, clo, chi, mc = wcb(X, y, s["st"].to_numpy(), 0, rng, nboot)
    return {"n": len(s), "beta": b, "se": se, "p": pv, "lo": clo, "hi": chi,
            "mcse": mc, "mean": s[yname].mean()}


def main():
    rng = np.random.default_rng(SEED)
    p = load_pdit()
    rd = "Research and Development Credit"
    sy = p.groupby(["State", "Base Year"])[rd].max().reset_index()
    sy["on"] = (sy[rd] > 0).astype(int)
    first = sy[sy.on == 1].groupby("State")["Base Year"].min()
    adopters = first[first > 1990]
    never = sorted(set(p["State"]) - set(first.index))
    adopt_fips = {FIPS[s]: y for s, y in adopters.items()}
    ctrl = {FIPS[s] for s in never}
    on = sy.copy(); on["st"] = on["State"].map(FIPS)
    on = on[["st", "Base Year", "on"]].rename(
        columns={"Base Year": "year", "on": "on_status"})

    b = load_bds(set(p["sector"].unique()))
    b = b[b["st"].isin(set(adopt_fips) | ctrl)]
    d = b.merge(on, on=["st", "year"], how="left")
    d["on_status"] = d["on_status"].fillna(0).astype(int)
    d["adopt"] = d["st"].map(adopt_fips)
    d["k"] = np.where(d["adopt"].notna(), d["year"] - d["adopt"], -999)
    d["post_on"] = d["on_status"].astype(float)
    dv = d[(d.year >= 1990) & (d.year <= 2015)].copy()

    # ---------- 1 ----------
    print("=" * 78)
    print("1. ERROR #15: EVERY WINDOW THE COVERAGE CRITERION ALLOWS")
    print("=" * 78)
    print("the stated rule was 'adopters with complete event-time coverage'.")
    print("script 50 only tried symmetric contractions, so k in [-3,+3] was")
    print("never run even though k=+3 has 20/20 coverage and [-3,+3] therefore")
    print("has the same 17/20 complete adopters as [-3,+2].\n")

    def complete(lo, hi):
        miss = set()
        for k in range(lo, hi + 1):
            miss |= {s for s, a in adopters.items() if not (1990 <= a + k <= 2015)}
        return len(adopters) - len(miss)

    grid = []
    for lo in range(-5, 0):
        for hi in range(0, 5):
            grid.append((lo, hi))
    rows = []
    print("%-12s %10s %7s %10s %8s %22s" %
          ("window", "complete", "N", "beta", "p", "95% CI"))
    for lo, hi in grid:
        r = est(dv, "entry_rate", rng, lo, hi)
        if r is None:
            continue
        c = complete(lo, hi)
        star = " *" if np.isfinite(r["p"]) and r["p"] < 0.05 else ""
        print("k[%+d,%+d]%s %8d/%d %7d %+10.4f %8.3f  [%+.4f,%+.4f]%s"
              % (lo, hi, "  ", c, len(adopters), r["n"], r["beta"], r["p"],
                 r["lo"], r["hi"], star))
        rows.append({"lo": lo, "hi": hi, "complete": c, "n": r["n"],
                     "beta": r["beta"], "p": r["p"], "lo_ci": r["lo"],
                     "hi_ci": r["hi"]})
    tab = pd.DataFrame(rows)
    print("\n  windows with 17/20 complete coverage:")
    sub17 = tab[tab.complete == 17]
    for _, r in sub17.iterrows():
        print("    k[%+d,%+d]  beta=%+.4f  p=%.3f"
              % (r["lo"], r["hi"], r["beta"], r["p"]))
    print("\n  share of all %d windows with p<0.05: %.2f"
          % (len(tab), (tab.p < 0.05).mean()))
    print("  beta range across all windows: %+.4f to %+.4f"
          % (tab.beta.min(), tab.beta.max()))
    print("  sign flips: %d" % (tab.beta < 0).sum())

    # ---------- 2 ----------
    print("\n" + "=" * 78)
    print("2. DOES absorb() ACTUALLY CONVERGE?")
    print("=" * 78)
    print("the review asked for a convergence warning. The stronger test is")
    print("orthogonality: after absorption every fixed-effect group mean of")
    print("the residual should be ~0, regardless of iteration count.\n")
    for lo, hi in [(-3, 2), (-3, 3), (-5, 4)]:
        s = dv[(dv["k"] <= -900) | dv["k"].between(lo, hi)]
        s = s.dropna(subset=["entry_rate"])
        s = s[np.isfinite(s["entry_rate"].to_numpy(float))]
        for fes, lab in [(FE, "2-way (st-sec, sec-yr)"),
                         ([("st", "year"), ("sector", "year"), ("st", "sector")],
                          "3-way")]:
            cl = codes_for(s, fes)
            _, info = absorb(s["entry_rate"].to_numpy(float), cl, report=True)
            print("  k[%+d,%+d] %-24s iters=%3d  last change=%.2e  converged=%s"
                  % (lo, hi, lab, info["iters"], info["last_change"],
                     info["converged"]))
            print("      max |group mean| after absorption: %s"
                  % "  ".join("%.2e" % x for x in info["max_group_mean"]))

    # ---------- 3 ----------
    print("\n" + "=" * 78)
    print("3. THE IMPLICIT WEIGHTS BEHIND SPEC B")
    print("=" * 78)
    print("the handoff wrote beta_B = gamma * E[g] + common with E[g] = 0.178,")
    print("the employment-weighted mean exposure. Spec B is unweighted FE OLS,")
    print("so its implicit weights come from residualised treatment variation.")
    print("Those weights are computed here so the right number can be quoted.\n")
    s = dv[(dv["k"] <= -900) | dv["k"].between(-3, 2)]
    s = s.dropna(subset=["entry_rate"]).copy()
    s = s[np.isfinite(s["entry_rate"].to_numpy(float))]
    cl = codes_for(s, FE)
    dres = absorb(s["post_on"].to_numpy(float), cl)
    w = pd.Series(dres ** 2, index=s["sector"]).groupby(level=0).sum()
    w = w / w.sum()
    emp = s.groupby("sector")["denom"].sum()
    emp = emp / emp.sum()

    live = p[p.groupby(["State", "Base Year"])[rd].transform("max") > 0]
    va = "Industry Value-Added"
    g = ((live[rd] * live[va]).groupby(live["sector"]).sum()
         / live[va].groupby(live["sector"]).sum())
    g = g / g.max()
    comp = pd.DataFrame({"exposure": g, "emp_weight": emp,
                         "implicit_weight": w}).dropna()
    print("%-8s %10s %12s %16s" %
          ("sector", "exposure", "emp weight", "implicit weight"))
    for sec, r in comp.sort_values("implicit_weight", ascending=False).iterrows():
        print("%-8s %10.3f %12.3f %16.3f"
              % (sec, r["exposure"], r["emp_weight"], r["implicit_weight"]))
    e_emp = float((comp.exposure * comp.emp_weight).sum())
    e_imp = float((comp.exposure * comp.implicit_weight).sum())
    print("\n  employment-weighted mean exposure  = %.4f   (used in the handoff)"
          % e_emp)
    print("  implicitly-weighted mean exposure  = %.4f   (what spec B uses)"
          % e_imp)
    print("\n  the two differ, so the 15%%/85%% split in section 5.1 of the")
    print("  handoff is not a decomposition. Report only: evaluating the")
    print("  gradient at mean exposure gives gamma * %.4f." % e_imp)

    # ---------- 4 ----------
    print("\n" + "=" * 78)
    print("4. HEADLINE WITH %d BOOTSTRAP DRAWS" % NBOOT_FINAL)
    print("=" * 78)
    print("999 draws give a Monte Carlo SE of about 0.006 at p=0.035, so the")
    print("conclusion will not flip, but this is the final number.\n")
    for lo, hi in [(-3, 2), (-3, 3)]:
        r = est(dv, "entry_rate", rng, lo, hi, nboot=NBOOT_FINAL)
        print("  k[%+d,%+d]  beta=%+.4f  se=%.4f  p=%.4f (MC SE %.4f)  CI [%+.4f,%+.4f]"
              % (lo, hi, r["beta"], r["se"], r["p"], r["mcse"], r["lo"], r["hi"]))

    # ---------- 5 ----------
    print("\n" + "=" * 78)
    print("5. DIRECT TEST OF COHORT EQUALITY")
    print("=" * 78)
    print("one estimate significant and one not does not establish that they")
    print("differ. H0: beta_early = beta_late, tested by interacting treatment")
    print("with a late-cohort indicator.\n")
    early = {FIPS[s] for s, a in adopters.items() if a <= 1998}
    late = {FIPS[s] for s, a in adopters.items() if a >= 2000}
    for lo, hi in [(-3, 2), (-3, 3), (-5, 4)]:
        s = dv[(dv["k"] <= -900) | dv["k"].between(lo, hi)].copy()
        s = s.dropna(subset=["entry_rate"])
        s = s[np.isfinite(s["entry_rate"].to_numpy(float))]
        s["late"] = s["st"].isin(late).astype(float)
        s["post_late"] = s["post_on"] * s["late"]
        cl = codes_for(s, FE)
        y = absorb(s["entry_rate"].to_numpy(float), cl)
        X = absorb(s[["post_on", "post_late"]].to_numpy(float), cl)
        bb, sse, pv, clo, chi, mc = wcb(X, y, s["st"].to_numpy(), 1, rng)
        print("  k[%+d,%+d]  late-minus-early = %+.4f  WCB p=%.3f  CI [%+.4f,%+.4f]"
              % (lo, hi, bb, pv, clo, chi))
    print("\n  if this interaction is not significant, the paper must say the")
    print("  data do not establish cohort heterogeneity, not that the split is")
    print("  'probably power'.")

    os.makedirs(OUTDIR, exist_ok=True)
    tab.to_csv(os.path.join(OUTDIR, "51_window_grid.csv"), index=False)
    print("\nwrote output\\51_window_grid.csv")


if __name__ == "__main__":
    main()