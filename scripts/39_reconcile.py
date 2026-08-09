# 39_reconcile.py
# Follow-up to 38. Three problems with reading that table as printed:
#
# 1. SCALE. C's treatment is post x exposure, with exposure normalised so
#    manufacturing = 1 and most sectors below 0.02. beta_C is therefore the
#    effect for a sector as R&D-exposed as manufacturing, not for a typical
#    sector. beta_A and beta_B are the effect of adoption itself. Printing
#    them side by side in script 38 without rescaling was my error.
#
#    If the true model is  y = (a + b*exposure_j)*post + ...
#    then B estimates roughly a + b*E[exposure] and C estimates b, because
#    state-year FE absorbs a*post. So:
#       gradient-attributable part = beta_C * E[exposure]
#       common part                = beta_B - that
#    E[exposure] is employment weighted, otherwise a tiny sector counts the
#    same as manufacturing.
#
# 2. REFERENCE PERIOD. Script 37 showed the gradient result was an artefact
#    of omitting k=-1. A, B and C all omit k=-1 by construction, so the same
#    check runs on all three before the B p=0.052 is believed.
#
# 3. CLUSTERS. Spec A has N=1215 with treatment varying only at state level.
#    Effective cluster count is reported before any p-value from A is used.
#
# Run: py scripts\39_reconcile.py

import os
import numpy as np
import pandas as pd

# reuse loaders, crosswalk, FE machinery and cluster_se from script 38
exec(open(os.path.join("scripts", "38_aggregate_effect.py")).read()
     .split("def main()")[0])

OUTDIR = "output"
REFS = [-1, -2, -3, "avg_pre"]


def wcb(X, y, gid, j, rng, nboot=NBOOT):
    """Override of the version in script 38.

    The original built the restricted model with np.delete, which leaves an
    empty matrix when X has one column (the avg_pre case, where only post
    survives). Falls back to a constant-only restricted model instead, and
    skips bootstrap draws that go singular rather than crashing.
    """
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


def event_terms(d, ref):
    """rebuild pre/post with a chosen omitted period"""
    d = d.copy()
    trt = d["k"] > -900
    if ref == "avg_pre":
        d["pre"] = 0.0                      # whole pre window is the base
    else:
        d["pre"] = ((d["k"] <= -2) & trt & (d["k"] != ref)).astype(float)
    d["post"] = ((d["k"] >= 0) & trt).astype(float)
    return d


def fit(d, yname, fe_specs, treat_cols, rng):
    sub = d.dropna(subset=[yname] + treat_cols).copy()
    sub = sub[np.isfinite(sub[yname].to_numpy(float))]
    if len(sub) < 100:
        return None
    y = sub[yname].to_numpy(float)
    X = sub[treat_cols].to_numpy(float)
    keep = X.std(axis=0) > 1e-12
    if not keep.any():
        return None
    names = [c for c, m in zip(treat_cols, keep) if m]
    X = X[:, keep]
    cl = codes_for(sub, fe_specs)
    y = absorb(y, cl)
    X = absorb(X, cl)
    j = names.index("post") if "post" in names else 0
    try:
        bj, sj, tj, pj, lo, hi = wcb(X, y, sub["st"].to_numpy(), j, rng)
    except np.linalg.LinAlgError:
        return None
    return {"n": len(sub), "beta": bj, "se": sj, "p": pj,
            "lo": lo, "hi": hi, "mean": sub[yname].mean()}


def show(tag, r, scale=1.0, mean=None):
    if r is None:
        print("  %-32s (not estimable)" % tag)
        return
    m = mean if mean is not None else r["mean"]
    b, lo, hi = r["beta"] * scale, r["lo"] * scale, r["hi"] * scale
    if not np.isfinite(lo):
        print("  %-32s beta=%+.4f  p=  n/a  (bootstrap failed)" % (tag, b))
        return
    print("  %-32s beta=%+.4f  p=%.3f  CI [%+.4f, %+.4f]  (%+.1f%%, %+.1f%%)"
          % (tag, b, r["p"], lo, hi, 100 * lo / m, 100 * hi / m))


def main():
    rng = np.random.default_rng(SEED)
    p = load_pdit()
    g, adopters, never = treatment_calendar(p)
    adopt_fips = {FIPS[s]: y for s, y in adopters.items()}
    ctrl = {FIPS[s] for s in never}
    keep_states = set(adopt_fips) | ctrl
    pdit_sectors = set(g.index)

    b = load_bds(pdit_sectors)
    b = b[b["st"].isin(keep_states)]
    d = sector_outcomes(b[b["in_pdit"]])
    d = add_event_time(d, adopt_fips)
    d["exposure"] = d["sector"].map(g)
    d = d.dropna(subset=["exposure"])
    s = add_event_time(state_outcomes(b, False), adopt_fips)

    # ---------- 1. the scale of exposure ----------
    print("=" * 78)
    print("1. WHAT IS THE AVERAGE SECTOR'S EXPOSURE?")
    print("=" * 78)
    print("beta_C is per unit of exposure and exposure is normalised so that")
    print("manufacturing = 1. On its own it is the effect for a sector as")
    print("R&D-exposed as manufacturing, not for a typical sector.\n")

    w = d.groupby("sector")["denom"].sum()
    ex = g.reindex(w.index)
    e_emp = float((ex * w).sum() / w.sum())
    e_eq = float(ex.mean())
    print("  employment-weighted mean exposure : %.4f" % e_emp)
    print("  equal-weighted mean exposure      : %.4f" % e_eq)
    print("\n  exposure and employment share by sector:")
    tab = pd.DataFrame({"exposure": ex, "emp_share": w / w.sum()})
    tab = tab.sort_values("exposure", ascending=False)
    for k, r in tab.head(6).iterrows():
        print("    sector %-6s exposure %.3f   emp share %.3f"
              % (k, r["exposure"], r["emp_share"]))
    print("    (remaining %d sectors all below 0.02 exposure)" % (len(tab) - 6))

    # ---------- 2. rescaled comparison ----------
    print("\n" + "=" * 78)
    print("2. THE THREE ESTIMATES, NOW ON THE SAME SCALE")
    print("=" * 78)
    rows = []
    for yname in ["entry_rate", "exit_rate", "jrr"]:
        sA = event_terms(s, -1)
        dd = event_terms(d, -1)
        dd["post_x"] = dd["post"] * dd["exposure"]
        dd["pre_x"] = dd["pre"] * dd["exposure"]
        dc = dd.rename(columns={"pre": "pre_raw", "post": "post_raw",
                                "pre_x": "pre", "post_x": "post"})

        rA = fit(sA, yname, [("st",), ("year",)], ["pre", "post"], rng)
        rB = fit(dd, yname, [("st", "sector"), ("sector", "year")],
                 ["pre", "post"], rng)
        rC = fit(dc, yname, [("st", "year"), ("sector", "year"),
                             ("st", "sector")], ["pre", "post"], rng)

        mean = rB["mean"] if rB else float("nan")
        print("\n%s  (mean %.2f):" % (yname, mean))
        show("A aggregate, state-year", rA)
        show("B aggregate, panel", rB)
        show("C gradient, per unit exposure", rC)
        show("C rescaled to mean sector", rC, scale=e_emp, mean=mean)

        if rA and rB and rC:
            grad = rC["beta"] * e_emp
            common = rB["beta"] - grad
            print("    decomposition of the aggregate panel effect:")
            print("      total (B)                       %+.4f" % rB["beta"])
            if abs(rB["beta"]) > 1e-6:
                print("      attributable to sector gradient %+.4f  (%.0f%%)"
                      % (grad, 100 * grad / rB["beta"]))
                print("      common across sectors           %+.4f  (%.0f%%)"
                      % (common, 100 * common / rB["beta"]))
                if abs(rB["beta"]) < 0.05:
                    print("      NOTE: total is near zero, so the shares above")
                    print("            are not meaningful. read the levels.")
            else:
                print("      attributable to sector gradient %+.4f" % grad)
                print("      common across sectors           %+.4f" % common)
            rows.append({"outcome": yname, "A": rA["beta"], "B": rB["beta"],
                         "C_raw": rC["beta"], "C_scaled": grad,
                         "common": common, "pA": rA["p"], "pB": rB["p"],
                         "pC": rC["p"], "mean": mean})

    # ---------- 3. reference period, all three specs ----------
    print("\n" + "=" * 78)
    print("3. REFERENCE PERIOD SENSITIVITY (entry_rate)")
    print("=" * 78)
    print("script 37 found the gradient was an artefact of omitting k=-1.")
    print("A and B omit k=-1 too, so the B p=0.052 needs the same check.")
    print("avg_pre is the version that does not depend on any single year.\n")
    print("%-10s %-28s %10s %8s   %s" % ("ref", "spec", "beta", "p", "95% CI"))
    for ref in REFS:
        sA = event_terms(s, ref)
        dd = event_terms(d, ref)
        dd["post_x"] = dd["post"] * dd["exposure"]
        dd["pre_x"] = dd["pre"] * dd["exposure"]
        dc = dd.rename(columns={"pre": "pre_raw", "post": "post_raw",
                                "pre_x": "pre", "post_x": "post"})
        cols = ["pre", "post"] if ref != "avg_pre" else ["post"]
        for tag, dat, fes in [
                ("A aggregate state-year", sA, [("st",), ("year",)]),
                ("B aggregate panel", dd,
                 [("st", "sector"), ("sector", "year")]),
                ("C gradient (raw scale)", dc,
                 [("st", "year"), ("sector", "year"), ("st", "sector")])]:
            r = fit(dat, "entry_rate", fes, cols, rng)
            if r is None:
                print("%-10s %-28s (not estimable)" % (str(ref), tag))
            elif not np.isfinite(r["lo"]):
                print("%-10s %-28s %+10.4f      n/a   (bootstrap failed)"
                      % (str(ref), tag, r["beta"]))
            else:
                print("%-10s %-28s %+10.4f %8.3f   [%+.4f, %+.4f]"
                      % (str(ref), tag, r["beta"], r["p"], r["lo"], r["hi"]))
        print()

    # ---------- 4. cluster diagnostics for the aggregate ----------
    print("=" * 78)
    print("4. HOW MANY CLUSTERS IDENTIFY THE AGGREGATE EFFECT?")
    print("=" * 78)
    sA = event_terms(s, -1)
    sub = sA.dropna(subset=["entry_rate"]).copy()
    sub = sub[np.isfinite(sub["entry_rate"].to_numpy(float))]
    cl = codes_for(sub, [("st",), ("year",)])
    post = absorb(sub["post"].to_numpy(float), cl)
    ss = pd.Series(post ** 2, index=sub["st"]).groupby(level=0).sum()
    ss = (ss / ss.sum()).sort_values(ascending=False)
    print("  N = %d, nominal clusters = %d" % (len(sub), sub.st.nunique()))
    print("  effective clusters ~= %.1f" % (1 / (ss ** 2).sum()))
    print("  top 5 states hold %.3f of residual treatment variance"
          % ss.head(5).sum())
    print("  " + ", ".join("FIPS %d %.3f" % (k, v) for k, v in ss.head(5).items()))

    # ---------- 5. is the aggregate driven by manuf + info? ----------
    print("\n" + "=" * 78)
    print("5. DOES THE AGGREGATE SURVIVE DROPPING MANUF + INFO?")
    print("=" * 78)
    print("if the aggregate really were concentrated in the two high-exposure")
    print("sectors, dropping them should kill it. this is the direct test.\n")
    dd = event_terms(d, -1)
    for tag, sel in [("all 16 sectors", dd),
                     ("drop 31-33 and 51", dd[~dd.sector.isin(["31-33", "51"])]),
                     ("only 31-33 and 51", dd[dd.sector.isin(["31-33", "51"])])]:
        r = fit(sel, "entry_rate", [("st", "sector"), ("sector", "year")],
                ["pre", "post"], rng)
        show(tag, r)

    print("\n" + "=" * 78)
    print("HOW TO READ THIS")
    print("=" * 78)
    print("  the object Lucia asked for is section 2: the aggregate effect and")
    print("  the sector gradient on a common scale, with the aggregate split")
    print("  into a gradient-attributable part and a part common to all")
    print("  sectors. section 3 decides whether any of it survives dropping")
    print("  the single-year reference. nothing is compared to the FGS 20%:")
    print("  different outcome data, and theirs is cumulative over ten years.")

    if rows:
        os.makedirs(OUTDIR, exist_ok=True)
        pd.DataFrame(rows).to_csv(
            os.path.join(OUTDIR, "39_reconciled.csv"), index=False)
        print("\nwrote output\\39_reconciled.csv")


if __name__ == "__main__":
    main()