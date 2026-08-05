# 30_variance_diagnostic.py
# Phase 2 step 1: can the state x sector x year panel support
# state-year + sector-year + state-sector FE?
# No policy involved. Only asks how much variation survives the FE.
# Run: py 30_variance_diagnostic.py

import os
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
BDS = os.path.join(RAW, "bds2023_st_sec.csv")
OUTDIR = "output"

VALID_SECTORS = ["11", "21", "22", "23", "31-33", "42", "44-45", "48-49",
                 "51", "52", "53", "54", "55", "56", "61", "62", "71",
                 "72", "81"]

WINDOWS = {"full_1978_2023": (1978, 2023), "phase1_1997_2023": (1997, 2023)}


def load():
    if not os.path.exists(BDS):
        raise SystemExit("missing file: " + BDS)
    df = pd.read_csv(BDS, dtype={"st": str, "sector": str}, low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]

    # BDS suppresses cells with D / S / X. force numeric, those become NaN.
    for c in df.columns:
        if c not in ("st", "sector"):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["sector"] = df["sector"].str.strip()
    df["st"] = df["st"].str.strip().str.zfill(2)
    df = df[df["sector"].isin(VALID_SECTORS)].copy()
    return df


def build_outcomes(df):
    # JRR the Phase 1 way: (JC + JD) / denom * 100.
    # NOT the bds reallocation_rate column, which is the excess rate.
    # NOTE includes age-0 firms -- st_sec has no age dimension.
    df["jrr"] = (df["job_creation"] + df["job_destruction"]) / df["denom"] * 100
    df["entry_rate"] = df["estabs_entry_rate"]
    df["exit_rate"] = df["estabs_exit_rate"]
    return df


def demean_by(vals, codes, ngroups):
    sums = np.bincount(codes, weights=vals, minlength=ngroups)
    cnts = np.bincount(codes, minlength=ngroups).astype(float)
    means = np.divide(sums, cnts, out=np.zeros(ngroups), where=cnts > 0)
    return vals - means[codes]


def absorb(vals, code_list, tol=1e-10, maxiter=500):
    """alternating projections to sweep out multiple FE"""
    r = vals.astype(float).copy()
    if len(code_list) == 1:
        return demean_by(r, code_list[0][0], code_list[0][1])
    for _ in range(maxiter):
        prev = r.copy()
        for codes, ng in code_list:
            r = demean_by(r, codes, ng)
        if np.max(np.abs(r - prev)) < tol:
            break
    return r


def n_params(code_list):
    return sum(ng for _, ng in code_list) - (len(code_list) - 1)


def run_window(df, label, y0, y1, outcomes):
    d = df[(df["year"] >= y0) & (df["year"] <= y1)].copy()

    print("")
    print("=" * 70)
    print("WINDOW %s   years %d-%d" % (label, y0, y1))
    print("=" * 70)
    print("raw rows: %d | states: %d | sectors: %d | years: %d"
          % (len(d), d["st"].nunique(), d["sector"].nunique(), d["year"].nunique()))

    if d["denom"].notna().any():
        q = d["denom"].quantile([.01, .05, .25, .50, .75]).round(0)
        print("denom pctiles p1/p5/p25/p50/p75: %s" % list(q.values))
        print("share of cells with denom < 1000: %.3f" % (d["denom"] < 1000).mean())

    rows = []
    for yname in outcomes:
        sub = d[["st", "sector", "year", yname, "denom"]].dropna(subset=[yname]).copy()
        sub = sub[np.isfinite(sub[yname])]
        n = len(sub)
        if n < 500:
            print("\n%s: too few usable rows (%d), skipping" % (yname, n))
            continue

        y = sub[yname].to_numpy(dtype=float)

        st_c, st_u = pd.factorize(sub["st"])
        yr_c, yr_u = pd.factorize(sub["year"])
        sy_c, sy_u = pd.factorize(sub["st"] + "_" + sub["year"].astype(str))
        jy_c, jy_u = pd.factorize(sub["sector"] + "_" + sub["year"].astype(str))
        sj_c, sj_u = pd.factorize(sub["st"] + "_" + sub["sector"])

        YR = (yr_c, len(yr_u))
        SY = (sy_c, len(sy_u))
        JY = (jy_c, len(jy_u))
        SJ = (sj_c, len(sj_u))

        specs = [
            ("0 none (raw)",                 []),
            ("1 year",                       [YR]),
            ("2 state-year",                 [SY]),
            ("3 sector-year",                [JY]),
            ("4 state-year + sector-year",   [SY, JY]),
            ("5 + state-sector  <-TARGET",   [SY, JY, SJ]),
        ]

        print("\n--- %s ---  N = %d, mean = %.3f, sd = %.3f"
              % (yname, n, y.mean(), y.std(ddof=1)))
        print("%-32s %8s %8s %9s %9s"
              % ("spec", "sd_res", "var_kept", "k_params", "sd_adj"))

        v_tot = None
        for name, cl in specs:
            if not cl:
                r = y - y.mean()
                k = 1
            else:
                r = absorb(y, cl)
                k = n_params(cl)
            rss = float(np.sum(r ** 2))
            var_raw = rss / n
            if v_tot is None:
                v_tot = var_raw
            dfree = max(n - k, 1)
            var_adj = rss / dfree
            print("%-32s %8.4f %8.3f %9d %9.4f"
                  % (name, np.sqrt(var_raw), var_raw / v_tot, k, np.sqrt(var_adj)))

            if name.startswith("5"):
                rows.append({
                    "window": label, "outcome": yname, "N": n,
                    "sd_raw": y.std(ddof=1), "sd_resid": np.sqrt(var_raw),
                    "sd_resid_adj": np.sqrt(var_adj),
                    "var_share_kept": var_raw / v_tot,
                    "k_params": k, "df_left": dfree,
                    "n_states": len(st_u),
                })
                print("    -> SE(beta) ~= %.4f / sd_resid(D)  [iid, no cluster]"
                      % (np.sqrt(var_adj) / np.sqrt(n)))
                print("    -> clustering on %d states inflates this; budget 2-3x"
                      % len(st_u))

        g = sub.groupby(["sector", "year"])[yname]
        cs = g.std(ddof=1)
        print("    cross-state sd within sector-year: mean %.4f, median %.4f"
              % (cs.mean(), cs.median()))
        print("    (overall sd for comparison: %.4f)" % y.std(ddof=1))

    return rows


def main():
    df = load()
    df = build_outcomes(df)
    outcomes = ["entry_rate", "jrr", "exit_rate"]

    print("loaded %d rows, years %d-%d"
          % (len(df), df["year"].min(), df["year"].max()))
    for c in outcomes:
        print("  %s missing share: %.4f" % (c, df[c].isna().mean()))

    allrows = []
    for label, (y0, y1) in WINDOWS.items():
        allrows += run_window(df, label, y0, y1, outcomes)

    if allrows:
        os.makedirs(OUTDIR, exist_ok=True)
        out = pd.DataFrame(allrows)
        p = os.path.join(OUTDIR, "30_variance_diagnostic.csv")
        out.to_csv(p, index=False)
        print("\nwrote %s" % p)


if __name__ == "__main__":
    main()