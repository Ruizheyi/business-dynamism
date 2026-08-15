# 31_build_pdit_panel.py
# Phase 2 step 2: merge PDIT incentives onto the BDS state x sector panel,
# then ask how much variation the TREATMENT has left after the target FE.
# Step 1 (script 30) showed the outcome keeps ~27% of variance. That was
# half the power question. This is the other half.
#
# PDIT notes (from the Upjohn "About the Data" page):
#   - values are pct of that industry's value added
#   - 12% version = PV of incentives / PV of value added, 12% real discount,
#     over a 20-year simulation starting in the base year
#   - the simulation ASSUMES the tax/incentive law stays fixed for 20 years,
#     so this is a policy STANCE measure, not a flow of money in year t
#   - 33 states, 45 industries, 1990-2015
#
# Run: py scripts\31_build_pdit_panel.py

import os
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
PDIT = os.path.join(RAW, "export.csv")
BDS = os.path.join(RAW, "bds2023_st_sec.csv")
OUTDIR = "output"

RATIO_MAIN = "Discounted - 12% Rate"
TREAT = "Research and Development Credit"
OTHER_INCENTIVES = ["Investment Tax Credit", "Job Creation Tax Credit",
                    "Property Tax Abatement", "Corporate Income Tax"]

# PDIT industry -> BDS 2-digit sector.
# 45 industries map onto 16 BDS sectors. BDS 11 (agriculture), 21 (mining)
# and 22 (utilities) have NO PDIT coverage and get dropped.
XWALK = {
    "Construction": "23",
    # --- manufacturing, all 19 collapse into 31-33 ---
    "Food, beverage, and tobacco Manufacturing": "31-33",
    "Textile Mills and textile product mills": "31-33",
    "Apparel, Leather and allied product Manufacturing": "31-33",
    "Wood Product Manufacturing": "31-33",
    "Paper Manufacturing": "31-33",
    "Printing and related support activities": "31-33",
    "Petroleum and coal products manufacturing": "31-33",
    "Chemical Manufacturing": "31-33",
    "Plastics and Rubber products manufacturing": "31-33",
    "Nonmetallic mineral product manufacturing": "31-33",
    "Primary Metal Manufacturing": "31-33",
    "Fabricated Metal Product Manufacturing": "31-33",
    "Machinery manufacturing": "31-33",
    "Computer and electronic product manufacturing": "31-33",
    "Electrical equipment, appliance, and component manufacturing": "31-33",
    "Motor Vehicles, bodies and trailers, and parts": "31-33",
    "Other Transportation Equipment": "31-33",
    "furniture and related product manufacturing": "31-33",
    "miscellaneous manufacturing": "31-33",
    # --- trade / transport ---
    "Wholesale Trade": "42",
    "Retail Trade": "44-45",
    "warehousing and storeage": "48-49",
    # --- information ---
    "Broadcasting and Telecommunications": "51",
    "Publishing industries (includes software)": "51",
    "Information and data processing services": "51",
    # --- finance ---
    "Credit Intermediation": "52",
    "Insurance carriers and related activities": "52",
    "Securities, commodity contracts, other financial investments, and related activities": "52",
    # --- real estate ---
    "Rental and leasing services and lessors of intangible assets": "53",
    # --- professional ---
    "Computer systems design and related services": "54",
    "Legal Services": "54",
    "Miscellaneous professional, scientific, and technical services": "54",
    # --- management / admin ---
    "Management of companies (holding companies)": "55",
    "Administrative and support services": "56",
    "Waste Management and remediation services": "56",
    # --- education / health ---
    "Educational Services": "61",
    "Hospitals, nursing, and residential care facilities": "62",
    "Offices of health practioners and outpatient care centers": "62",
    "Miscellaneous health care and social assistance": "62",
    # --- arts / accommodation / other ---
    "Amusement, gambling, and recreation industries": "71",
    "Performing arts, spectator sports, museums and entertainment": "71",
    "Accommodation": "72",
    "Food services and drinking places": "72",
    "Other services": "81",
}

# postal -> FIPS, for the 33 PDIT states. BDS st is numeric FIPS.
FIPS = {"AL": 1, "AZ": 4, "CA": 6, "CO": 8, "CT": 9, "DC": 11, "FL": 12,
        "GA": 13, "IA": 19, "IL": 17, "IN": 18, "KY": 21, "LA": 22,
        "MA": 25, "MD": 24, "MI": 26, "MN": 27, "MO": 29, "NC": 37,
        "NE": 31, "NJ": 34, "NM": 35, "NV": 32, "NY": 36, "OH": 39,
        "OR": 41, "PA": 42, "SC": 45, "TN": 47, "TX": 48, "VA": 51,
        "WA": 53, "WI": 55}


def load_pdit():
    d = pd.read_csv(PDIT)
    d["Industry"] = d["Industry"].str.strip()
    d = d[~d["Industry"].isin(["All Export", "All Non-Export"])].copy()

    unmapped = sorted(set(d["Industry"]) - set(XWALK))
    if unmapped:
        raise SystemExit("industries not in crosswalk:\n  " + "\n  ".join(unmapped))

    d["sector"] = d["Industry"].map(XWALK)
    d["st"] = d["State"].map(FIPS)
    if d["st"].isna().any():
        raise SystemExit("unmapped states: %s" % sorted(d.loc[d.st.isna(), "State"].unique()))
    d = d.rename(columns={"Base Year": "year", "Industry Value-Added": "va",
                          "Export Industry": "export_ind"})
    return d


def collapse(d, ratio):
    """value-added weighted average of each incentive within state x sector x year"""
    s = d[d["Ratio Type"] == ratio].copy()
    cols = [TREAT] + OTHER_INCENTIVES
    out = []
    for c in cols:
        w = s["va"]
        num = (s[c] * w).groupby([s["st"], s["sector"], s["year"]]).sum()
        den = w.groupby([s["st"], s["sector"], s["year"]]).sum()
        out.append((num / den).rename(c))
    # export share of the sector, to test whether targeting drives results
    ex = ((s["export_ind"] * s["va"]).groupby([s["st"], s["sector"], s["year"]]).sum()
          / s["va"].groupby([s["st"], s["sector"], s["year"]]).sum()).rename("export_share")
    out.append(ex)
    r = pd.concat(out, axis=1).reset_index()
    r.columns = ["st", "sector", "year", "rd", "itc", "jctc", "pta", "cit", "export_share"]
    return r


def load_bds():
    b = pd.read_csv(BDS, dtype={"st": str, "sector": str}, low_memory=False)
    b.columns = [c.strip().lower() for c in b.columns]
    for c in b.columns:
        if c not in ("st", "sector"):
            b[c] = pd.to_numeric(b[c], errors="coerce")
    b["st"] = pd.to_numeric(b["st"], errors="coerce")
    b["sector"] = b["sector"].str.strip()
    b["jrr"] = (b["job_creation"] + b["job_destruction"]) / b["denom"] * 100
    b["entry_rate"] = b["estabs_entry_rate"]
    b["exit_rate"] = b["estabs_exit_rate"]
    return b[["year", "st", "sector", "entry_rate", "jrr", "exit_rate", "denom", "emp"]]


def demean_by(v, codes, ng):
    sums = np.bincount(codes, weights=v, minlength=ng)
    cnts = np.bincount(codes, minlength=ng).astype(float)
    m = np.divide(sums, cnts, out=np.zeros(ng), where=cnts > 0)
    return v - m[codes]


def absorb(v, cl, tol=1e-10, maxiter=500):
    r = v.astype(float).copy()
    if len(cl) == 1:
        return demean_by(r, cl[0][0], cl[0][1])
    for _ in range(maxiter):
        prev = r.copy()
        for codes, ng in cl:
            r = demean_by(r, codes, ng)
        if np.max(np.abs(r - prev)) < tol:
            break
    return r


def fe_ladder(df, var, label):
    sub = df.dropna(subset=[var])
    v = sub[var].to_numpy(dtype=float)
    n = len(v)
    sy = pd.factorize(sub["st"].astype(str) + "_" + sub["year"].astype(str))
    jy = pd.factorize(sub["sector"] + "_" + sub["year"].astype(str))
    sj = pd.factorize(sub["st"].astype(str) + "_" + sub["sector"])
    SY = (sy[0], len(sy[1])); JY = (jy[0], len(jy[1])); SJ = (sj[0], len(sj[1]))

    print("\n--- %s (%s) ---  N=%d  mean=%.6f  sd=%.6f"
          % (var, label, n, v.mean(), v.std(ddof=1)))
    print("%-34s %10s %9s" % ("spec", "sd_res", "var_kept"))
    v0 = None
    res = None
    for name, cl in [("0 none (raw)", []),
                     ("2 state-year", [SY]),
                     ("3 sector-year", [JY]),
                     ("4 state-year + sector-year", [SY, JY]),
                     ("5 + state-sector  <-TARGET", [SY, JY, SJ])]:
        r = (v - v.mean()) if not cl else absorb(v, cl)
        var = float(np.sum(r ** 2)) / n
        if v0 is None:
            v0 = var
        print("%-34s %10.6f %9.3f" % (name, np.sqrt(var), var / v0))
        if name.startswith("5"):
            res = np.sqrt(var)
    return n, res


def main():
    p = load_pdit()
    print("PDIT: %d rows, %d states, %d industries, %d-%d"
          % (len(p), p["st"].nunique(), p["Industry"].nunique(),
             p["year"].min(), p["year"].max()))

    pd12 = collapse(p, RATIO_MAIN)
    print("collapsed to state x sector x year: %d rows, %d sectors"
          % (len(pd12), pd12["sector"].nunique()))

    b = load_bds()
    m = pd12.merge(b, on=["st", "sector", "year"], how="inner")
    print("merged with BDS: %d rows" % len(m))
    print("  states %d | sectors %d | years %d-%d"
          % (m["st"].nunique(), m["sector"].nunique(), m["year"].min(), m["year"].max()))
    print("  expected 33 x 16 x 26 = %d" % (33 * 16 * 26))

    print("\ntreatment summary (rd credit, pct of value added):")
    print("  mean %.5f | sd %.5f | max %.5f | share zero %.3f"
          % (m.rd.mean(), m.rd.std(), m.rd.max(), (m.rd == 0).mean()))

    # ---- the number this whole script exists for ----
    print("\n" + "=" * 62)
    print("HOW MUCH TREATMENT VARIATION SURVIVES THE TARGET FE?")
    print("=" * 62)
    n, sd_d = fe_ladder(m, "rd", RATIO_MAIN)
    for v in ["itc", "cit", "export_share"]:
        fe_ladder(m, v, RATIO_MAIN)

    # ---- power ----
    print("\n" + "=" * 62)
    print("POWER (entry rate as outcome)")
    print("=" * 62)
    _, sd_y = fe_ladder(m, "entry_rate", "BDS")
    if sd_d and sd_d > 0:
        se_iid = sd_y / (sd_d * np.sqrt(n))
        print("\n  sd_resid(D) = %.6f" % sd_d)
        print("  sd_resid(Y) = %.4f pp" % sd_y)
        print("  SE(beta), iid, no cluster        = %.2f" % se_iid)
        for infl, lab in [(3, "moderate"), (6, "conservative")]:
            se = se_iid * infl
            print("  SE(beta), cluster inflation x%d (%s) = %.2f  -> MDE = %.2f"
                  % (infl, lab, se, 2.8 * se))
        print("\n  MDE is in pp of entry rate per 1 unit of rd (i.e. per 100pp")
        print("  of value added). A 1sd move in rd is %.5f, so the MDE for a" % sd_d)
        print("  1sd treatment change = MDE * %.5f." % sd_d)

    os.makedirs(OUTDIR, exist_ok=True)
    m.to_csv(os.path.join(OUTDIR, "31_pdit_bds_panel.csv"), index=False)
    print("\nwrote output\\31_pdit_bds_panel.csv  (%d rows)" % len(m))


if __name__ == "__main__":
    main()