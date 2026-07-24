import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# Sector-year panel
bds = pd.read_csv("data/raw/bds2023_vcn3_fa.csv")
for c in ["denom", "job_creation", "job_destruction"]:
    bds[c] = pd.to_numeric(bds[c], errors="coerce")

mature = bds[~bds["fage"].str.startswith("a) 0", na=False)]
ind = (mature.groupby(["year", "vcnaics3"])
       .agg(jc=("job_creation", "sum"), jd=("job_destruction", "sum"),
            denom=("denom", "sum")).reset_index())
ind = ind[ind["denom"] > 0]
ind["naics2"] = ind["vcnaics3"].astype(str).str[:2]

inten = pd.read_csv("data/processed/sector_intangible_intensity.csv")
rows = []
for _, r in inten.iterrows():
    for p in str(r["naics_prefix"]).split(","):
        rows.append({"year": int(r["year"]), "naics2": p.strip(),
                     "sector": r["sector"], "ipp": r["ipp_share"]})
sec_year = pd.DataFrame(rows)

m = ind.merge(sec_year, on=["year", "naics2"], how="inner")
panel = (m.groupby(["sector", "year"])
         .agg(jc=("jc", "sum"), jd=("jd", "sum"), denom=("denom", "sum"),
              ipp=("ipp", "first")).reset_index())
panel["jrr"] = (panel["jc"] + panel["jd"]) / panel["denom"] * 100
panel = panel.sort_values(["sector", "year"])
panel["ipp_lag3"] = panel.groupby("sector")["ipp"].shift(3)
panel["yr"] = panel["year"].astype(str)

print(f"panel: {len(panel)} sector-years, {panel['sector'].nunique()} sectors\n")

def fit_panel(x, data=None, extra="", label=""):
    """Fit a sector-year panel regression"""
    d = (panel if data is None else data).dropna(subset=[x, "jrr", "denom"])
    f = f"jrr ~ {x} + C(sector) + C(yr)" + extra
    r = smf.wls(f, data=d, weights=d["denom"]).fit(
        cov_type="cluster", cov_kwds={"groups": d["sector"]})
    return {"var": label or x, "N": int(r.nobs),
            "beta": round(r.params[x], 3), "se": round(r.bse[x], 3),
            "p": round(r.pvalues[x], 3)}

# main panel regressions
res = [
    fit_panel("ipp", label="IPP share (main)"),
    fit_panel("ipp_lag3", label="IPP, 3-yr lag"),
    fit_panel("ipp", extra=" + C(sector):year", label="IPP + sector trends"),
]

# longtime difference:from early to late period
early = panel[panel["year"].between(1997, 2007)].groupby("sector").agg(
    jrr0=("jrr", "mean"), ipp0=("ipp", "mean"), w=("denom", "mean"))
late = panel[panel["year"].between(2013, 2023)].groupby("sector").agg(
    jrr1=("jrr", "mean"), ipp1=("ipp", "mean"))
ld = early.join(late).reset_index()
ld["d_jrr"] = ld["jrr1"] - ld["jrr0"]
ld["d_ipp"] = ld["ipp1"] - ld["ipp0"]

rl = smf.wls("d_jrr ~ d_ipp", data=ld, weights=ld["w"]).fit(cov_type="HC1")
res.append({"var": "long difference", "N": int(rl.nobs),
            "beta": round(rl.params["d_ipp"], 3), "se": round(rl.bse["d_ipp"], 3),
            "p": round(rl.pvalues["d_ipp"], 3)})

out = pd.DataFrame(res)
out.to_csv("data/processed/panel_design_results.csv", index=False)
print(out.to_string(index=False))
print("\nSector & year FE; SEs clustered by sector.")
print("Hypothesis: beta < 0")