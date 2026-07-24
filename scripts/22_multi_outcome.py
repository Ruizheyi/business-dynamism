import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

# --- build sector-year outcomes from BDS ---
bds = pd.read_csv("data/raw/bds2023_vcn3_fa.csv")
cols = ["emp", "denom", "firms", "estabs", "estabs_entry", "estabs_exit",
        "job_creation", "job_creation_births", "job_destruction",
        "job_destruction_deaths", "firmdeath_firms", "firmdeath_emp"]
for c in cols:
    bds[c] = pd.to_numeric(bds[c], errors="coerce")

young = ["a) 0", "b) 1", "c) 2", "d) 3", "e) 4", "f) 5"]
bds["naics2"] = bds["vcnaics3"].astype(str).str[:2]

inten = pd.read_csv("data/processed/sector_intangible_intensity.csv")
rows = []
for _, r in inten.iterrows():
    for p in str(r["naics_prefix"]).split(","):
        rows.append({"year": int(r["year"]), "naics2": p.strip(),
                     "sector": r["sector"], "ipp": r["ipp_share"]})
sec_year = pd.DataFrame(rows)

d = bds.merge(sec_year, on=["year", "naics2"], how="inner")
d = d[d["year"].between(1997, 2023)]

# all-age aggregates
allage = d.groupby(["sector", "year"]).agg(
    emp=("emp", "sum"), estabs=("estabs", "sum"), firms=("firms", "sum"),
    entry=("estabs_entry", "sum"), exit=("estabs_exit", "sum"),
    fdeath=("firmdeath_firms", "sum"), ipp=("ipp", "first")).reset_index()
young_emp = (d[d["fage"].isin(young)].groupby(["sector", "year"])["emp"]
             .sum().rename("young_emp").reset_index())

# age>0 flows, matching the JRR convention used elsewhere
mature = d[~d["fage"].str.startswith("a) 0", na=False)]
flows = mature.groupby(["sector", "year"]).agg(
    jc=("job_creation", "sum"), jd=("job_destruction", "sum"),
    jcb=("job_creation_births", "sum"), jdd=("job_destruction_deaths", "sum"),
    denom=("denom", "sum")).reset_index()

p = allage.merge(young_emp, on=["sector", "year"]).merge(flows, on=["sector", "year"])
p["jrr"]         = (p["jc"] + p["jd"]) / p["denom"] * 100
p["entry_rate"]  = p["entry"] / p["estabs"] * 100
p["exit_rate"]   = p["exit"] / p["estabs"] * 100
p["firm_death"]  = p["fdeath"] / p["firms"] * 100
p["young_share"] = p["young_emp"] / p["emp"] * 100
p["jc_births"]   = p["jcb"] / p["denom"] * 100
p["jd_deaths"]   = p["jdd"] / p["denom"] * 100
p["yr"] = p["year"].astype(str)
p = p.sort_values(["sector", "year"])
p["ipp_lag3"] = p.groupby("sector")["ipp"].shift(3)

OUTCOMES = {
    "jrr":         "Job reallocation rate",
    "entry_rate":  "Establishment entry rate",
    "exit_rate":   "Establishment exit rate",
    "firm_death":  "Firm death rate",
    "young_share": "Young-firm employment share",
    "jc_births":   "Job creation from births",
    "jd_deaths":   "Job destruction from deaths",
}

print(f"panel: {len(p)} sector-years, {p['sector'].nunique()} sectors\n")
print("outcome means, 1997-2023 (%):")
print(p[list(OUTCOMES)].mean().round(2).to_string(), "\n")


def fit(y, x, extra=""):
    dd = p.dropna(subset=[y, x, "denom"])
    f = f"{y} ~ {x} + C(sector) + C(yr)" + extra
    r = smf.wls(f, data=dd, weights=dd["denom"]).fit(
        cov_type="cluster", cov_kwds={"groups": dd["sector"]})
    return r.params[x], r.bse[x], r.pvalues[x], int(r.nobs)


res = []
for y, label in OUTCOMES.items():
    b0, s0, p0, n0 = fit(y, "ipp")
    b3, s3, p3, n3 = fit(y, "ipp_lag3")
    bt, st, pt, _  = fit(y, "ipp", " + C(sector):year")
    res.append({"outcome": label, "N": n0,
                "beta": round(b0, 3), "p": round(p0, 3),
                "beta_lag3": round(b3, 3), "p_lag3": round(p3, 3),
                "beta_trend": round(bt, 3), "p_trend": round(pt, 3)})

out = pd.DataFrame(res)
out.to_csv("data/processed/multi_outcome_results.csv", index=False)
print(out.to_string(index=False))
print("\nSector and year fixed effects; SEs clustered by sector (19 clusters).")
print("beta = contemporaneous, beta_lag3 = three-year lag,")
print("beta_trend = contemporaneous with sector-specific linear trends.")
print("Seven outcomes x three specifications = 21 tests; some will cross p<0.05")
print("by chance. All results are reported.")