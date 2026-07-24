import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

WINDOW = 5
BASE_K = -1

# prep sector-year panel loading
bds = pd.read_csv("data/raw/bds2023_vcn3_fa.csv")
for c in ["denom", "job_creation", "job_destruction"]:
    bds[c] = pd.to_numeric(bds[c], errors="coerce")

mature = bds[~bds["fage"].str.startswith("a) 0", na=False)]
ind = (mature.groupby(["year", "vcnaics3"])
       .agg(jc=("job_creation", "sum"), jd=("job_destruction", "sum"),
            denom=("denom", "sum")).reset_index())
ind = ind[ind["denom"] > 0]
ind["naics2"] = ind["vcnaics3"].astype(str).str[:2]

# pre-crisis sector intensity loaddddddding
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

# pre-crisis intensity fixed          treatment variable
pre = (panel[panel["year"].between(1997, 2007)]
       .groupby("sector")["ipp"].mean().rename("ipp_pre"))
panel = panel.merge(pre, on="sector")

med = pre.median()
mu = pre.mean()
sd = pre.std()

print(f"sector-year panel: {len(panel)} rows, {panel['sector'].nunique()} sectors\n")

def run_event_study(event_year, treat_type="continuous", drop_sects=None,
                    base_k=BASE_K, weighted=True, label=""):
    """Run event study regression"""
    d = panel.copy()
    if drop_sects:
        d = d[~d["sector"].isin(drop_sects)]
    
    # for treatment: binary or continuous
    if treat_type == "binary":
        d["T"] = (d["ipp_pre"] > med).astype(float)
    else:
        d["T"] = (d["ipp_pre"] - mu) / sd
    
    d["k"] = d["year"] - event_year
    d = d[d["k"].between(-WINDOW, WINDOW)].copy()
    d["yr"] = d["year"].astype(str)
    
    # interaction dummies for each event time k
    ks = [k for k in range(-WINDOW, WINDOW + 1) if k != base_k]
    names = []
    for k in ks:
        name = f"tk_m{abs(k)}" if k < 0 else f"tk_p{k}"
        d[name] = d["T"] * (d["k"] == k)
        names.append(name)
    
    # Regression: JRR ~ interactions + sector FE + year FE
    formula = f"jrr ~ {' + '.join(names)} + C(sector) + C(yr)"
    mod = smf.wls(formula, data=d, weights=d["denom"]) if weighted else smf.ols(formula, data=d)
    res = mod.fit(cov_type="cluster", cov_kwds={"groups": d["sector"]})
    
    # avg pre and post coefficients
    pre_ks = [k for k in ks if k < base_k]
    post_ks = [k for k in ks if k >= 0]
    
    g = lambda k: (f"tk_m{abs(k)}" if k < 0 else f"tk_p{k}")
    pre_avg = np.mean([res.params[g(k)] for k in pre_ks]) if pre_ks else np.nan
    post_avg = np.mean([res.params[g(k)] for k in post_ks])
    n_sig = sum(res.pvalues[g(k)] < 0.05 for k in post_ks)
    
    return {"spec": label, "event": event_year, "N": int(res.nobs),
            "pre_avg": round(pre_avg, 2), "post_avg": round(post_avg, 2),
            "sig_5pct": f"{n_sig}/{len(post_ks)}"}

DROP = ["Manufacturing", "Arts, entertainment, recreation"]
specs = [
    run_event_study(2008, "binary", label="1. binary treatment"),
    run_event_study(2008, "continuous", label="2. continuous treatment"),
    run_event_study(2008, "continuous", DROP, label="3. drop mfg + arts"),
    run_event_study(2008, "continuous", base_k=-2, label="4. base year t-2"),
    run_event_study(2008, "continuous", weighted=False, label="5. unweighted"),
    run_event_study(2001, "continuous", label="6. dot-com 2001"),
    run_event_study(2020, "continuous", label="7. COVID 2020"),
]

out = pd.DataFrame(specs)
out.to_csv("data/processed/event_study_specs.csv", index=False)
print(out.to_string(index=False))
print("\nSector & year FE; SEs clustered by sector (19 clusters).")
print("Hypothesis: post_avg < 0")