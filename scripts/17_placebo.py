import pandas as pd
import numpy as np
import statsmodels.formula.api as smf

SECTORS = {
    2: (["11"], "Agriculture"), 5: (["21"], "Mining"), 9: (["22"], "Utilities"),
    13: (["23"], "Construction"), 14: (["31", "32", "33"], "Manufacturing"),
    40: (["42"], "Wholesale trade"), 43: (["44", "45"], "Retail trade"),
    48: (["48", "49"], "Transportation"), 57: (["51"], "Information"),
    62: (["52"], "Finance and insurance"), 73: (["53"], "Real estate"),
    76: (["54"], "Professional services"), 80: (["55"], "Management of companies"),
    81: (["56"], "Administrative services"), 84: (["61"], "Educational services"),
    85: (["62"], "Health care"), 90: (["71"], "Arts and recreation"),
    93: (["72"], "Accommodation and food"), 96: (["81"], "Other services"),
}

def load_bea(path):
    d = pd.read_csv(path, skiprows=3)
    years = [c for c in d.columns if str(c).strip().isdigit()]
    return d, years

def get_row(df, idx, yr_cols):
    s = pd.to_numeric(df.loc[idx, yr_cols].astype(str).str.replace(",", ""), errors="coerce")
    s.index = [int(str(y).strip()) for y in yr_cols]
    return s

pfa, yrs = load_bea("data/raw/bea_pfa_by_industry.csv")
ipp, _ = load_bea("data/raw/bea_ipp_by_industry.csv")
equip, _ = load_bea("data/raw/bea_equip_by_industry.csv")
struct, _ = load_bea("data/raw/bea_struct_by_industry.csv")

rows = []
for idx, (prefixes, name) in SECTORS.items():
    denom = get_row(pfa, idx, yrs)
    for asset_type, df in [("ipp", ipp), ("equip", equip), ("struct", struct)]:
        share = (get_row(df, idx, yrs) / denom) * 100
        for y in share.index:
            for p in prefixes:
                rows.append({"year": y, "naics2": p, "sector": name,
                             "asset_type": asset_type, "share": share.loc[y]})

shares = pd.DataFrame(rows)

bds = pd.read_csv("data/raw/bds2023_vcn3_fa.csv")
for c in ["denom", "job_creation", "job_destruction"]:
    bds[c] = pd.to_numeric(bds[c], errors="coerce")

mature = bds[~bds["fage"].str.startswith("a) 0", na=False)]
ind = (mature.groupby(["year", "vcnaics3"])
       .agg(jc=("job_creation", "sum"), jd=("job_destruction", "sum"),
            denom=("denom", "sum")).reset_index())
ind["naics2"] = ind["vcnaics3"].astype(str).str[:2]

m = ind.merge(shares, on=["year", "naics2"], how="inner")
m = m[m["year"].between(1997, 2023)]

panel = (m.groupby(["sector", "year"])
         .agg(jc=("jc", "sum"), jd=("jd", "sum"), denom=("denom", "sum"))
         .reset_index())
panel["jrr"] = (panel["jc"] + panel["jd"]) / panel["denom"] * 100
panel["yr"] = panel["year"].astype(str)

for asset_type in ["ipp", "equip", "struct"]:
    asset_shares = (m.groupby(["sector", "year"])
                    .apply(lambda x: x[x["asset_type"] == asset_type]["share"].iloc[0]
                           if len(x[x["asset_type"] == asset_type]) > 0 else np.nan))
    panel = panel.set_index(["sector", "year"]).join(
        asset_shares.rename(f"{asset_type}_share"), how="left").reset_index()

panel = panel.sort_values(["sector", "year"])

for col in ["ipp_share", "equip_share", "struct_share"]:
    panel[f"{col}_lag3"] = panel.groupby("sector")[col].shift(3)

print(f"panel: {len(panel)} sector-years\n")
print("mean asset shares, 1997-2023 (%):")
print(panel[["ipp_share", "equip_share", "struct_share"]].mean().round(1).to_string(), "\n")

def fit_reg(y, x, extra=""):
    d = panel.dropna(subset=[y, x, "denom"])
    f = f"{y} ~ {x} + C(sector) + C(yr)" + extra
    r = smf.wls(f, data=d, weights=d["denom"]).fit(
        cov_type="cluster", cov_kwds={"groups": d["sector"]})
    return r.params[x], r.bse[x], r.pvalues[x], int(r.nobs)

res = []
for asset_base in ["ipp_share", "equip_share", "struct_share"]:
    b, s, p, n = fit_reg("jrr", asset_base)
    res.append({"regressor": asset_base, "N": n,
                "beta": round(b, 3), "se": round(s, 3), "p": round(p, 3)})
    
    b, s, p, n = fit_reg("jrr", f"{asset_base}_lag3")
    res.append({"regressor": f"{asset_base}_lag3", "N": n,
                "beta": round(b, 3), "se": round(s, 3), "p": round(p, 3)})
    
    b, s, p, n = fit_reg("jrr", asset_base, " + C(sector):year")
    res.append({"regressor": f"{asset_base}_trend", "N": n,
                "beta": round(b, 3), "se": round(s, 3), "p": round(p, 3)})

out = pd.DataFrame(res)
out.to_csv("data/processed/placebo_results.csv", index=False)
print(out.to_string(index=False))
print("\nSector & year FE; clustered by sector.")
print("IPP should differ in sign from equipment & structures if specific to intangibles.")