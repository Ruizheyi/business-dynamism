import pandas as pd

SECTORS = {
    2: (["11"],"Agriculture"), 5: (["21"],"Mining"), 9: (["22"],"Utilities"),
    13: (["23"],"Construction"), 14: (["31","32","33"],"Manufacturing"),
    40: (["42"],"Wholesale trade"), 43: (["44","45"],"Retail trade"),
    48: (["48","49"],"Transportation"), 57: (["51"],"Information"),
    62: (["52"],"Finance and insurance"), 73: (["53"],"Real estate"),
    76: (["54"],"Professional services"), 80: (["55"],"Management of companies"),
    81: (["56"],"Administrative services"), 84: (["61"],"Educational services"),
    85: (["62"],"Health care"), 90: (["71"],"Arts and recreation"),
    93: (["72"],"Accommodation and food"), 96: (["81"],"Other services"),
}

def load(p):
    d = pd.read_csv(p, skiprows=3)
    return d, [c for c in d.columns if str(c).strip().isdigit()]

def row(df, idx, yrs):
    s = pd.to_numeric(df.loc[idx, yrs].astype(str).str.replace(",", ""), errors="coerce")
    s.index = [int(str(y).strip()) for y in yrs]
    return s

pfa, yrs = load("data/raw/bea_pfa_by_industry.csv")
ipp, _ = load("data/raw/bea_ipp_by_industry.csv")
rows = []
for idx, (pref, label) in SECTORS.items():
    sh = row(ipp, idx, yrs) / row(pfa, idx, yrs) * 100
    for y in sh.index:
        for p in pref:
            rows.append({"year": y, "naics2": p, "sector": label, "ipp": sh.loc[y]})
shares = pd.DataFrame(rows)

bds = pd.read_csv("data/raw/bds2023_vcn3_fa.csv")
for c in ["denom", "job_creation", "job_destruction"]:
    bds[c] = pd.to_numeric(bds[c], errors="coerce")
mature = bds[~bds["fage"].str.startswith("a) 0", na=False)]
ind = (mature.groupby(["year", "vcnaics3"])
       .agg(jc=("job_creation","sum"), jd=("job_destruction","sum"),
            denom=("denom","sum")).reset_index())
ind["naics2"] = ind["vcnaics3"].astype(str).str[:2]
m = ind.merge(shares, on=["year","naics2"], how="inner")
panel = (m.groupby(["sector","year"])
         .agg(jc=("jc","sum"), jd=("jd","sum"), denom=("denom","sum")).reset_index())


def decompose(p0, p1):
    def per(a, b):
        g = panel[panel["year"].between(a, b)].groupby("sector").agg(
            jc=("jc","sum"), jd=("jd","sum"), denom=("denom","sum"))
        g["jrr"] = (g["jc"] + g["jd"]) / g["denom"] * 100
        g["w"] = g["denom"] / g["denom"].sum()
        return g[["jrr","w"]]
    a, b = per(*p0), per(*p1)
    d = a.join(b, lsuffix="_0", rsuffix="_1").dropna()
    total = (d["w_1"]*d["jrr_1"]).sum() - (d["w_0"]*d["jrr_0"]).sum()
    wbar, jbar = (d["w_0"]+d["w_1"])/2, (d["jrr_0"]+d["jrr_1"])/2
    W = (wbar * (d["jrr_1"]-d["jrr_0"])).sum()
    B = (jbar * (d["w_1"]-d["w_0"])).sum()
    return {"periods": f"{p0[0]}-{p0[1]} vs {p1[0]}-{p1[1]}",
            "total": round(total, 2), "within": round(W, 2), "between": round(B, 2),
            "within_%": round(W/total*100, 1)}


splits = [((1997,2007),(2013,2023)), ((1997,2006),(2014,2023)),
          ((1997,2005),(2015,2023)), ((1997,2007),(2010,2019)),
          ((1997,2001),(2019,2023)), ((1997,2019),(1997,2019))]
splits[-1] = ((1997,2004),(2016,2023))

res = pd.DataFrame([decompose(*s) for s in splits])
res.to_csv("data/processed/decomp_sensitivity.csv", index=False)
print(res.to_string(index=False))
print("\nAll figures in percentage points. Within-share close to 100% across")
print("splits would indicate the decomposition is not driven by the period cut.")