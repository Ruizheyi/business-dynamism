import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

P0, P1 = (1997, 2007), (2013, 2023)

SECTORS = {
    2:  (["11"], "Agriculture"), 5: (["21"], "Mining"), 9: (["22"], "Utilities"),
    13: (["23"], "Construction"), 14: (["31", "32", "33"], "Manufacturing"),
    40: (["42"], "Wholesale trade"), 43: (["44", "45"], "Retail trade"),
    48: (["48", "49"], "Transportation"), 57: (["51"], "Information"),
    62: (["52"], "Finance and insurance"), 73: (["53"], "Real estate"),
    76: (["54"], "Professional services"), 80: (["55"], "Management of companies"),
    81: (["56"], "Administrative services"), 84: (["61"], "Educational services"),
    85: (["62"], "Health care"), 90: (["71"], "Arts and recreation"),
    93: (["72"], "Accommodation and food"), 96: (["81"], "Other services"),
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
    share = row(ipp, idx, yrs) / row(pfa, idx, yrs) * 100
    for y in share.index:
        for p in pref:
            rows.append({"year": y, "naics2": p, "sector": label, "ipp": share.loc[y]})
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
panel = (m.groupby(["sector", "year"])
         .agg(jc=("jc", "sum"), jd=("jd", "sum"), denom=("denom", "sum"),
              ipp=("ipp", "first")).reset_index())
panel["jrr"] = (panel["jc"] + panel["jd"]) / panel["denom"] * 100


def period(a, b):
    d = panel[panel["year"].between(a, b)]
    g = d.groupby("sector").agg(jc=("jc", "sum"), jd=("jd", "sum"),
                                denom=("denom", "sum"), ipp=("ipp", "mean"))
    g["jrr"] = (g["jc"] + g["jd"]) / g["denom"] * 100
    g["w"] = g["denom"] / g["denom"].sum()
    return g[["jrr", "w", "ipp"]]

p0, p1 = period(*P0), period(*P1)
d = p0.join(p1, lsuffix="_0", rsuffix="_1").dropna()

agg0 = (d["w_0"] * d["jrr_0"]).sum()
agg1 = (d["w_1"] * d["jrr_1"]).sum()
d["w_bar"], d["jrr_bar"] = (d["w_0"] + d["w_1"]) / 2, (d["jrr_0"] + d["jrr_1"]) / 2
d["d_jrr"], d["d_w"] = d["jrr_1"] - d["jrr_0"], d["w_1"] - d["w_0"]
d["within"] = d["w_bar"] * d["d_jrr"]
d["between"] = d["jrr_bar"] * d["d_w"]
d["d_ipp"] = d["ipp_1"] - d["ipp_0"]

total = agg1 - agg0
W, B = d["within"].sum(), d["between"].sum()

print(f"aggregate JRR: {agg0:.2f}% ({P0[0]}-{P0[1]})  ->  {agg1:.2f}% ({P1[0]}-{P1[1]})")
print(f"total change: {total:+.2f} pp\n")
print(f"  within-sector  : {W:+.2f} pp  ({W/total*100:5.1f}% of the change)")
print(f"  between-sector : {B:+.2f} pp  ({B/total*100:5.1f}% of the change)")
print(f"  check (W+B)    : {W+B:+.2f} pp\n")

print(d[["jrr_0", "jrr_1", "d_jrr", "d_ipp", "w_0", "w_1", "within", "between"]]
      .sort_values("within").round(2).to_string())

# does the within-sector decline line up with rising intangible intensity?
r = smf.wls("d_jrr ~ d_ipp", data=d.reset_index(), weights=d["w_bar"]).fit(cov_type="HC1")
print(f"\ncross-section: d_jrr on d_ipp (weighted, {int(r.nobs)} sectors)")
print(f"  slope {r.params['d_ipp']:+.3f}  se {r.bse['d_ipp']:.3f}  "
      f"p {r.pvalues['d_ipp']:.3f}  R2 {r.rsquared:.3f}")
print("  descriptive association, not a causal estimate")

d.to_csv("data/processed/decomposition.csv")

fig, ax = plt.subplots(figsize=(8, 5.5))
ax.scatter(d["d_ipp"], d["d_jrr"], s=d["w_bar"] * 3000, alpha=0.45,
           color="#2c3e50", edgecolor="white", linewidth=1.2)
xs = np.linspace(d["d_ipp"].min() - 1, d["d_ipp"].max() + 1, 50)
ax.plot(xs, r.params["Intercept"] + r.params["d_ipp"] * xs,
        color="#c0392b", lw=1.8, label=f"slope = {r.params['d_ipp']:.2f}")
for s, rw in d.iterrows():
    ax.annotate(s, (rw["d_ipp"], rw["d_jrr"]), fontsize=7.5,
                color="#5d6d7e", xytext=(4, 4), textcoords="offset points")
ax.axhline(0, color="black", lw=0.7, alpha=0.4)
ax.axvline(0, color="black", lw=0.7, alpha=0.4)
ax.set_xlabel("Change in IPP share of fixed investment (pp)")
ax.set_ylabel("Change in job reallocation rate (pp)")
ax.set_title(f"Within-sector change, {P0[0]}-{P0[1]} vs {P1[0]}-{P1[1]}\n"
             "marker size = employment weight", fontsize=11.5)
ax.legend(frameon=False, fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig("output/decomposition_scatter.png", dpi=200)
print("saved: output/decomposition_scatter.png")