import pandas as pd

#  BDS data loading...
bds = pd.read_csv("data/raw/bds2023_vcn3_fa.csv")

# Convert to numeric, coerce errors to NaN
for col in ["emp", "denom", "job_creation", "job_destruction"]:
    bds[col] = pd.to_numeric(bds[col], errors="coerce")

# jrr excludes age-0 firms (those created this year)
young_ages = ["a) 0", "b) 1", "c) 2", "d) 3", "e) 4", "f) 5"]
mature_bds = bds[~bds["fage"].str.startswith("a) 0", na=False)]

total_emp = bds.groupby("year")["emp"].sum().rename("total_emp")
young_emp = bds[bds["fage"].isin(young_ages)].groupby("year")["emp"].sum().rename("young_emp")

# JRR = (job creation + destruction) / denominator
# denom is DHS average (current + lagged employment)
agg = mature_bds.groupby("year").agg(
    jc=("job_creation", "sum"),
    jd=("job_destruction", "sum"),
    emp=("denom", "sum")
)

# together
national = pd.concat([total_emp, young_emp, agg], axis=1).reset_index()
national["jrr"] = (national["jc"] + national["jd"]) / national["emp"] * 100
national["young_share"] = national["young_emp"] / national["total_emp"] * 100

nipa = pd.read_csv("data/raw/bea_nipa_535.csv", skiprows=3)
nipa = nipa.rename(columns={"Unnamed: 1": "item"})
nipa["item"] = nipa["item"].astype(str).str.strip()

year_cols = [c for c in nipa.columns if str(c).strip().isdigit()]

def get_series(label):
    # Find the row starts with label
    row = nipa[nipa["item"].str.startswith(label, na=False)]
    print(f"matched '{label}': {row['item'].tolist()}")
    
    # value and convert to numeric
    s = row.iloc[0][year_cols]
    s.index = s.index.astype(int)
    s = pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
    return s

# software, IPP, PFI shares
pfi = get_series("Private fixed investment")
software = get_series("Software")
ipp = get_series("Intellectual property products")

# investment share df
invest = pd.DataFrame({
    "pfi": pfi,
    "software": software,
    "ipp": ipp
})
invest.index.name = "year"
invest = invest.reset_index()

# shares as pct of PFI
invest["software_share"] = invest["software"] / invest["pfi"] * 100
invest["ipp_share"] = invest["ipp"] / invest["pfi"] * 100

# Merge with national data
panel = national.merge(invest[["year", "software_share", "ipp_share"]], on="year")
panel = panel[panel["year"].between(1997, 2023)].copy()

# rebase all to 1997 = 100
for col in ["jrr", "young_share", "software_share", "ipp_share"]:
    base_val = panel.loc[panel["year"] == 1997, col].values[0]
    panel[col + "_index"] = (panel[col] / base_val) * 100

# Scissor gap: software index minus jrr index
panel["scissor_gap"] = panel["software_share_index"] - panel["jrr_index"]

# the key columns
keep_cols = ["year", "jrr", "young_share", "software_share", "ipp_share",
             "jrr_index", "young_share_index", "software_share_index", "ipp_share_index",
             "scissor_gap"]
panel[keep_cols].to_csv("data/processed/national_panel.csv", index=False)

# Print to check
print("\n" + panel[keep_cols].round(1).to_string(index=False))
print(f"\n2023 scissor gap: {panel.loc[panel['year'] == 2023, 'scissor_gap'].values[0]:.1f}")