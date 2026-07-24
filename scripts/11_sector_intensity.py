import pandas as pd

# BEA top-level sectors (row index in the raw table) -> NAICS 2-digit prefixes
SECTORS = {
    2:  (["11"], "Agriculture, forestry, fishing, hunting"),
    5:  (["21"], "Mining"),
    9:  (["22"], "Utilities"),
    13: (["23"], "Construction"),
    14: (["31", "32", "33"], "Manufacturing"),
    40: (["42"], "Wholesale trade"),
    43: (["44", "45"], "Retail trade"),
    48: (["48", "49"], "Transportation and warehousing"),
    57: (["51"], "Information"),
    62: (["52"], "Finance and insurance"),
    73: (["53"], "Real estate, rental and leasing"),
    76: (["54"], "Professional, scientific, technical services"),
    80: (["55"], "Management of companies"),
    81: (["56"], "Administrative and waste management"),
    84: (["61"], "Educational services"),
    85: (["62"], "Health care and social assistance"),
    90: (["71"], "Arts, entertainment, recreation"),
    93: (["72"], "Accommodation and food services"),
    96: (["81"], "Other services"),
}

def load(path):
    df = pd.read_csv(path, skiprows=3)
    years = [c for c in df.columns if str(c).strip().isdigit()]
    return df, years

ipp, years = load("data/raw/bea_ipp_by_industry.csv")
pfa, years_pfa = load("data/raw/bea_pfa_by_industry.csv")
assert years == years_pfa, "year columns differ between the two tables"

def row_series(df, idx):
    s = df.loc[idx, years]
    s = pd.to_numeric(s.astype(str).str.replace(",", ""), errors="coerce")
    s.index = [int(str(y).strip()) for y in years]
    return s

rows = []
for idx, (prefixes, label) in SECTORS.items():
    a, b = row_series(ipp, idx), row_series(pfa, idx)
    share = a / b * 100
    for y in share.index:
        rows.append({"year": y, "sector": label,
                     "naics_prefix": ",".join(prefixes),
                     "ipp_share": share.loc[y]})

intensity = pd.DataFrame(rows)
intensity = intensity[intensity["year"].between(1997, 2023)]
intensity.to_csv("data/processed/sector_intangible_intensity.csv", index=False)

print(f"rows: {len(intensity)}  sectors: {intensity['sector'].nunique()}  "
      f"years: {intensity['year'].min()}-{intensity['year'].max()}")
print(f"expected: {19 * 27} rows\n")

pre = (intensity[intensity["year"].between(1997, 2007)]
       .groupby("sector")["ipp_share"].mean().sort_values(ascending=False))
print("Average IPP share of fixed investment, 1997-2007 (%):\n")
print(pre.round(1).to_string())