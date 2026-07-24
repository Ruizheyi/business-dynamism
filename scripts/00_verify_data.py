import pandas as pd

FILES = {
    "pfa":    "data/raw/bea_pfa_by_industry.csv",
    "ipp":    "data/raw/bea_ipp_by_industry.csv",
    "equip":  "data/raw/bea_equip_by_industry.csv",
    "struct": "data/raw/bea_struct_by_industry.csv",
}
IDX = [2, 5, 9, 13, 14, 40, 43, 48, 57, 62, 73, 76, 80, 81, 84, 85, 90, 93, 96]

tables = {}
for k, p in FILES.items():
    d = pd.read_csv(p, skiprows=3)
    tables[k] = d
    yrs = [c for c in d.columns if str(c).strip().isdigit()]
    print(f"{k:7s} shape={d.shape}  years {yrs[0]}-{yrs[-1]} ({len(yrs)})")

print("\nrow alignment check:")
ref = tables["pfa"]["Unnamed: 1"]
ok = True
for k, d in tables.items():
    if len(d) != len(ref):
        print(f"  {k}: ROW COUNT MISMATCH ({len(d)} vs {len(ref)})")
        ok = False
        continue
    bad = (d["Unnamed: 1"].fillna("") != ref.fillna("")).sum()
    print(f"  {k}: {'aligned' if bad == 0 else f'{bad} ROWS DIFFER'}")
    ok &= (bad == 0)

print("\nsector labels at the hardcoded indices:")
for i in IDX:
    print(f"  {i:3d} | {str(ref.iloc[i]).strip()}")

print("\nsanity: does equip + struct + ipp equal total fixed assets?")
yrs = [c for c in tables["pfa"].columns if str(c).strip().isdigit()]
def val(k, i):
    return pd.to_numeric(tables[k].loc[i, yrs].astype(str).str.replace(",", ""), errors="coerce")
for i in [14, 57, 76]:
    tot = val("pfa", i)
    parts = val("equip", i) + val("struct", i) + val("ipp", i)
    gap = ((parts - tot) / tot * 100).abs().max()
    print(f"  row {i} ({str(ref.iloc[i]).strip()[:30]}): max deviation {gap:.2f}%")

print("\nALL CHECKS PASSED" if ok else "\nPROBLEMS FOUND — do not trust the ratios")