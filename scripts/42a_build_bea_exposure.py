# 42a_build_bea_exposure.py
# Builds a second, independent sector exposure measure from BEA.
#
# WHY THIS EXISTS. The gradient test so far used PDIT exposure, which measures
# how much R&D credit money a sector's firms can claim. The hypothesis is about
# intangible capital, which is a different thing. Checked: the two correlate
# only 0.573 (Spearman 0.506), and the rankings disagree sharply --
# professional services is 1st on BEA intangible intensity but 3rd on PDIT;
# arts and entertainment is 2nd on BEA and 14th on PDIT.
#
# So the gradient test may have been sorting sectors on the wrong variable.
# This script builds the BEA measure so both can enter the regression together.
#
# Measure: IPP / total private fixed investment, summed over 1985-1989.
# Pre-treatment on purpose -- the first adoption in the estimation sample is
# MA 1991, so this window predates every treated state.
#
# KNOWN CONTAMINATION: BEA IPP bundles R&D, software AND entertainment
# originals. Sector 71 (arts, entertainment, recreation) scores 0.515 almost
# entirely on film and music copyrights, which is not the intangible capital
# in the substitution story. Script 42 reports results with and without it.
#
# Run: py scripts\42a_build_bea_exposure.py

import csv
import os
import numpy as np
import pandas as pd

RAW = os.path.join("data", "raw")
OUT = os.path.join("output", "42a_bea_exposure.csv")
PRE_LO, PRE_HI = 1985, 1989

BEA2BDS = {
    "Agriculture, forestry, fishing, and hunting": "11",
    "Mining": "21",
    "Utilities": "22",
    "Construction": "23",
    "Manufacturing": "31-33",
    "Wholesale trade": "42",
    "Retail trade": "44-45",
    "Transportation and warehousing": "48-49",
    "Information": "51",
    "Finance and insurance": "52",
    "Real estate and rental and leasing": "53",
    "Professional, scientific, and technical services": "54",
    "Management of companies and enterprises 5": "55",
    "Administrative and waste management services": "56",
    "Educational services": "61",
    "Health and social assistance": "62",
    "Arts, entertainment, and recreation": "71",
    "Accommodation and food services": "72",
    "Other services, except government": "81",
}


def load_bea(fname):
    """BEA sector rows are the ones with zero indent in the label column"""
    path = os.path.join(RAW, fname)
    if not os.path.exists(path):
        raise SystemExit("missing: " + path)
    lines = open(path, encoding="utf-8", errors="replace").read().split("\n")
    hdr = [i for i, l in enumerate(lines) if l.startswith("Line,")][0]
    yrs = [y.strip() for y in list(csv.reader([lines[hdr]]))[0][2:] if y.strip()]
    out = {}
    for r in csv.reader(lines[hdr + 1:]):
        if not r or not r[0].strip().isdigit():
            continue
        lab = r[1]
        if len(lab) - len(lab.lstrip()) != 0:
            continue
        vals = []
        for v in r[2:2 + len(yrs)]:
            try:
                vals.append(float(v.replace(",", "")))
            except ValueError:
                vals.append(np.nan)
        out[lab.strip()] = pd.Series(vals, index=[int(y) for y in yrs])
    return out


def main():
    ipp = load_bea("bea_ipp_by_industry.csv")
    pfa = load_bea("bea_pfa_by_industry.csv")
    eq = load_bea("bea_equip_by_industry.csv")

    missing = [k for k in BEA2BDS if k not in ipp]
    if missing:
        raise SystemExit("BEA labels not matched:\n  " + "\n  ".join(missing))

    rows = []
    for lab, sec in BEA2BDS.items():
        i = ipp[lab].loc[PRE_LO:PRE_HI].sum()
        p = pfa[lab].loc[PRE_LO:PRE_HI].sum()
        e = eq[lab].loc[PRE_LO:PRE_HI].sum()
        rows.append({"sector": sec, "bea_label": lab,
                     "ipp": i, "pfa": p, "equip": e,
                     "bea_intensity": i / p,
                     "equip_intensity": e / p})
    d = pd.DataFrame(rows).set_index("sector")
    d["bea_norm"] = d["bea_intensity"] / d["bea_intensity"].max()
    d["equip_norm"] = d["equip_intensity"] / d["equip_intensity"].max()

    print("=" * 74)
    print("BEA PRE-TREATMENT INTANGIBLE INTENSITY, %d-%d" % (PRE_LO, PRE_HI))
    print("=" * 74)
    print("IPP / total private fixed investment. equip intensity is kept as a")
    print("placebo: the substitution story is about intangibles, so a gradient")
    print("on tangible capital would point at something else.\n")
    show = d.sort_values("bea_intensity", ascending=False)
    print("%-8s %-46s %10s %10s" % ("sector", "BEA label", "intangible", "equip"))
    for s, r in show.iterrows():
        print("%-8s %-46s %10.3f %10.3f"
              % (s, r["bea_label"][:46], r["bea_intensity"], r["equip_intensity"]))

    os.makedirs("output", exist_ok=True)
    d.to_csv(OUT)
    print("\nwrote %s" % OUT)
    print("\nNOTE sector 71 scores high almost entirely on entertainment")
    print("originals, not software or R&D. Script 42 reports with and without.")


if __name__ == "__main__":
    main()