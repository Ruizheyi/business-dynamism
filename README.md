\# Declining Business Dynamism and Intangible Capital



Replication and extension of Decker et al. (2016) using BDS and BEA data. Tests whether intangible capital substitution explains why U.S. business dynamism has been declining.



Short version: intangible investment and business dynamism diverge sharply and consistently. Whether they're causally linked can't be determined from public sector-level data—documenting why is the point.



\---



\## What this is about



U.S. business dynamism has been falling since the 1980s. Decker et al. showed it's happening across multiple dimensions: job reallocation rates down, establishment entry rates down, young-firm employment shares down. They answered "what" but not "why."



This project tests one candidate mechanism: firms increasingly substitute software and other intangible capital for labor churn. Do the sectoral patterns support it?



\---



\## Data



| What | Source | Coverage |

|---|---|---|

| Job flows \& employment | Census BDS | 3-digit NAICS × firm age, 1978–2023 |

| National investment | BEA NIPA Table 5.3.5 | Software \& IPP spending, 1947–2023 |

| By-sector investment | BEA Fixed Assets Tables 3.7\* | IPP, equipment, structures by industry, 1947–2023 |



\*\*Job reallocation rate:\*\* (job creation + destruction) / BDS denominator × 100, excluding age-0 entrants.



\*\*Young-firm share:\*\* employment in firms age ≤5 / total employment.



\*\*Intangible intensity:\*\* IPP (intellectual property products—R\&D, software, entertainment originals) as % of total fixed investment per sector. Software-only numbers aren't published by industry, so IPP is the finest detail available.



\---



\## What I found 



\### The aggregate picture



JRR fell from 30.7% (1997–2007 average) to 23.1% (2013–2023 average). Meanwhile software's share of fixed investment rose from 6.2% to 13.0%. The scissor gap hits 136 index points by 2023.



\### The structural fact



Shift-share decomposition: 104.9% of the total JRR decline is within-sector. Between-sector reallocation contributes nothing (slightly offsets it). This result is rock-solid—doesn't change much if you try different period cuts (within-share stays 102–109%).



\*\*What this means:\*\* the story isn't "the economy shifted toward low-dynamism industries." It's "every industry got quieter."



\### The sectoral pattern



Sectors that intangified more saw their JRR fall more. 19-sector scatter: slope −0.23 (p=0.12, R²=0.18). Not statistically tight, but directional. Some sectors (educational services, other services) clearly break the pattern.



\### Panel results: which outcome matters?



Ran seven dynamism measures (JRR, entry rate, exit rate, firm death, young-firm share, job creation from births, job destruction from deaths) against IPP intensity at sector-year level:



Entry rate hits p=0.035 (contemporaneous) and p=0.015 (3-year lag). That's the clearest signal. All coefficients point the predicted direction, and effects grow over longer horizons. Looks like a slow mechanism.



But add sector-specific linear trends to the model: everything disappears. IPP flips to +0.14 (p=0.03). This kills the identification.



\*\*Why it matters:\*\* can't separate "intangible capital replacing labor" from "whatever else makes some sectors stagnate over time." Treatment varies only at 19-sector level. Need establishment-level data (LBD) to go further.



\### Asset-type check



Equipment and structures correlate differently than IPP. Equipment actually has opposite sign (+0.09 contemp., +0.16 lag). Supports specificity of the IPP association. But this also disappears with sector trends.



\---



\## What didn't work



\*\*Recession shocks:\*\* Looked at 2001, 2008, 2020 as events. High-intangible sectors didn't recover differently post-shock. So it's not a cyclical mechanism.



\*\*Causal identification:\*\* Can't separate causation from confounding with public data at this level. Reverse causality plausible too (stable industries invest more in long-run intangibles because they can).



\---



\## The code

