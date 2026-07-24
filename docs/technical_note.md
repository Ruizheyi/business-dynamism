\# Technical Note: Identifying the Intangible Capital Channel in Declining Business Dynamism



\## 1. Data and Denominators



The job reallocation rate (JRR) is constructed as:



$$\\text{JRR}\_{t,i} = \\frac{\\text{JC}\_{t,i} + \\text{JD}\_{t,i}}{\\text{Denom}\_{t,i}} \\times 100$$



where JC and JD are job creation and destruction, and Denom is the BDS employment denominator.



\*\*Critical choice: which denominator?\*\* BDS publishes two: end-of-period employment (`emp`) and the DHS average (average of current and prior-year employment, `denom`). The literature standard is `denom`. 



Early versions of this project used `emp`, which produced different sectoral rankings and inflated rates in sectors with high employment growth. The error was caught via audit (script 14) comparing the two denominators' correlation with industry size. The `denom` version was used throughout the final analysis. This matters because employment suppression in BDS is row-wise (whole establishments), not partial—using the wrong denominator systematically biases cross-industry comparisons.



\*\*Age 0 exclusion:\*\* JRR excludes age-0 firms (created in the current year). This is the original Decker et al. convention. The young-firm employment share, by contrast, includes age-0 entrants in both numerator and denominator, because the denominator (total employment) naturally includes them. The two indicators have different age conventions; results are robust to this.



\## 2. Aggregation Level: Why Sector-Year, Not 3-Digit NAICS?



Treatment (IPP intensity) comes from BEA Fixed Assets, published at the 2-digit BEA industry level (19 sectors). The job data come from BDS at 3-digit NAICS.



Early exploration ran regressions at 3-digit NAICS level, treating IPP intensity as the sectoral average applied to each 3-digit industry. This created two problems:



1\. \*\*No identifying variation:\*\* IPP intensity is constant within sector-year. Adding 3-digit fixed effects alongside sector fixed effects reduced the model to just within-3-digit variation, which does not exist (one sector per 3-digit industry).



2\. \*\*Singular cluster VCV:\*\* 3-digit NAICS gives \~86 industries but only 19 cluster groups (BEA sectors). With \~40 parameters and 19 clusters, the clustered covariance matrix is singular.



\*\*Solution:\*\* Aggregate to sector-year level. This gives 19×27 = 513 observations (sector-years). Treatment varies at the sector-year level, no collinearity, and 19 clusters is still few but mathematically valid.



\*\*Trade-off:\*\* 3-digit granularity is preserved in the data construction (weights in the aggregation), but not in the regression. The 3-digit detail affects which establishments are weighted, not the identifying variation in the regression.



\## 3. Which Outcome Matters? The Role of Decomposition



The job reallocation rate is a composite: JRR = (job creation + job destruction) / denom. If intangible capital primarily deters \*entry\* of new firms (raising barriers to scale software stacks), it might suppress job creation from births but leave job destruction from incumbent firm churn unchanged.



The shift-share decomposition revealed that 105% of the aggregate JRR decline is within-sector. This eliminated sectoral reallocation as an explanation and motivated testing finer-grained outcomes.



Regressing seven separate outcomes (entry rate, exit rate, firm death rate, JRR, young-firm share, job creation from births, job destruction from deaths) showed entry rate to be the strongest signal (p=0.035 contemporaneous, p=0.015 lag-3). But entry-rate-driven employment (jobs created by births) was not significant (p=0.13), nor was young-firm employment share (p=0.82).



\*\*Interpretation:\*\* IPP intensity correlates with \*numbers\* of entering establishments, not the scale of entry. This weakens the mechanistic story. A small-scale entry decline is economically meaningful for dynamism but leaves reverse-causality concerns unresolved (do high-IPP sectors have structurally low churn for other reasons, and therefore can't support new entrants?).



\## 4. Why Identification Failed



Three specifications tested the causal hypothesis:



\### 4a. Event Study (No Effect)



Centered 2008 GFC, 2001 dot-com, 2020 COVID shocks on pre-shock intangible intensity (binary or standardized). If intangibles dampen crisis recovery differentially, we'd see negative post-event coefficients for high-intensity sectors. We don't. Zero significant post-shock effects across seven specifications.



\*\*Why this matters:\*\* The hypothesis could have been a slow secular trend \*or\* a crisis-triggered mechanism. The null event-study result eliminates the latter.



\### 4b. Panel Regression + Sector Trends (Identification Breaks)



Main result: IPP share, −0.057 (p=0.376). Add sector-specific linear trends: IPP share, +0.142 (p=0.025), sign flips.



This is the killer. Treatment varies only at 19-sector level. Cannot include sector-by-year fixed effects without loss of identifying variation. Any slow-moving sectoral characteristic (aging population, regulatory environment, consolidation) correlated with IPP intensity will be confounded.



The sector-trend specification absorbs these unobservables; the sign reversal suggests the raw panel association was driven by cross-sector long-run comovement (e.g., "high-tech sectors naturally have lower churn"), not within-sector substitution dynamics.



\### 4c. Placebo Test (Specificity, But No Robustness)



Equipment (+0.091, p=0.167) and structures (−0.044, p=0.401) flip signs relative to IPP. This supports asset-type specificity: the effect is not "capital-intensive sectors churn less." But equipment itself becomes insignificant with sector trends (−0.046, p=0.327).



\## 5. What Would Fix It



Establishment-level data (Census LBD or RE-LBD) would allow:

\- IPP intensity to vary within sector-year (across establishments)

\- Sector-by-year fixed effects without collinearity

\- Difference-in-differences logic: compare JRR trends within sector across time, within firms across capital composition changes



Firm-level intangible capital data (R\&D, capitalized software, purchased intangibles) would further isolate the mechanism.



Public data does not support this level of identification.



\## 6. Robustness



\### Decomposition across six period cuts

Within-share: 102.4%, 105.6%, 106.2%, 106.5%, 109.3%, 104.9%. Conclusion is stable.



\### Multi-outcome stability

All 21 tests (7 outcomes × 3 specs) reported. No selective reporting. Entry rate remains strongest; all disappear with trends.



\### Data quality

\- BEA row alignment verified (00\_verify\_data.py)

\- BDS denom vs. emp audit passed

\- BEA three-asset-class sum-to-total: 0.00% deviation (1997+)



\---



\*\*Conclusion:\*\* The aggregate divergence is real. Sectoral patterns have the predicted direction. Entry rates respond most. But identification of the causal channel is blocked by inability to control sector-specific trends with public data.

