<h1>Technical Note: Identifying the Intangible Capital Channel in Declining Business Dynamism</h1>

<h2>Measurement Considerations</h2>

<p>The job reallocation rate derives from BDS data: job creation and destruction, normalized by an employment denominator. The choice of denominator proves consequential.</p>

<p>BDS offers two measures: end-of-period employment and the DHS average (current and prior year, divided by 2). Literature convention favors the DHS average. Initial versions of this analysis relied on end-of-period employment, producing sectoral rankings that diverged from the standard and rates that appeared artificially elevated in sectors with expanding employment. Auditing both approaches revealed systematic differences in cross-industry comparisons, likely reflecting that BDS suppression operates at the row level (entire establishments) rather than partial records.</p>

<p>The treatment of new establishments warrants mention. JRR excludes age-0 firms consistent with original methodology. Young-firm employment shares, by contrast, include age-0 entrants in both numerator and denominator, following standard practice. The age conventions differ between measures, but sensitivity analysis suggests the substantive findings do not depend on this choice.</p>

<h2>Aggregation Strategy: From 3-Digit to Sector-Year</h2>

<p>Treatment data (IPP intensity) originates from BEA Fixed Assets at the 2-digit BEA industry level, yielding 19 sectors. Job data derive from BDS at 3-digit NAICS. Preliminary analysis conducted regressions at the 3-digit level by assigning each 3-digit industry the IPP intensity of its parent sector. This approach encountered limitations.</p>

<p>Treatment variation becomes zero within the sector-year cell, making 3-digit fixed effects redundant when sector fixed effects are present. The approach also introduces a clustering problem: approximately 86 industries but only 19 cluster groups, yielding a singular clustered covariance matrix with roughly 40 parameters.</p>

<p>Aggregation to the sector-year level resolves both issues. The resulting dataset contains 513 observations with meaningful treatment variation. The cost is acceptable: 3-digit detail remains embedded in the aggregation weights, influencing which establishments contribute to each sector-year observation, though it does not directly enter the identifying variation in the regression.</p>

<h2>Identifying the Most Important Outcome</h2>

<p>Job reallocation combines job creation and destruction. If intangible capital primarily raises barriers to establishment entry by increasing the capital scale required for competitive software stacks, suppressed births might coexist with unchanged incumbent churn.</p>

<p>Shift-share analysis indicated that 105% of aggregate JRR decline originated within sectors, suggesting that sectoral composition shifts offer minimal explanatory power and motivating disaggregated outcome analysis.</p>

<p>Testing seven separate measures identified entry rates as carrying the strongest association (p=0.035 contemporaneous, p=0.015 at three-year lag). Entry-driven employment creation, however, showed weaker results (p=0.13), as did young-firm employment shares (p=0.82). The pattern is suggestive rather than conclusive: IPP intensity correlates with the number of entering establishments rather than their average scale. Such a decline carries implications for dynamism measurement but does not rule out confounding mechanisms.</p>

<h2>Sources of Identification Challenge</h2>

<p>Three identification strategies merit discussion. Event study analysis centered on 2001 dot-com, 2008 GFC, and 2020 COVID episodes. The expectation was that high-IPP-intensity sectors would exhibit differential post-shock recovery patterns if intangible capital buffers or amplifies cyclical effects. Across seven specifications, post-event coefficients remained statistically indistinguishable from zero. This finding suggests, if anything, that the mechanism operates along secular rather than cyclical dimensions.</p>

<p>The panel specification reveals a more serious constraint. The baseline point estimate is IPP share, −0.057 (p=0.376). When sector-specific linear trends enter the model, the coefficient reverses: IPP share, +0.142 (p=0.025).</p>

<p>This reversal reflects a fundamental identification problem. Treatment varies only at the sector level (19 groups). Sector-by-year fixed effects cannot be included without absorbing all identifying variation. Consequently, any slow-moving sectoral characteristic—demographic aging, regulatory evolution, industry consolidation, structural labor-force composition—that correlates with both IPP investment and low churn will bias the estimate.</p>

<p>The sector-trend specification approximates absorption of these time-invariant or slowly-varying confounders. The sign reversal suggests the raw panel association primarily reflects cross-sector long-run comovement (technology-intensive sectors structurally exhibit lower churn) rather than within-sector substitution dynamics between capital types.</p>

<p>A secondary diagnostic, the asset-type placebo test, does indicate specificity: equipment investment correlates positively with JRR (+0.09 contemporaneous, +0.16 lagged), opposite to IPP. This pattern rules out the hypothesis that capital-intensive sectors generically exhibit low churn. Yet the equipment effect also attenuates with sector trends, suggesting fragility in even this more refined result.</p>

<h2>Complementary Approaches and Extensions</h2>

<p>Establishment-level data from the Census Bureau (LBD or RE-LBD) would permit treatment variation within sector-year, enabling sector-by-year fixed effects. Within-sector comparison across time, conditional on establishment identity, offers an alternative identifying strategy that would not require strong assumptions about cross-sector comovement.</p>

<p>Firm-level accounting data on intangible capital stocks—capitalized R&D, software, and purchased intangibles—would allow direct measurement rather than reliance on investment flows as a proxy. Administrative tax records or specialized surveys could provide such detail.</p>

<p>Alternative samples merit exploration: restricting to sectors with historically high or low intangible intensity and examining whether the relationship stabilizes; splitting by establishment size to test whether the pattern concentrates among young or small firms; or examining entry versus survival separately using establishment-level flow data where accessible.</p>

<p>Public sector-level data as currently available does not appear sufficient to isolate the causal channel. Identification fundamentally depends on resolving confounding with sectoral trends that operate over decades.</p>

<h2>Validation and Robustness</h2>

<p>Shift-share decomposition proves consistent across six alternative period definitions, with within-sector shares ranging from 102% to 109%. All 21 empirical tests (seven outcomes, three specifications each) are reported without selection. Entry rates maintain the strongest signal; all specifications weaken or reverse when sector trends are added. Data quality checks confirm row alignment across BEA tables, validity of the BDS denominator choice, and <0.01% residual across the asset-class decomposition (IPP, equipment, structures as shares of total fixed assets) during the analysis period.</p>