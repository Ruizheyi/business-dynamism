<h1>Technical Note: Identifying the Intangible Capital Channel in Declining Business Dynamism</h1>



<h2>Data and Denominators</h2>



<p>The job reallocation rate is constructed from BDS data: job creation plus destruction, divided by an employment denominator, times 100. But which denominator matters.</p>



<p>BDS publishes two versions: end-of-period employment and the DHS average (current year plus prior year, divided by 2). The literature standard is the DHS average. Early versions of this project used end-of-period employment, which produced different sectoral rankings and artificially inflated rates in sectors with high employment growth. The error was caught through an audit comparing the two. Since BDS suppression is row-wise (whole establishments, not partial), using the wrong denominator systematically biases cross-industry comparisons.</p>



<p>One other choice: JRR excludes age-0 firms (created in the current year). This follows the original Decker et al. convention. The young-firm employment share, by contrast, includes age-0 entrants in both numerator and denominator. The two indicators have different age conventions; robustness checks confirm the results don't hinge on this.</p>



<h2>Why Sector-Year, Not 3-Digit NAICS</h2>



<p>Treatment data (IPP intensity) comes from BEA Fixed Assets, published at the 2-digit BEA industry level (19 sectors). Job data come from BDS at 3-digit NAICS. Early exploration ran regressions at the 3-digit level, treating each 3-digit industry as having the IPP intensity of its parent sector. This created problems.</p>



<p>First, there's no identifying variation. IPP intensity is constant within sector-year, so adding 3-digit fixed effects alongside sector fixed effects just erases the signal. Second, there's the cluster issue: 86 industries but only 19 cluster groups (the BEA sectors). With \~40 parameters and 19 clusters, the clustered covariance matrix becomes singular.</p>



<p>The solution is to aggregate to sector-year level. This gives 513 observations. Treatment varies at the sector-year level, no collinearity, and 19 clusters is still few but mathematically workable.</p>



<p>The trade-off is clear: 3-digit granularity is preserved in the data construction (it affects the weighting in the aggregation) but not in the regression. The 3-digit detail determines which establishments are weighted, not the identifying variation.</p>



<h2>Which Outcome Matters</h2>



<p>The job reallocation rate is a composite: job creation plus job destruction. If intangible capital primarily deters entry of new firms (raising barriers to scale software stacks), it might suppress job creation from births but leave job destruction from incumbent churn unchanged.</p>



<p>The shift-share decomposition revealed that 105% of the aggregate JRR decline is within-sector, eliminating sectoral reallocation as an explanation and motivating tests of finer-grained outcomes.</p>



<p>Regressing seven separate outcomes showed entry rate to be the strongest signal (p=0.035 contemporaneous, p=0.015 lag-3). But entry-rate-driven employment (jobs created by births) was not significant (p=0.13), nor was young-firm employment share (p=0.82).</p>



<p>This matters. IPP intensity correlates with numbers of entering establishments, not the scale of entry. A small-scale entry decline is economically meaningful for dynamism but leaves reverse-causality concerns unresolved. Do high-IPP sectors have structurally low churn for reasons unrelated to capital composition?</p>



<h2>Why Identification Failed</h2>



<p>Three specifications tested the causal hypothesis. The event study centered on 2008 GFC, 2001 dot-com, and 2020 COVID. If intangibles dampen crisis recovery differentially, high-intensity sectors should show negative post-event coefficients. They don't. Zero significant effects across seven specifications. This rules out crisis-triggered mechanisms; the hypothesis could be slow secular trend or cyclical acceleration, and the null event-study rules out the latter.</p>



<p>The core problem emerges in the panel specification. Main result: IPP share, −0.057 (p=0.376). Add sector-specific linear trends: IPP share, +0.142 (p=0.025). The sign flips.</p>



<p>This is fatal for identification. Treatment varies only at 19 sectors. Cannot include sector-by-year fixed effects without loss of identifying variation. Any slow-moving sectoral characteristic correlated with IPP intensity will confound the estimate: aging population, regulatory environment, consolidation, structural labor-force composition changes. All of these could drive both IPP investment and low churn, independently.</p>



<p>The sector-trend specification absorbs these unobservables. The sign reversal suggests the raw association was driven by cross-sector long-run comovement (high-tech sectors naturally have lower churn), not within-sector substitution dynamics.</p>



<p>The placebo test (equipment and structures) shows asset-type specificity: equipment flips sign relative to IPP. This is good; it rules out "capital-intensive sectors churn less." But equipment itself becomes insignificant with sector trends. Specificity is demonstrated but fragile.</p>



<h2>What Would Fix It</h2>



<p>Establishment-level data (Census LBD or RE-LBD) would allow treatment to vary within sector-year across establishments. This enables sector-by-year fixed effects without collinearity. You could then compare JRR trends within sector across time, asking whether establishments that intensify intangible capital show different churn dynamics. That's the difference-in-differences logic you need.</p>



<p>Firm-level intangible capital data (R\&D, capitalized software, purchased intangibles) would further isolate the mechanism. Public data does not support this level of identification.</p>



<h2>Robustness</h2>



<p>Decomposition holds across six period cuts: within-share ranges 102–109%. All 21 tests reported (7 outcomes × 3 specs). No selective reporting. Entry rate remains strongest; all disappear with trends. BEA row alignment verified, BDS denominator audit passed, and the three asset classes (IPP, equipment, structures) sum to total fixed assets with \&lt;0.01% deviation in the analysis period (1997+).</p>

