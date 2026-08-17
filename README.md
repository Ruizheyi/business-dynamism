<h1>Declining Business Dynamism and Intangible Capital</h1>

<p>Replication and extension of Decker et al. (2016), the starting reference for this project, using BDS and BEA data. Tests whether intangible capital substitution explains why U.S. business dynamism has been declining.</p>

<p>Short version: intangible investment and business dynamism diverge sharply, and the divergence is entirely within sectors. Using the staggered adoption of state R&amp;D tax credits, establishment entry rises after adoption, but the response does not vary with how intangible-intensive a sector is — under either of two independent exposure measures. Whatever the credit does, it does not work by raising intangible intensity where intangibles matter most.</p>

<hr>

<h2>What this is about</h2>

<p>U.S. business dynamism has been falling since the 1980s. Decker et al. showed it's happening across multiple dimensions: job reallocation rates down, establishment entry rates down, young-firm employment shares down. They answered "what" but not "why."</p>

<p>This project tests one candidate mechanism: firms increasingly substitute software and other intangible capital for labor churn. Do the sectoral patterns support it? And when a policy makes intangible capital cheaper, does dynamism respond?</p>

<hr>

<h2>Data</h2>

<p>Job flows and employment come from Census BDS (3-digit NAICS × firm age, 1978–2023; state × sector, 1978–2023). National investment figures come from BEA NIPA Table 5.3.5. By-sector investment data come from BEA Fixed Assets Tables 3.7* (IPP, equipment, structures by industry, 1947–2023). State R&amp;D tax credit variation comes from the Panel Database on Incentives and Taxes (Bartik 2017, Upjohn Institute), 33 states × 45 industries × 1990–2015.</p>

<p><b>Job reallocation rate:</b> (job creation + destruction) / BDS denominator × 100, excluding age-0 entrants. The denominator is the Davis-Haltiwanger-Schuh average of current and lagged employment, not <code>emp</code>.</p>

<p><b>Young-firm share:</b> employment in firms age ≤5 / total employment.</p>

<p><b>Intangible intensity:</b> IPP (intellectual property products—R&amp;D, software, entertainment originals) as % of total fixed investment per sector. Software-only numbers aren't published by industry, so IPP is the finest detail available.</p>

<p><b>Entry and exit rates</b> in the state-by-sector work are rebuilt from establishment counts using the DHS denominator rather than taken from the published rate column, because aggregating published sector rates would weight accommodation the same as manufacturing. The rebuilt series correlates 0.9998 with the published one.</p>

<p>Raw data is gitignored. Sources and export settings are documented in the script headers.</p>

<hr>

<h2>What the descriptive work found</h2>

<p>Job reallocation fell from 30.7% (1997) to 23.1% (2023). Software's share of private fixed investment went from 6.2% to 13.0%. Indexed to 1997, the gap reaches 136 points.</p>

<p>The strongest result is the shift-share decomposition. Total change −5.56pp, within-sector −5.83pp (104.9%), between-sector +0.27pp. The within share stays between 102% and 109% across six alternative period cuts. This is an accounting identity, so it holds regardless of what you believe about causality: <b>the decline is not about the economy shifting toward less dynamic industries. It's happening inside industries.</b></p>

<p>Seven dynamism outcomes were regressed on sector intangible intensity, three specifications each, all 21 reported:</p>

<table>
<tr><th>Outcome</th><th>β</th><th>p</th><th>β lag3</th><th>p</th><th>β + trends</th><th>p</th></tr>
<tr><td>Job reallocation rate</td><td>−0.057</td><td>0.376</td><td>−0.148</td><td>0.092</td><td>+0.142</td><td>0.025</td></tr>
<tr><td><b>Establishment entry rate</b></td><td><b>−0.068</b></td><td><b>0.035</b></td><td><b>−0.068</b></td><td><b>0.015</b></td><td>−0.002</td><td>0.939</td></tr>
<tr><td>Establishment exit rate</td><td>−0.015</td><td>0.455</td><td>−0.044</td><td>0.041</td><td>+0.023</td><td>0.572</td></tr>
<tr><td>Firm death rate</td><td>−0.006</td><td>0.740</td><td>−0.037</td><td>0.063</td><td>+0.010</td><td>0.722</td></tr>
<tr><td>Young-firm employment share</td><td>−0.016</td><td>0.822</td><td>−0.049</td><td>0.465</td><td>+0.014</td><td>0.635</td></tr>
<tr><td>Job creation from births</td><td>−0.033</td><td>0.130</td><td>−0.033</td><td>0.087</td><td>+0.007</td><td>0.754</td></tr>
<tr><td>Job destruction from deaths</td><td>−0.006</td><td>0.744</td><td>−0.036</td><td>0.088</td><td>+0.046</td><td>0.031</td></tr>
</table>

<p>Entry counts move while entry-driven employment doesn't (p=0.13) and young-firm share doesn't (p=0.82). Everything dies once sector-specific trends go in.</p>

<p><b>The honest limit here:</b> with 19 BEA sectors and treatment constant within sector-year, sector-by-year fixed effects are impossible. Substitution can't be separated from any other slow-moving sectoral trend. That's what the causal half is for.</p>

<hr>

<h2>What the causal work found</h2>

<p>The variation problem is a geography problem, not a data-source problem. BDS publishes state × sector tables, which makes state×year and sector×year fixed effects possible at the same time.</p>

<p>The shock is the staggered adoption of state R&amp;D tax credits. 20 states adopt inside the window (MA 1991 through FL 2012), 7 never adopt and serve as controls, 6 already had credits in 1990 and are dropped as left-censored. The sample runs 1990–2015, which is where PDIT observes whether a credit is actually in force.</p>

<p><b>Treatment is not absorbing.</b> Four states repealed after adopting: Missouri (no credit 2005–2015, eleven of fifteen post years), Texas (2008–2013), Michigan (2012–2015), Washington (2015). Treatment is coded as actual credit status, not as an indicator for years after adoption.</p>

<p>Two things are estimated. The <b>aggregate effect</b> uses state-sector and sector-year fixed effects with adoption itself as treatment. The <b>sector gradient</b> adds state-year fixed effects and interacts treatment with sector exposure. State-year fixed effects absorb the aggregate effect by construction, so the gradient answers only whether the response is larger where intangibles matter more. All inference is wild cluster bootstrap-t, Rademacher weights, null imposed, clustered on state.</p>

<h3>The aggregate effect, and why no single window is reported</h3>

<p>An earlier version picked one event window and defended it. That was wrong: the window was chosen after seeing the full-window result, and the stated justification did not uniquely imply it. All twenty-five windows are now reported instead. Entry rate, coefficient by window endpoint:</p>

<table>
<tr><th></th><th>to k=0</th><th>to k=+1</th><th>to k=+2</th><th>to k=+3</th><th>to k=+4</th></tr>
<tr><td>from k=−5</td><td>+0.119</td><td>+0.220</td><td>+0.266*</td><td>+0.301*</td><td>+0.301*</td></tr>
<tr><td>from k=−4</td><td>+0.143</td><td>+0.245*</td><td>+0.288*</td><td>+0.323*</td><td>+0.314*</td></tr>
<tr><td>from k=−3</td><td>+0.102</td><td>+0.209</td><td>+0.251*</td><td>+0.286*</td><td>+0.278*</td></tr>
<tr><td>from k=−2</td><td>+0.089</td><td>+0.190</td><td>+0.236*</td><td>+0.273*</td><td>+0.253*</td></tr>
<tr><td>from k=−1</td><td>+0.029</td><td>+0.152</td><td>+0.203*</td><td>+0.244*</td><td>+0.219*</td></tr>
</table>

<p>* significant at 5%. The estimate is determined almost entirely by the right endpoint and is close to insensitive to the left. Sixteen of twenty-five are significant, none flips sign, and the coefficient rises monotonically as later post-periods enter. Entry rate mean is 12.0pp, so +0.25 is about 2.1%.</p>

<p>The event study explains that pattern — the effect accumulates rather than jumping:</p>

<pre>
k = −3    −0.063   (se 0.143)
k = −2    −0.049   (se 0.092)
k =  0    +0.053   (se 0.096)
k = +1    +0.267   (se 0.132, t 2.02)
k = +2    +0.336   (se 0.133, t 2.52)

joint Wald on leads: chi2(2) = 0.30, p = 0.859
</pre>

<p>Windows ending at k=0 or k=+1 are the ones that come out insignificant, and those are exactly the windows containing only the years before the effect appears.</p>

<p><b>Other outcomes are zero everywhere.</b> Exit rate, job creation rate, and job reallocation do not respond in any specification. Combined with the Phase 1 finding that entry counts moved while entry-driven employment did not, the pattern across both halves of the project is that <b>establishment counts are policy-responsive and employment reallocation is not.</b></p>

<h3>The sector gradient is zero under two independent measures</h3>

<p>An early version sorted sectors on PDIT exposure alone — how much R&amp;D credit money a sector can claim. But the hypothesis is about intangible capital, which is a different thing. A second measure was built from BEA: IPP over total fixed investment, 1985–1989, entirely pre-treatment. The two correlate only 0.573 and rank sectors very differently — professional services is 1st on the BEA measure and 3rd on PDIT; arts and entertainment is 2nd on BEA and 14th on PDIT.</p>

<table>
<tr><th>Specification</th><th>β</th><th>p</th><th>95% CI</th><th>VIF</th></tr>
<tr><td>PDIT exposure alone</td><td>+0.015</td><td>0.914</td><td>[−0.279, +0.318]</td><td>—</td></tr>
<tr><td>BEA exposure alone</td><td>+0.038</td><td>0.772</td><td>[−0.223, +0.286]</td><td>—</td></tr>
<tr><td>Both, PDIT coefficient</td><td>−0.014</td><td>0.952</td><td>[−0.467, +0.422]</td><td>1.49</td></tr>
<tr><td>Both, BEA coefficient</td><td>+0.045</td><td>0.784</td><td>[−0.309, +0.427]</td><td>1.49</td></tr>
</table>

<p>Eleven gradient estimates in total, across continuous and binary exposure, four reference periods, several windows, corrected and uncorrected treatment coding, with and without never-treated states. Range −0.32 to +0.78, none distinguishable from zero. Individual sector coefficients bear no relation to exposure: professional services (PDIT exposure 0.224) is −0.145, manufacturing (1.000) is −0.178, accommodation (0.007) is +0.169.</p>

<p><b>Reading.</b> Entry responds to adoption. That response does not vary with sector intangible exposure, which is what the substitution mechanism requires. This says nothing about the mechanism behind the aggregate effect — establishing that would require ruling out the state-wide channels, not merely observing that the sector channel is absent.</p>

<hr>

<h2>A coefficient that passed four tests and failed the fifth</h2>

<p>This is the most methodologically useful thing in the project.</p>

<p>Using equipment intensity as a placebo, the gradient on job creation came out at <b>+1.697, p=0.012</b>. It then passed randomisation across sectors with 5000 permutations (p=0.0074), leave-one-sector-out (zero sign flips in sixteen, range +1.57 to +2.25), fake adoption timing with 1000 draws (p=0.041), and controls for pre-1990 cyclical sensitivity, pre-period growth, structures intensity and both intangible measures — under which the coefficient <i>rose</i> to +1.65, p=0.005.</p>

<p>Then the dynamic specification:</p>

<pre>
k = −3   −1.591  (t −2.65)
k = −2   −0.201
k =  0   −0.080
k = +2   +0.675
k = +3   +1.613  (t 3.01)
k = +4   +0.152  (t 0.23)

joint Wald on leads: chi2(3) = 10.97   vs 5% critical value 7.81 → rejects
</pre>

<p>More than half the movement happens before adoption, and the post path spikes at k=3 then vanishes. The coefficient is a pre-existing differential trend.</p>

<p><b>The general lesson:</b> randomisation, leave-one-out, and fake-timing tests all operate on the same post-minus-pre mean difference. They can establish that a number is not random noise. They cannot establish that it is a treatment effect, because none of them can see a pre-trend.</p>

<hr>

<h2>Things that could be wrong with this</h2>

<p><b>Standard staggered-DiD estimators do not apply.</b> Callaway–Sant'Anna assumes treatment is absorbing, and four states repeal. The relevant framework for non-absorbing treatment is de Chaisemartin–D'Haultfœuille. Not implemented.</p>

<p><b>Correcting the treatment coding lowered the estimate rather than raising it</b>, from +0.225 to +0.149 on the full window, the opposite of what attenuation bias predicts. That means entry ran relatively high during the repeal years. With only 22 such state-years this may be chance, but it doesn't support the positive-effect reading.</p>

<p><b>Why states repeal is unexamined.</b> Missouri 2005, Texas 2008 and Michigan 2012 all fall in fiscally stressed periods. If repeal is endogenous to state economic conditions, adoption probably is too, and that threatens exogeneity more than any individual coefficient does.</p>

<p><b>Cohort differences are not established.</b> Early adopters (1991–98) give +0.195 and late adopters (2000–12) give +0.353, but a direct test of equality gives +0.13 with p=0.64. The data do not establish heterogeneity across adoption cohorts.</p>

<p><b>Identifying variation is concentrated.</b> For the gradient, effective cluster count is 11.7 against a nominal 33, the top five states hold 58% of residual treatment variance, and sectors 31-33 and 51 hold 84%. The aggregate is better behaved at 25.8 effective clusters.</p>

<p><b>Event-time coverage is uneven.</b> The sample runs 1990–2015 but MA and IL adopt in 1991, so k=−5 falls outside the sample for them. Leads are estimated off 14–18 states, lags off 19–20. This is why the window grid is reported rather than one window.</p>

<p><b>PDIT's industry dimension is model-imputed.</b> States don't generally set different R&amp;D credit rates by industry; the cross-industry spread comes from applying industry characteristics inside Bartik's hypothetical-firm simulation. The cross-industry ranking is at least stable across 1992–2015.</p>

<p><b>R&amp;D isn't the intangible capital in the story.</b> The mechanism is about software and organizational capital. Qualified research expenses are mostly wages, so an R&amp;D credit partly subsidises research employment, which runs opposite to the substitution prediction.</p>

<p><b>Sector coverage.</b> PDIT has no agriculture, mining or utilities, so 16 of 19 BDS sectors are used. Value-added weights are national and state-invariant.</p>

<p><b>Everything is hand-written.</b> Fixed-effect absorption, cluster covariance and the wild bootstrap are implemented directly with numpy, not through an econometrics package. Absorption was verified to converge in 23–25 iterations with post-absorption group means below 1e-10, but no cross-check against an established implementation has been run.</p>

<p><b>Why not TCJA §174.</b> It looks like the obvious shock but BDS ends in 2023, leaving two post years both inside the post-COVID reallocation surge. It was reversed retroactively by OBBBA in July 2025 via new §174A, so it only ever bound for tax years 2022–2024, with repeal attempts live throughout. And for a profitable firm the NPV cost of five-year amortization is roughly 2–3 cents per dollar of R&amp;D.</p>

<hr>

<h2>On comparison with Fazio, Guzman and Stern (2020)</h2>

<p>FGS use the same PDIT data in a state-level difference-in-differences and find R&amp;D credits raise high-quality new firm formation by about 20% over ten years. An earlier version of this README scaled the confidence intervals here against that figure. <b>That comparison does not hold.</b> Their outcome is business registration data including non-employer firms and quality-weighted; the outcome here is BDS employer establishment entry. Their 20% is a cumulative ten-year effect; the estimates here are average post-adoption effects.</p>

<p>The relationship between the two designs is that state-year fixed effects absorb, by construction, exactly the parameter FGS estimate. The gradient result therefore neither confirms nor contradicts them.</p>

<hr>

<h2>Scripts</h2>

<p>Run everything from the repository root, not from inside <code>scripts/</code> — paths are relative to the working directory.</p>

<pre>
00_verify_data.py             BEA row alignment and additivity checks
06_build_panel.py             national series
07_plot_scissor.py            Figure 1, the scissor gap
11_sector_intensity.py        sector IPP intensity, 513 rows
13_event_specs.py             event studies around 2001, 2008, 2020
15_panel_design.py            panel regressions
17_placebo.py                 equipment and structures placebos
18_decomposition.py           shift-share decomposition + Figure 2
21_decomp_sensitivity.py      six alternative period cuts
22_multi_outcome.py           7 outcomes x 3 specifications

30_variance_diagnostic.py     does the outcome survive the target FE? 26.8% does
31_build_pdit_panel.py        merge PDIT onto BDS, power calculation
32_treatment_diagnostics.py   treatment separability, variation concentration
33_baseline_and_robustness.py TWFE, leave-one-state-out, lead test
34_outlier_check.py           influence and winsorising checks
36_event_study_wcb.py         event study with wild bootstrap
37_ref_period.py              reference period sensitivity
38_aggregate_effect.py        aggregate alongside gradient
39_reconcile.py               puts them on a common scale
40_binary_gradient.py         binary split done correctly, placebo sectors
41_fix_treatment.py           treatment recoded as non-absorbing
42a_build_bea_exposure.py     BEA pre-treatment intangible intensity
42_exposure_horserace.py      two exposure measures, proper Wald pre-trend
43_equip_check.py             equipment placebo, first pass
44_equip_deepcheck.py         5000 permutations, leave-one-sector-out
45_equip_diagnose.py          cycle, scale, structures, generosity
46_equip_final.py             capital composition, fake timing, event shape
47_aggregate_event.py         aggregate event study
48_pretrend_deep.py           reference sensitivity, linear trend, sensitivity
49_reconcile_specs.py         resolves a 3.3x gap between two specifications
50_final_robustness.py        window coverage, adoption cohorts
51_audit_and_window.py        all 25 windows, convergence, cohort equality test
</pre>

<p>Scripts 33 and 40 overlap on purpose. Script 33 estimated the sector split as separate subsample regressions; with two sectors in the high group, state-year fixed effects leave only the manufacturing-minus-information difference and the estimator degenerates. That is where a spurious coefficient of 2807 came from. Script 40 redoes it as a dummy interacted in the full sample. Scripts containing errors that were later corrected are kept rather than deleted, so both the error and the fix stay visible.</p>

<pre>
py -m pip install -r requirements.txt
py scripts/00_verify_data.py
</pre>

<hr>

<h2>References</h2>

<p>Bartik, T. (2017). A New Panel Database on Business Incentives for Economic Development. Upjohn Institute.</p>

<p>Callaway, B. &amp; Sant'Anna, P. (2021). Difference-in-differences with multiple time periods. <i>Journal of Econometrics</i> 225(2).</p>

<p>Cameron, A.C., Gelbach, J. &amp; Miller, D. (2008). Bootstrap-based improvements for inference with clustered errors. <i>Review of Economics and Statistics</i> 90(3).</p>

<p>de Chaisemartin, C. &amp; D'Haultfœuille, X. (2020). Two-way fixed effects estimators with heterogeneous treatment effects. <i>American Economic Review</i> 110(9).</p>

<p>Decker, R., Haltiwanger, J., Jarmin, R. &amp; Miranda, J. (2016). Declining business dynamism: what we know and the way forward. <i>American Economic Review: Papers &amp; Proceedings</i> 106(5), 203–207.</p>

<p>Fazio, C., Guzman, J. &amp; Stern, S. (2020). The impact of state-level R&amp;D tax credits on the quantity and quality of entrepreneurship. <i>Economic Development Quarterly</i> 34(2).</p>

<p>Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. <i>Journal of Econometrics</i> 225(2).</p>

<p>Rambachan, A. &amp; Roth, J. (2023). A more credible approach to parallel trends. <i>Review of Economic Studies</i> 90(5).</p>

<p>Wilson, D. (2009). Beggar thy neighbor? The in-state, out-of-state, and aggregate effects of R&amp;D tax credits. <i>Review of Economics and Statistics</i> 91(2).</p>