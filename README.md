<h1>Declining Business Dynamism and Intangible Capital</h1>

<p>Replication and extension of Decker et al. (2016) using BDS and BEA data. Tests whether intangible capital substitution explains why U.S. business dynamism has been declining.</p>

<p>Short version: intangible investment and business dynamism diverge sharply, and the divergence is entirely within sectors. Using the staggered adoption of state R&amp;D tax credits, establishment entry does respond to the policy in the five years after adoption, but the response does not vary with how R&amp;D-exposed a sector is. Whatever the credit does, it does not work by raising intangible intensity where intangibles matter most.</p>

<hr>

<h2>What this is about</h2>

<p>U.S. business dynamism has been falling since the 1980s. Decker et al. showed it's happening across multiple dimensions: job reallocation rates down, establishment entry rates down, young-firm employment shares down. They answered "what" but not "why."</p>

<p>This project tests one candidate mechanism: firms increasingly substitute software and other intangible capital for labor churn. Do the sectoral patterns support it? And when a policy makes intangible capital cheaper, does dynamism respond?</p>

<hr>

<h2>Data</h2>

<p>Job flows and employment come from Census BDS (3-digit NAICS × firm age, 1978–2023; state × sector, 1978–2023). National investment figures come from BEA NIPA Table 5.3.5 (software and IPP spending, 1947–2023). By-sector investment data come from BEA Fixed Assets Tables 3.7* (IPP, equipment, structures by industry, 1947–2023). State R&amp;D tax credit variation comes from the Panel Database on Incentives and Taxes (Bartik 2017, Upjohn Institute), 33 states × 45 industries × 1990–2015.</p>

<p><b>Job reallocation rate:</b> (job creation + destruction) / BDS denominator × 100, excluding age-0 entrants.</p>

<p><b>Young-firm share:</b> employment in firms age ≤5 / total employment.</p>

<p><b>Intangible intensity:</b> IPP (intellectual property products—R&amp;D, software, entertainment originals) as % of total fixed investment per sector. Software-only numbers aren't published by industry, so IPP is the finest detail available.</p>

<p><b>Entry and exit rates</b> in the state-by-sector work are rebuilt from establishment counts using the DHS denominator rather than taken from the published rate column, because aggregating published sector rates would weight a tiny sector the same as manufacturing. The rebuilt series correlates 0.9998 with the published one.</p>

<p><b>Credit exposure:</b> present value of R&amp;D credits over a 20-year simulation as a % of industry value added, aggregated from PDIT's 45 industries to BDS sectors with value-added weights, then normalised so manufacturing = 1.</p>

<p>Raw data is gitignored. <code>data/raw/MANIFEST.md</code> lists every file with its source, export settings and expected dimensions.</p>

<hr>

<h2>What the descriptive work found</h2>

<p>Job reallocation fell from 30.7% (1997) to 23.1% (2023). Software's share of private fixed investment went from 6.2% to 13.0%. Indexed to 1997, the gap between them reaches 136 points. That's Figure 1.</p>

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

<p>Entry rate is the strongest signal. But entry <i>counts</i> move while entry-driven <i>employment</i> doesn't (p=0.13) and young-firm share doesn't (p=0.82), so whatever is there is economically small. And everything dies once sector-specific trends go in.</p>

<p>Event studies around 2001, 2008 and 2020 find nothing — zero significant post-event coefficients in six of seven specifications. As a placebo, equipment intensity flips sign relative to IPP (+0.091 contemporaneous, +0.159 at lag 3, p=0.040) and structures sits near zero, so the asset-type specificity holds. Equipment dies with trends too.</p>

<p><b>The honest limit here:</b> with 19 BEA sectors and treatment constant within sector-year, sector-by-year fixed effects are impossible. Substitution can't be separated from any other slow-moving sectoral trend. That's what the causal half of the project is for.</p>

<hr>

<h2>What the causal work found</h2>

<p>The variation problem is a geography problem, not a data-source problem. BDS publishes state × sector tables, which expands the panel from 513 cells to 44,574 and — the part that matters — makes state×year and sector×year fixed effects possible at the same time.</p>

<p>The shock is the staggered adoption of state R&amp;D tax credits. 20 states adopt inside the window (MA 1991 through FL 2012), 7 never adopt and serve as controls, 6 already had credits in 1990 and get dropped as left-censored. The sample runs 1990–2015, which is where PDIT observes whether a credit is actually in force.</p>

<p>Two things are estimated and reported side by side. The <b>aggregate effect</b> uses state-sector and sector-year fixed effects, with adoption itself as the treatment. The <b>sector gradient</b> adds state-year fixed effects and interacts treatment with sector exposure; state-year fixed effects absorb the aggregate effect by construction, so this coefficient answers only whether the response is larger where R&amp;D exposure is higher. All inference is wild cluster bootstrap-t, Rademacher weights, null imposed.</p>

<p>Entry rate, restricted to a balanced event window of five years either side of adoption:</p>

<table>
<tr><th>Estimate</th><th>β</th><th>p</th><th>95% CI</th><th>% of mean</th></tr>
<tr><td><b>Aggregate effect</b></td><td><b>+0.301</b></td><td><b>0.018</b></td><td><b>[+0.050, +0.537]</b></td><td><b>+0.4% to +4.5%</b></td></tr>
<tr><td>Sector gradient, continuous</td><td>+0.015</td><td>0.933</td><td>[−0.286, +0.305]</td><td>−2.4% to +2.5%</td></tr>
<tr><td>Sector gradient, binary</td><td>+0.061</td><td>0.663</td><td>[−0.223, +0.340]</td><td>−1.9% to +2.8%</td></tr>
</table>

<p>Over the full post window rather than a balanced one the aggregate halves to +0.149 (p=0.307) and the confidence interval covers zero, so the effect is concentrated in the first few years after adoption and dilutes later. Exit rate and job reallocation are zero in every specification tried.</p>

<p>The gradient has now been estimated eleven ways — continuous and binary exposure, four reference periods, balanced and unbalanced windows, corrected and uncorrected treatment coding, with and without never-treated states. Estimates range from −0.32 to +0.78 and none is distinguishable from zero. Individual sector coefficients bear no relation to sector exposure: professional services, at exposure 0.224, comes out more negative than manufacturing at 1.000, and accommodation at 0.007 comes out positive.</p>

<p><b>Reading.</b> Entry responds to adoption in the short run. That response does not vary with sector R&amp;D exposure, which is what the intangible-substitution mechanism requires. Note that this says nothing about the mechanism behind the aggregate effect — establishing that would require ruling out the state-wide channels rather than merely observing that the sector channel is absent.</p>

<hr>

<h2>Things that could be wrong with this</h2>

<p><b>The balanced window was chosen after seeing the full-window result.</b> A single post dummy over the full window averages Massachusetts's 25th post-treatment year with Florida's 4th, so restricting to a common horizon is the conceptually cleaner estimand — but the sequence was: run the full window, notice the horizon problem, then restrict. That ordering is stated rather than hidden, and the full-window estimate is reported alongside.</p>

<p><b>Treatment is not absorbing, and coding it as absorbing was an error.</b> Four states in the sample repealed after adopting: Missouri (off 2005–2015, eleven of fifteen post years), Texas (2008–2013), Michigan (2012–2015), Washington (2015). That is 22 of 328 observable post state-years, 6.7%, coded as treated when no credit existed. Correcting it <i>lowers</i> the aggregate estimate from +0.225 to +0.149, the opposite of what attenuation bias predicts, which means entry ran relatively high during the repeal years. With only 22 such state-years this may be chance, but it doesn't support the positive-effect reading.</p>

<p><b>The outcome window can't be extended past 2015.</b> An earlier version ran outcomes to 2023 on the reasoning that an event study only needs the adoption year. That reasoning holds only for absorbing treatment. Since PDIT ends in 2015, treatment status for 2016–2023 is simply unobserved.</p>

<p><b>Never-treated states matter more than they should.</b> They supply no treatment variation in the gradient specification, but they do help estimate the sector-year fixed effects, which changes the residualised outcome for everyone else. Dropping them moves the gradient from +0.048 to +0.233. The aggregate specifications are less exposed, since there the never-treated states are genuine comparison units.</p>

<p><b>Identifying variation is concentrated.</b> For the gradient, effective cluster count is 11.7 against a nominal 33, the top five states hold 58% of residual treatment variance, and sectors 31-33 and 51 hold 84%. The aggregate is better behaved: 25.8 effective clusters, top five states 23.8%.</p>

<p><b>Pre-trends aren't rejected, but they aren't confirmed either.</b> Joint pre-period statistic 7.0 against a 5% critical value of 11.3. Individual leads run 0.10 to 0.43 — not small, just imprecise. Written up as a failure to reject.</p>

<p><b>PDIT's industry dimension is model-imputed.</b> States don't generally set different R&amp;D credit rates by industry; the cross-industry spread comes from applying industry characteristics inside Bartik's hypothetical-firm simulation. Defensible as an exposure measure, but not observed policy variation. The cross-industry ranking is at least stable across 1992–2015.</p>

<p><b>R&amp;D isn't the intangible capital in the story.</b> The substitution mechanism is mostly about software and organizational capital. Qualified research expenses are mostly wages, so an R&amp;D credit is partly a direct subsidy to research employment, which runs opposite to the substitution prediction in the short run.</p>

<p><b>TWFE under staggered adoption is biased</b> (Goodman-Bacon 2021). Using never-treated states as controls mitigates this but does not fix it. A Callaway–Sant'Anna estimator with continuous treatment has not been implemented.</p>

<p><b>Sector coverage.</b> PDIT has no agriculture, mining or utilities, so 16 of 19 BDS sectors are used. Value-added weights are national and state-invariant.</p>

<p><b>Why not TCJA §174.</b> It looks like the obvious shock — a dated, IPP-specific change in the cost of R&amp;D — but BDS ends in 2023, leaving two post years both inside the post-COVID reallocation surge. It was also reversed retroactively by OBBBA in July 2025 via new §174A, so it only ever bound for tax years 2022–2024, with repeal attempts live throughout. Firms expecting retroactive reversal have little reason to restructure real R&amp;D. And for a profitable firm the NPV cost of five-year amortization is roughly 2–3 cents per dollar of R&amp;D.</p>

<hr>

<h2>On comparison with Fazio, Guzman and Stern (2020)</h2>

<p>FGS use the same PDIT data in a state-level difference-in-differences and find that R&amp;D credits raise high-quality new firm formation by about 20% over ten years. An earlier version of this README scaled the confidence intervals here against that figure. <b>That comparison does not hold.</b> Their outcome is business registration data, including non-employer firms and quality-weighted; the outcome here is BDS employer establishment entry. Their 20% is a cumulative ten-year effect; the estimates here are average post-adoption effects. The two numbers are not on the same scale and are not compared.</p>

<p>The relationship between the two designs is that state-year fixed effects absorb, by construction, exactly the parameter FGS estimate. The gradient result therefore neither confirms nor contradicts them.</p>

<hr>

<h2>Scripts</h2>

<p>Run everything from the repository root, not from inside <code>scripts/</code> — paths are relative to the working directory.</p>

<pre>
00_verify_data.py             BEA row alignment and additivity checks
06_build_panel.py             national series (JRR, young share, software, IPP)
07_plot_scissor.py            Figure 1, the scissor gap
11_sector_intensity.py        sector IPP intensity, 513 rows
13_event_specs.py             event study, 7 specifications
15_panel_design.py            panel regressions, 4 specifications
17_placebo.py                 equipment and structures as placebos
18_decomposition.py           shift-share decomposition + Figure 2
21_decomp_sensitivity.py      decomposition across 6 period cuts
22_multi_outcome.py           7 outcomes x 3 specifications

30_variance_diagnostic.py     does the outcome survive the target FE? (26.8% does)
31_build_pdit_panel.py        merge PDIT onto BDS, power calculation
32_treatment_diagnostics.py   treatment separability, variation concentration
33_baseline_and_robustness.py TWFE, leave-one-state-out, lead test
34_outlier_check.py           influence and winsorising checks
36_event_study_wcb.py         event study with wild cluster bootstrap
37_ref_period.py              reference period sensitivity
38_aggregate_effect.py        aggregate effect alongside the sector gradient
39_reconcile.py               puts the two on a common scale, decomposes them
40_binary_gradient.py         binary split done correctly, placebo sectors
41_fix_treatment.py           treatment recoded as non-absorbing, balanced window
</pre>

<p>Scripts 33 and 40 overlap on purpose. Script 33 estimated the sector split as separate subsample regressions; with two sectors in the high group, state-year fixed effects leave only the manufacturing-minus-information difference and the estimator degenerates. That is where a spurious coefficient of 2807 came from. Script 40 redoes it as a dummy interacted in the full sample. Both are kept so the error and the fix are both visible.</p>

<pre>
py -m pip install -r requirements.txt
py scripts/00_verify_data.py
</pre>

<hr>

<h2>References</h2>

<p>Bartik, T. (2017). A New Panel Database on Business Incentives for Economic Development. Upjohn Institute.</p>

<p>Cameron, A.C., Gelbach, J. &amp; Miller, D. (2008). Bootstrap-based improvements for inference with clustered errors. <i>Review of Economics and Statistics</i> 90(3).</p>

<p>Decker, R., Haltiwanger, J., Jarmin, R. &amp; Miranda, J. (2016). Declining business dynamism: implications for productivity. <i>Brookings Institution</i>.</p>

<p>Fazio, C., Guzman, J. &amp; Stern, S. (2020). The impact of state-level R&amp;D tax credits on the quantity and quality of entrepreneurship. <i>Economic Development Quarterly</i> 34(2).</p>

<p>Goodman-Bacon, A. (2021). Difference-in-differences with variation in treatment timing. <i>Journal of Econometrics</i> 225(2).</p>

<p>Wilson, D. (2009). Beggar thy neighbor? The in-state, out-of-state, and aggregate effects of R&amp;D tax credits. <i>Review of Economics and Statistics</i> 91(2).</p>