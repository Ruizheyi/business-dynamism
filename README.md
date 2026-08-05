<h1>Declining Business Dynamism and Intangible Capital</h1>

<p>Replication and extension of Decker et al. (2016) using BDS and BEA data. Tests whether intangible capital substitution explains why U.S. business dynamism has been declining.</p>

<p>Short version: intangible investment and business dynamism diverge sharply and consistently, but the divergence is entirely within sectors, and a policy shock aimed straight at intangible capital doesn't move dynamism at all. The decline looks like something other than firms swapping software for workers.</p>

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

<p><b>Credit exposure:</b> present value of R&amp;D credits over a 20-year simulation as a % of industry value added, aggregated from PDIT's 45 industries to BDS sectors using value-added weights.</p>

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

<p>The shock is the staggered adoption of state R&amp;D tax credits. 20 states adopt inside the window (MA 1991 through FL 2012), 7 never adopt and serve as controls, 6 already had credits in 1990 and get dropped as left-censored. Event studies only need the adoption year, so outcomes run to 2023 even though PDIT stops in 2015. Treatment is credit generosity interacted with fixed sector exposure; manufacturing is normalized to 1, information is 0.725, professional services 0.224, everything else under 0.02.</p>

<p>Effect of adoption relative to the full pre-window, with wild cluster bootstrap inference:</p>

<table>
<tr><th>Outcome</th><th>β</th><th>p</th><th>95% CI (% of mean)</th></tr>
<tr><td>Establishment entry rate</td><td>−0.068</td><td>0.681</td><td><b>[−3.7%, +2.3%]</b></td></tr>
<tr><td>Establishment exit rate</td><td>−0.080</td><td>0.491</td><td>[−3.0%, +1.4%]</td></tr>
<tr><td>Job reallocation rate</td><td>−0.402</td><td>0.529</td><td>[−5.4%, +2.7%]</td></tr>
</table>

<p>Fazio, Guzman &amp; Stern (2020) estimate a <b>+20%</b> state-level effect of the same credits on high-quality new firm formation. These intervals rule out a sector-differential effect anywhere close to that.</p>

<p><b>What this does and doesn't say.</b> State×year fixed effects absorb exactly the parameter FGS estimate, so this isn't a contradiction of their result — it's a different estimand. What's identified here is whether the credit's effect <i>varies with sector R&amp;D intensity</i>, which is precisely what the substitution mechanism requires. It doesn't. If state R&amp;D credits do raise entrepreneurship, the channel looks state-wide (financing, talent, general business climate) rather than running through raised intangible intensity in R&amp;D-heavy sectors.</p>

<hr>

<h2>Things that could be wrong with this</h2>

<p><b>Identifying variation is concentrated.</b> Effective cluster count is 11.7 against a nominal 33; the top five states hold 58% of residual treatment variance and sectors 31-33 and 51 hold 84%. All inference uses wild cluster bootstrap-t, Rademacher weights, null imposed. Leave-one-state-out gives zero sign flips in 33 drops.</p>

<p><b>Pre-trends aren't rejected, but they aren't confirmed either.</b> Joint pre-period statistic is 7.0 against a 5% critical value of 11.3. Individual leads run 0.10 to 0.43 — not small, just imprecise. That's a failure to reject, and it's written up as one.</p>

<p><b>The reference period does real work, so the sensitivity table is in the paper.</b> With k=−1 omitted, the 0≤k&lt;5 coefficient is +0.412 (p=0.057), which looks marginal. But the pre-period coefficient in the same specification is +0.366, nearly identical — k=−1 is just a low point. Omit the whole pre-window instead and it drops to +0.081 (p=0.586) with the post coefficient flipping sign. Headline numbers use the pre-window-average version, which doesn't depend on any single omitted year.</p>

<p><b>PDIT's industry dimension is model-imputed.</b> States don't generally set different R&amp;D credit rates by industry; the cross-industry spread comes from applying industry characteristics inside Bartik's hypothetical-firm simulation. It's a defensible exposure measure but it isn't observed policy variation.</p>

<p><b>R&amp;D isn't the intangible capital in the story.</b> The substitution mechanism is mostly about software and organizational capital. Qualified research expenses are mostly wages, so an R&amp;D credit is partly a direct subsidy to research employment, which runs opposite to the substitution prediction in the short run.</p>

<p><b>Sector coverage.</b> PDIT has no agriculture, mining or utilities, so 16 of 19 BDS sectors are used. Value-added weights are national and state-invariant.</p>

<p><b>Why not TCJA §174.</b> It looks like the obvious shock — a dated, IPP-specific change in the cost of R&amp;D — but BDS ends in 2023, leaving two post years both inside the post-COVID reallocation surge. It was also reversed retroactively by OBBBA in July 2025 via new §174A, so it only ever bound for tax years 2022–2024, with repeal attempts live throughout. Firms expecting retroactive reversal have little reason to restructure real R&amp;D. And for a profitable firm the NPV cost of five-year amortization is roughly 2–3 cents per dollar of R&amp;D.</p>

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
</pre>

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
