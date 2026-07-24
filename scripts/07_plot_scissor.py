import pandas as pd
import matplotlib.pyplot as plt

# national panel
panel = pd.read_csv("data/processed/national_panel.csv")

fig, ax = plt.subplots(figsize=(9.5, 5.5))

# mark recessions (vertical lines)
recessions = [(2001, "Dot-com"), (2008, "GFC"), (2020, "COVID")]
for year, label in recessions:
    ax.axvline(year, color="#95a5a6", lw=0.9, ls=":", alpha=0.8, zorder=1)
    ax.text(year, 57.5, label, ha="center", va="bottom", fontsize=8.5, color="#7f8c8d")

# shaded area between software and JRR
ax.fill_between(panel["year"], panel["jrr_index"], panel["software_share_index"],
                color="#c0392b", alpha=0.06, zorder=2)

# four main series
ax.plot(panel["year"], panel["software_share_index"],
        color="#c0392b", lw=2.4, zorder=5, label="Software share")
ax.plot(panel["year"], panel["ipp_share_index"],
        color="#e08e79", lw=1.8, ls="-.", zorder=4, label="IPP share")
ax.plot(panel["year"], panel["jrr_index"],
        color="#2c3e50", lw=2.2, zorder=5, label="Job reallocation rate")
ax.plot(panel["year"], panel["young_share_index"],
        color="#95a5a6", lw=1.6, ls="--", zorder=3, label="Young-firm employment share")

# reference line 1997 baseline
ax.axhline(100, color="black", lw=0.8, alpha=0.4, zorder=1)

# annotate the gap in 2023
last = panel.iloc[-1]
ax.annotate("", xy=(last["year"], last["software_share_index"]),
            xytext=(last["year"], last["jrr_index"]),
            arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.2))
ax.text(last["year"] - 0.4, (last["software_share_index"] + last["jrr_index"]) / 2,
        f"gap = {last['scissor_gap']:.0f}", ha="right", va="center",
        fontsize=9.5, color="#c0392b", fontweight="bold")

ax.set_xlim(1996, 2025.5)
ax.set_ylim(55, 228)
ax.set_xlabel("Year")
ax.set_ylabel("Index (1997 = 100)")
ax.set_title("Diverging paths: intangible investment vs. business dynamism, 1997-2023",
             fontsize=12.5, pad=14)
ax.legend(frameon=False, loc="upper left", fontsize=9, bbox_to_anchor=(0.01, 0.99))
ax.spines[["top", "right"]].set_visible(False)
ax.grid(axis="y", alpha=0.2)

fig.tight_layout()
fig.savefig("output/scissor_gap.png", dpi=200)
print("saved: output/scissor_gap.png")