"""Step 5 - draw all report figures from the saved result tables (no heavy computation)."""
import json, textwrap
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import TAB, FIG

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.titlesize": 10.5,
                     "pdf.fonttype": 42, "font.family": "DejaVu Sans"})
BLUE, ORANGE, GREY = "#1f4e79", "#c55a11", "#9aa7b4"

# ------------------------------------------------------------ Figure 1: exit rate by event type
t1 = pd.read_csv(TAB / "rq1_state_rates.csv"); o = t1[t1.state != "Lesson"].sort_values("exit_rate_event_weighted")
fig, ax = plt.subplots(figsize=(7.0, 3.3)); y = np.arange(len(o))
ax.errorbar(o.exit_rate_event_weighted * 100, y + .13, xerr=[(o.exit_rate_event_weighted - o.ew_ci_lo) * 100, (o.ew_ci_hi - o.exit_rate_event_weighted) * 100],
            fmt="o", c=BLUE, capsize=2, ms=5, label="Event-weighted")
ax.errorbar(o.exit_rate_student_weighted * 100, y - .13, xerr=[(o.exit_rate_student_weighted - o.sw_ci_lo) * 100, (o.sw_ci_hi - o.exit_rate_student_weighted) * 100],
            fmt="s", c=ORANGE, capsize=2, ms=5, label="Student-weighted")
ax.set_yticks(y); ax.set_yticklabels(o.label); ax.set_xlabel("Probability the quiz session ends after the event (%)")
ax.legend(frameon=False, loc="lower right"); ax.grid(axis="x", alpha=.3)
fig.tight_layout(); fig.savefig(FIG / "fig1_exit_by_event.pdf"); fig.savefig(FIG / "fig1_exit_by_event.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------ Figure 2: hotspots
ps = pd.read_csv(TAB / "rq2_primary_split.csv", index_col=0); full = pd.read_csv(TAB / "rq2_topic_full.csv", index_col=0)
rep = json.load(open(TAB / "rq2_summary.json"))["topic_level"]
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.7), gridspec_kw={"width_ratios": [1, 1.25]})
la, lb = np.log2(ps.oeA.clip(lower=.25)), np.log2(ps.oeB.clip(lower=.25)); top = ps.top10_A.values
ax[0].scatter(la[~top], lb[~top], s=13, c=GREY, alpha=.75, label="Other topics")
ax[0].scatter(la[top], lb[top], s=26, c=ORANGE, label="10 strongest in sample A")
ax[0].axhline(0, c="k", lw=.5); ax[0].axvline(0, c="k", lw=.5)
ax[0].set_xlabel("log$_2$(observed/expected), sample A"); ax[0].set_ylabel("log$_2$(observed/expected), sample B")
ax[0].set_title("(a) Discovery vs validation sample"); ax[0].set_ylim(-2.2, 1.9); ax[0].legend(frameon=False, fontsize=8.5, loc="upper left", handletextpad=.2)
t12 = full.head(12).iloc[::-1]; yy = np.arange(len(t12)); sig = (t12.q < .05).values
lab = [textwrap.shorten(t, 30, placeholder="…") for t in t12.topic]
for i, r in enumerate(t12.itertuples()):
    s = r.q < .05
    ax[1].errorbar(r.oe, i, xerr=[[r.oe - r.oe_lo], [r.oe_hi - r.oe]], fmt="o", c=BLUE, mfc=BLUE if s else "white", capsize=2, ms=5)
ax[1].axvline(1, c="k", lw=.5); ax[1].set_yticks(yy); ax[1].set_yticklabels(lab, fontsize=8.5)
ax[1].set_xlabel("Observed / expected exits (95% CI)"); ax[1].set_title("(b) Top 12 topics, full sample")
ax[1].plot([], [], "o", c=BLUE, label="FDR q < .05"); ax[1].plot([], [], "o", c=BLUE, mfc="white", label="not significant")
ax[1].legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(.45, -.2), ncol=2)
fig.tight_layout(); fig.savefig(FIG / "fig2_hotspots.pdf"); fig.savefig(FIG / "fig2_hotspots.png", dpi=200); plt.close(fig)

# ------------------------------------------------------------ Figure 3: year groups
yt = pd.read_csv(TAB / "rq3_by_year.csv", index_col=0); tb = pd.read_csv(TAB / "rq3_topics_young_vs_old.csv", index_col=0)
r3 = json.load(open(TAB / "rq3_summary.json")); x = yt.index.values
fig, ax = plt.subplots(1, 2, figsize=(7.0, 3.5), gridspec_kw={"width_ratios": [1.1, 1]})
ax[0].errorbar(x - .08, yt.ew * 100, yerr=[(yt.ew - yt.ew_lo) * 100, (yt.ew_hi - yt.ew) * 100], fmt="o-", c=BLUE, capsize=2, ms=4, label="Event-weighted")
ax[0].errorbar(x + .08, yt.sw * 100, yerr=[(yt.sw - yt.sw_lo) * 100, (yt.sw_hi - yt.sw) * 100], fmt="s--", c=ORANGE, capsize=2, ms=4, label="Student-weighted")
ax[0].plot(x, yt.other_ew * 100, ":", c="#555555", label="After other answers")
ax[0].set_xlabel("Year group"); ax[0].set_ylabel("Exit after failed retry (%)"); ax[0].set_ylim(0, 38); ax[0].set_xticks(x)
ax[0].legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(.5, -.22), ncol=2, columnspacing=1); ax[0].set_title("(a) Exit rate by year group")
lx, ly = np.log2(tb.oe_young.clip(lower=.25)), np.log2(tb.oe_old.clip(lower=.25)); rho = r3["hotspots_young_vs_old"]["rho"]
ax[1].scatter(lx, ly, s=14, c=BLUE, alpha=.7); ax[1].axhline(0, c="k", lw=.5); ax[1].axvline(0, c="k", lw=.5)
ax[1].set_xlabel("log$_2$(O/E), Years 5–7"); ax[1].set_ylabel("log$_2$(O/E), Years 9–11")
ax[1].set_title(f"(b) Topic O/E by age band ($\\rho$ = {rho:.2f})")
fig.tight_layout(); fig.savefig(FIG / "fig3_year_groups.pdf"); fig.savefig(FIG / "fig3_year_groups.png", dpi=200); plt.close(fig)
print("figures written")
