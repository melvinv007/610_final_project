"""Figures for the report (reads results/tables, writes results/figures/*.pdf and *.png)."""
import json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import TAB, FIG

FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7, "font.family": "DejaVu Sans", "figure.dpi": 150})
C = {"blue": "#2b6cb0", "orange": "#d9730d", "grey": "#6b7280", "red": "#b83232", "green": "#2f855a", "light": "#cbd5e1"}
rd = lambda n: pd.read_csv(TAB / n)
def tag(ax, t): ax.text(-0.02, 1.07, t, transform=ax.transAxes, fontsize=9.5, fontweight="bold", va="bottom", ha="right")
def save(fig, name): fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight", pad_inches=0.12); fig.savefig(FIG / f"{name}.png", bbox_inches="tight", pad_inches=0.12, dpi=200); plt.close(fig)


def fig_q1():
    cat, dose = rd("q1_by_category.csv"), rd("q1_dose_fine.csv"); eff = rd("q1_effects.csv"); bal = rd("q1_balance.csv")
    skip = cat.set_index("category").loc["skip"]
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.7), gridspec_kw={"width_ratios": [1.5, 1, 1], "wspace": 1.2})
    a = ax[0]; x = np.arange(len(dose))
    a.axhspan(skip.lo, skip.hi, color=C["red"], alpha=.15, lw=0); a.axhline(skip.success, color=C["red"], lw=1, ls="--")
    a.text(len(dose) - 0.6, skip.success - 0.012, "skipped lesson", color=C["red"], ha="right", va="top", fontsize=7.5)
    a.errorbar(x, dose.success, yerr=[dose.success - dose.lo, dose.hi - dose.success], fmt="o-", color=C["blue"], ms=3.5, lw=1.2, capsize=1.5)
    a.set_xticks(x); a.set_xticklabels(dose.bin, rotation=55, ha="right", fontsize=6.5); a.set_xlabel("Seconds spent on the lesson")
    a.set_ylabel("Correct on first check-out try"); a.set_ylim(0.25, 0.58); tag(a, "a"); a.set_title("Dose-response", loc="left", pad=4)
    b = ax[1]; names = {"Time to fail the retry (log s)": "Time to fail retry", "Earlier failures in session": "Earlier failures", "Question position": "Question position",
                        "Student prior check-in accuracy": "Student's past accuracy", "Accuracy earlier in this session": "Accuracy so far in session",
                        "Concept difficulty (mean accuracy)": "Concept difficulty", "Year group": "Year group", "After calendar break": "After 3 May"}
    bb = bal[bal.covariate.isin(names)].copy(); bb["lab"] = bb.covariate.map(names); bb = bb.iloc[::-1]; y = np.arange(len(bb))
    b.axvspan(-0.1, 0.1, color=C["light"], alpha=.5, lw=0); b.axvline(0, color=C["grey"], lw=.8)
    b.scatter(bb.smd, y, color=[C["red"] if abs(v) > .25 else C["blue"] for v in bb.smd], s=18, zorder=3)
    b.set_yticks(y); b.set_yticklabels(bb.lab); b.set_xlabel("Standardised difference"); b.set_xlim(-1.3, 0.9); tag(b, "b"); b.set_title("Who skips?", loc="left", pad=4)
    c = ax[2]; labs = ["Raw", "Same students,\nwith vs. without", "Fixed effects\n+ controls", "Adjusted logit", "Weighting (IPW)"]; yy = np.arange(len(eff))[::-1]
    c.errorbar(eff.est * 100, yy, xerr=[(eff.est - eff.lo) * 100, (eff.hi - eff.est) * 100], fmt="s", color=C["blue"], ms=4, capsize=2, lw=1.2)
    c.axvline(0, color=C["grey"], lw=.8); c.set_yticks(yy); c.set_yticklabels(labs); c.set_xlabel("Effect (pp)"); c.set_xlim(-1, 17); tag(c, "c")
    c.set_title("Five estimates", loc="left", pad=4)
    save(fig, "fig3_lesson")


def fig_overview():
    reach = rd("q2_reach.csv"); mo = rd("data_monthly.csv"); cp = json.load(open(TAB / "cp_summary.json"))
    fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.2), gridspec_kw={"wspace": 0.45})
    a = ax[0]; a.bar(reach.question, reach.share_sessions_reaching * 100, color=C["blue"], width=.6)
    for q, v in zip(reach.question, reach.share_sessions_reaching * 100): a.text(q, v + 1.5, f"{v:.1f}%", ha="center", fontsize=7.5)
    a.set_ylim(0, 112); a.set_xlabel("Question number in the quiz"); a.set_ylabel("Sessions that reach it (%)"); tag(a, "a"); a.set_title("Sessions thin out slowly", loc="left", pad=4)
    b = ax[1]; x = np.arange(len(mo)); b.bar(x - .2, mo.sessions / 1000, .4, color=C["blue"], label="sessions (thousands)"); b.bar(x + .2, mo.students / 1000, .4, color=C["orange"], label="students (thousands)")
    bx = (pd.Timestamp(cp["break_date"]) - pd.Timestamp("2022-02-01")).days / 30.44 - 0.5; b.axvline(bx, color=C["grey"], ls="--", lw=1)
    b.set_xticks(x); b.set_xticklabels([m[5:] for m in mo.month]); b.set_xlabel("Month of 2022"); b.legend(frameon=False, loc="upper right", fontsize=6.8, ncol=1); b.set_ylim(0, 16); tag(b, "b")
    b.set_title("Activity by month", loc="left", pad=4)
    save(fig, "fig2_overview")


def fig_q2():
    st = rd("q2_state_rates.csv"); st = st[st.state != "Lesson"].sort_values("event_rate")
    mg = rd("q2_fr_marginal.csv"); fe = rd("q2_fr_fe.csv"); mo = rd("cp_monthly.csv"); cp = json.load(open(TAB / "cp_summary.json"))
    fig, ax = plt.subplots(2, 2, figsize=(6.6, 4.5), gridspec_kw={"hspace": 0.8, "wspace": 0.7})
    a = ax[0, 0]; y = np.arange(len(st)); colr = [C["red"] if s == "CheckinRetry_fail" else C["light"] for s in st.state]
    a.barh(y, st.event_rate * 100, color=colr, xerr=[(st.event_rate - st.event_lo) * 100, (st.event_hi - st.event_rate) * 100], error_kw={"lw": .8, "capsize": 1.5})
    a.set_yticks(y); a.set_yticklabels(st.label); a.set_xlabel("Chance the session ends here (%)"); tag(a, "a"); a.set_title("Quitting sits at one step", loc="left", pad=4)
    b = ax[0, 1]; pos = mg[mg.variable == "position"]; fl = mg[mg.variable == "failures so far"]
    for d, c, l in [(pos, C["blue"], "question position (1-4)"), (fl, C["orange"], "earlier failed retries (0-3)")]:
        xx = d.level.astype(int).values; xx = xx if d is pos else xx + 1
        b.errorbar(xx, d.adj_prob * 100, yerr=[(d.adj_prob - d.lo) * 100, (d.hi - d.adj_prob) * 100], marker="o", ms=3.5, color=c, capsize=2, lw=1.2, label=l)
    b.set_xticks([1, 2, 3, 4]); b.set_xticklabels(["1\n0", "2\n1", "3\n2", "4\n3"]); b.set_xlabel("Question position (top row)\nearlier failed retries (bottom row)")
    b.set_ylabel("Adjusted quit chance after\na failed retry (%)"); b.legend(frameon=False, loc="upper right"); b.set_ylim(0, 22); tag(b, "b")
    b.set_title("Pooled: risk falls over the session", loc="left", pad=4)
    c = ax[1, 0]; terms = [("pfr_1", "1 earlier failed retry"), ("pfr_2", "2 earlier failed retries"), ("q_2", "Position 2 (vs 1)"), ("q_3", "Position 3"), ("q_4", "Position 4"), ("lat_<=3 s", "Answered in <=3 s")]
    f = fe.set_index("term").loc[[t for t, _ in terms]]; yy = np.arange(len(terms))[::-1]
    c.errorbar(f.OR_pooled, yy + .15, xerr=[f.OR_pooled - f.lo_pooled, f.hi_pooled - f.OR_pooled], fmt="o", color=C["grey"], ms=3.5, capsize=1.5, lw=1.1, label="Pooled")
    c.errorbar(f.OR_within_student, yy - .15, xerr=[f.OR_within_student - f.lo, f.hi - f.OR_within_student], fmt="s", color=C["orange"], ms=3.5, capsize=1.5, lw=1.1, label="Within student")
    c.axvline(1, color="k", lw=.7); c.set_xscale("log"); c.set_xticks([.3, .5, 1, 2, 4]); c.set_xticklabels(["0.3", "0.5", "1", "2", "4"]); c.set_yticks(yy); c.set_yticklabels([l for _, l in terms])
    c.set_xlabel("Odds ratio of quitting (95% CI)"); c.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2); tag(c, "c"); c.set_title("Failures raise risk within a student", loc="left", pad=4)
    d = ax[1, 1]; x = np.arange(len(mo)); d.plot(x, mo.rate * 100, "o-", color=C["red"], ms=3.5, lw=1.3)
    bx = (pd.Timestamp(cp["break_date"]) - pd.Timestamp("2022-02-01")).days / 30.44 - 0.5
    d.axvline(bx, color=C["grey"], ls="--", lw=1); d.text(bx + .1, 6.5, "3 May", color=C["grey"], fontsize=7.5, va="bottom")
    d.set_xticks(x); d.set_xticklabels([m[5:] for m in mo.month]); d.set_xlabel("Month of 2022"); d.set_ylabel("Quit chance after a failed\nretry (%)"); d.set_ylim(5, 24); tag(d, "d")
    d.set_title("A step change in early May", loc="left", pad=4)
    save(fig, "fig5_quitting")


def fig_q3():
    cur, ps, ts = rd("q3_return_curves.csv"), rd("q3_persistence.csv"), rd("q3_trait_state.csv")
    fig, ax = plt.subplots(1, 3, figsize=(6.6, 2.7), gridspec_kw={"wspace": 0.95, "width_ratios": [1.2, 1, 0.9]})
    a = ax[0]
    for k, c, l in [("completed", C["green"], "completed"), ("exit after failed retry", C["red"], "quit after failed retry"), ("other exit", C["blue"], "other quit")]:
        d = cur[(cur.kind == k) & (cur.hours > 0)]; a.plot(d.hours, d.p_return * 100, color=c, lw=1.4, label=l); a.fill_between(d.hours, d.lo * 100, d.hi * 100, color=c, alpha=.18, lw=0)
    [a.text(0.22, y0, l, color=c, fontsize=7.5, va="bottom") for y0, l, c in [(49, "completed", C["green"]), (39, "quit after failed retry", C["red"]), (25, "other quit", C["blue"])]]
    a.set_xscale("log"); a.set_xticks([1, 6, 24, 168, 720]); a.set_xticklabels(["1h", "6h", "1d", "1w", "30d"]); a.set_xlim(.2, 720)
    a.set_xlabel("Time since the session ended"); a.set_ylabel("Student has started another\nsession (%)"); tag(a, "a"); a.set_title("Return", loc="left", pad=4)
    b = ax[1]; g = ps[ps.gap.isin(["all", "<1 h (same sitting)", "1-24 h", "1-7 d", ">7 d"])]; lab = ["All", "<1 h", "1-24 h", "1-7 d", ">7 d"]; x = np.arange(len(g)); w = .38
    b.bar(x - w / 2, g.rate_after_complete * 100, w, color=C["green"], label="after a completed session"); b.bar(x + w / 2, g.rate_after_exit * 100, w, color=C["red"], label="after a quit")
    b.set_xticks(x); b.set_xticklabels(lab, rotation=40, ha="right"); b.set_xlabel("Gap since previous session"); b.set_ylabel("Chance this session\nends in a quit (%)"); b.legend(frameon=False, loc="upper left", fontsize=6.8); b.set_ylim(0, 60); tag(b, "b")
    b.set_title("Persistence", loc="left", pad=4)
    c = ax[2]; labs = ["Raw", "Control for\nusual rate", "Within\nstudent"]; y = np.arange(3)[::-1]
    c.errorbar(ts.effect_of_previous_exit * 100, y, xerr=[(ts.effect_of_previous_exit - ts.lo) * 100, (ts.hi - ts.effect_of_previous_exit) * 100], fmt="s", color=C["blue"], ms=4, capsize=2, lw=1.2)
    c.axvline(0, color=C["grey"], lw=.8); c.set_yticks(y); c.set_yticklabels(labs); c.set_xlim(-1, 32); c.set_xlabel("Extra quit chance\n(pp, 95% CI)"); tag(c, "c"); c.set_title("Trait vs. state", loc="left", pad=4)
    save(fig, "fig6_persistence_return")


def fig_where():
    w = rd("q2_where.csv"); fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.5), gridspec_kw={"wspace": 0.5})
    def band(a, d, x, col, lab):
        a.errorbar(x, d.rate * 100, yerr=[(d.rate - d.lo) * 100, (d.hi - d.rate) * 100], marker="o", ms=3.5, color=col, capsize=1.5, lw=1.2, label=lab)
    for a, var, xl, t, tg in [(ax[0], "position", "Question number", "Position", "a"), (ax[1], "time in session", "Minutes since session start", "Time in session", "b")]:
        for stp, col in [("after a failed retry", C["red"]), ("any other step", C["grey"])]:
            d = w[(w.variable == var) & (w.step == stp)].reset_index(drop=True); band(a, d, np.arange(len(d)), col, stp)
        a.set_xticks(np.arange(len(d))); a.set_xticklabels(d.level, rotation=0 if var == "position" else 40, ha="center" if var == "position" else "right"); a.set_xlabel(xl); tag(a, tg); a.set_title(t, loc="left", pad=4)
    ax[0].set_ylabel("Chance the session ends (%)"); ax[0].legend(frameon=False, loc="center right", fontsize=6.8); ax[1].set_yscale("log"); ax[1].set_yticks([0.5, 1, 2, 5, 10, 20]); ax[1].set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax[0].set_yscale("log"); ax[0].set_yticks([0.5, 1, 2, 5, 10, 20]); ax[0].set_yticklabels(["0.5", "1", "2", "5", "10", "20"])
    c = ax[2]
    for var, col, lab in [("hour of day (before break)", C["blue"], "before 3 May"), ("hour of day", C["orange"], "after 3 May")]:
        d = w[w.variable == var].reset_index(drop=True); band(c, d, np.arange(len(d)), col, lab)
    c.set_xticks(np.arange(9)); c.set_xticklabels(d.level, rotation=50, ha="right"); c.set_xlabel("Hour of the day"); c.legend(frameon=False, fontsize=6.8, loc="upper right"); tag(c, "c")
    c.set_title("Time of day (failed retry)", loc="left", pad=4); c.set_ylim(0, 27)
    save(fig, "fig4_where")


def fig_conc():
    st = rd("q3_student_rates.csv"); lz = rd("q3_lorenz.csv"); sk = rd("q3_streak.csv")
    fig, ax = plt.subplots(1, 3, figsize=(6.8, 2.5), gridspec_kw={"wspace": 0.75})
    a = ax[0]; a.hist(st.rate * 100, bins=np.arange(0, 105, 5), color=C["blue"], edgecolor="white", lw=.5)
    a.set_yscale("log"); a.set_xlabel("Student's quit rate (%)"); a.set_ylabel("Number of students (log)"); tag(a, "a"); a.set_title("Most students rarely quit", loc="left", pad=4)
    b = ax[1]; b.plot(lz.share_students * 100, lz.share_exits * 100, color=C["red"], lw=1.6); b.plot([0, 100], [0, 100], color=C["grey"], ls="--", lw=.9)
    b.set_xlabel("Students ranked by quits (%)"); b.set_ylabel("Cumulative share of all quits (%)"); tag(b, "b"); b.set_title("Quits are concentrated", loc="left", pad=4)
    c = ax[2]; c.bar(sk.quits_in_previous_3, sk.rate * 100, yerr=[(sk.rate - sk.lo) * 100, (sk.hi - sk.rate) * 100], color=[C["light"], "#e5a98c", "#d9730d", C["red"]], capsize=2, width=.65)
    c.set_xticks(sk.quits_in_previous_3); c.set_xlabel("Quits in last 3 sessions"); c.set_ylabel("Chance the next session\nends in a quit (%)"); tag(c, "c"); c.set_title("A run of quits", loc="left", pad=4)
    save(fig, "fig7_concentration")


if __name__ == "__main__":
    fig_q1(); fig_overview(); fig_q2(); fig_where(); fig_q3(); fig_conc(); print("figures written to", FIG)
