"""Step 2 - RQ1: is a student far more likely to leave right after failing a retry?

Unit of analysis : one in-scope event (question 1-4). Outcome: exit = 1 if it was the
                   last event of its quiz session.
Two summaries    : event-weighted  = exits / events (every event counts equally;
                                     heavy users dominate)
                   student-weighted = mean over students of (their own exit rate for
                                     that event type); each student counts once
Uncertainty      : cluster bootstrap - resample STUDENTS with replacement (N_BOOT times)
                   so that repeated events by the same student are not treated as
                   independent. 95% CI = 2.5th-97.5th percentile.
Headline contrast: exit rate after a failed check-in retry vs after every other event
                   (difference and ratio, both with bootstrap CIs).

Writes: results/rq1_exit_by_state.csv, results/rq1_contrast.csv, results/rq1_summary.txt
        figures/rq1_exit_by_event.pdf and .png
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import EVENTS, RESULTS, FIGS, SEED, N_BOOT

TARGET = "CheckinRetry_fail"
LABELS = {
    "Checkin_ok": "Check-in, correct",
    "Checkin_fail": "Check-in, wrong",
    "CheckinRetry_ok": "Check-in retry, correct",
    "CheckinRetry_fail": "Check-in retry, wrong",
    "Lesson": "Lesson",
    "Checkout_ok": "Check-out, correct",
    "Checkout_fail": "Check-out, wrong",
    "CheckoutRetry_ok": "Check-out retry, correct",
    "CheckoutRetry_fail": "Check-out retry, wrong",
}


def pct(a, q):
    return np.percentile(a, q, axis=0)


def main():
    ev = pd.read_pickle(EVENTS)
    s = ev[ev.in_scope]
    states = list(LABELS)
    users = np.sort(s.UserId.unique())
    U = len(users)

    def mat(agg):
        return (s.pivot_table(index="UserId", columns="state", values="exit", aggfunc=agg, fill_value=0)
                  .reindex(index=users, columns=states, fill_value=0).to_numpy(float))

    N, X = mat("size"), mat("sum")                       # students x states
    R = np.divide(X, N, out=np.zeros_like(X), where=N > 0)   # each student's own rate
    M = (N > 0).astype(float)                            # did the student experience this state?
    kt = states.index(TARGET)

    def summarise(w):
        Nb, Xb = w @ N, w @ X
        ev_rate = Xb / Nb
        sw_rate = (w @ R) / (w @ M)
        rt, ro = Xb[kt] / Nb[kt], (Xb.sum() - Xb[kt]) / (Nb.sum() - Nb[kt])
        return ev_rate, sw_rate, np.array([rt, ro, rt - ro, rt / ro])

    point_ev, point_sw, point_c = summarise(np.ones(U))

    rng = np.random.default_rng(SEED)
    B_ev, B_sw, B_c = [], [], []
    for _ in range(N_BOOT):
        w = np.bincount(rng.integers(0, U, U), minlength=U).astype(float)
        a, b, c = summarise(w)
        B_ev.append(a); B_sw.append(b); B_c.append(c)
    B_ev, B_sw, B_c = map(np.array, (B_ev, B_sw, B_c))

    tab = pd.DataFrame({
        "state": states,
        "label": [LABELS[k] for k in states],
        "events": N.sum(0).astype(int),
        "students": M.sum(0).astype(int),
        "exits": X.sum(0).astype(int),
        "event_rate": point_ev, "event_lo": pct(B_ev, 2.5), "event_hi": pct(B_ev, 97.5),
        "student_rate": point_sw, "student_lo": pct(B_sw, 2.5), "student_hi": pct(B_sw, 97.5),
    }).sort_values("event_rate", ascending=False)
    tab.to_csv(RESULTS / "rq1_exit_by_state.csv", index=False)

    names = ["rate_failed_checkin_retry", "rate_all_other_events", "difference", "ratio"]
    con = pd.DataFrame({"quantity": names, "estimate": point_c,
                        "ci_lo": pct(B_c, 2.5), "ci_hi": pct(B_c, 97.5)})
    con.to_csv(RESULTS / "rq1_contrast.csv", index=False)

    # second-highest state as a stricter comparison
    second = tab.iloc[1]
    k2 = states.index(second.state)
    d2 = B_ev[:, kt] - B_ev[:, k2]

    lines = [
        f"students in scope: {U:,}; in-scope events: {int(N.sum()):,}; exits: {int(X.sum()):,}",
        f"exit rate after a failed check-in retry : {point_ev[kt]:.1%} "
        f"[{pct(B_ev, 2.5)[kt]:.1%}, {pct(B_ev, 97.5)[kt]:.1%}] (event-weighted)",
        f"                                          {point_sw[kt]:.1%} "
        f"[{pct(B_sw, 2.5)[kt]:.1%}, {pct(B_sw, 97.5)[kt]:.1%}] (student-weighted)",
        f"exit rate after every other event       : {point_c[1]:.2%}",
        f"difference                              : {point_c[2]*100:.1f} pp "
        f"[{pct(B_c, 2.5)[2]*100:.1f}, {pct(B_c, 97.5)[2]*100:.1f}]",
        f"ratio                                   : {point_c[3]:.1f}x "
        f"[{pct(B_c, 2.5)[3]:.1f}, {pct(B_c, 97.5)[3]:.1f}]",
        f"vs next-highest state ({second.label}): difference "
        f"{(point_ev[kt]-second.event_rate)*100:.1f} pp "
        f"[{np.percentile(d2, 2.5)*100:.1f}, {np.percentile(d2, 97.5)*100:.1f}]",
        "",
        tab[["label", "events", "students", "exits", "event_rate", "student_rate"]]
        .to_string(index=False, float_format=lambda v: f"{v:.4f}"),
    ]
    txt = "\n".join(lines)
    (RESULTS / "rq1_summary.txt").write_text(txt, encoding="utf-8")
    print(txt)
    plot(tab)


def plot(tab):
    INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e6e5e1"
    BASE, FOCUS = "#b9b8b2", "#2a78d6"
    t = tab.iloc[::-1].reset_index(drop=True)      # largest bar at the top
    y = np.arange(len(t))
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    cols = [FOCUS if k == TARGET else BASE for k in t.state]
    ax.barh(y, t.event_rate * 100, height=0.55, color=cols, zorder=2)
    ax.errorbar(t.event_rate * 100, y,
                xerr=[(t.event_rate - t.event_lo) * 100, (t.event_hi - t.event_rate) * 100],
                fmt="none", ecolor=INK, elinewidth=1, capsize=2, zorder=3)
    ax.errorbar(t.student_rate * 100, y,
                xerr=[(t.student_rate - t.student_lo) * 100, (t.student_hi - t.student_rate) * 100],
                fmt="D", color=INK, mfc="white", mec=INK, ms=4.5, elinewidth=1, capsize=2, zorder=4)
    for yi, r in zip(y, t.event_rate):
        ax.text(max(r * 100, 0) + 0.15, yi + 0.36, f"{r:.1%}", fontsize=8, color=MUTED, va="center")
    ax.set_yticks(y)
    ax.set_yticklabels(t.label)
    ax.set_xlabel("Exit rate after the event (%)")
    ax.xaxis.grid(True, color=GRID, lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    ax.legend(handles=[Patch(color=BASE, label="Event-weighted (bar, 95% CI)"),
                       Line2D([], [], marker="D", ls="", mfc="white", mec=INK, color=INK,
                              label="Student-weighted (diamond, 95% CI)")],
              loc="lower right", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS / "rq1_exit_by_event.pdf")
    fig.savefig(FIGS / "rq1_exit_by_event.png", dpi=200)
    print(f"Saved figure to {FIGS}")


if __name__ == "__main__":
    main()
