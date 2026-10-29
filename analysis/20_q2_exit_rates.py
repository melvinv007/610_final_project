"""RQ2 (part 1) - Where in a session does quitting concentrate?  Exit rates, process structure, robustness, who-vs-what.

Unit    : one in-scope event (questions 1-4). Outcome: exit = the event is the last of a session that never reached Q5.
Weights : event-weighted (every event counts) and student-weighted (every student counts once), because a few heavy
          users produce most events.
Uncertainty: cluster bootstrap over STUDENTS (95% percentile intervals).
Writes  : results/tables/q2_state_rates.csv, q2_contrast.json, q2_transitions.csv, q2_sensitivity.csv, q2_commitment.csv, q2_icc.csv
"""
import json, warnings
import numpy as np
import pandas as pd
from common import TAB, SEED, load_events, cluster_bootstrap, pct_ci

warnings.filterwarnings("ignore")
N_BOOT = 2000
TARGET = "CheckinRetry_fail"
LABELS = {"Checkin_ok": "Check-in, correct", "Checkin_fail": "Check-in, wrong", "CheckinRetry_ok": "Check-in retry, correct",
          "CheckinRetry_fail": "Check-in retry, wrong", "Lesson": "Lesson", "Checkout_ok": "Check-out, correct",
          "Checkout_fail": "Check-out, wrong", "CheckoutRetry_ok": "Check-out retry, correct", "CheckoutRetry_fail": "Check-out retry, wrong"}
STATES = list(LABELS)


def student_matrices(d, col="is_exit"):
    users = np.sort(d.UserId.unique())
    mk = lambda agg: (d.pivot_table(index="UserId", columns="state", values=col, aggfunc=agg, fill_value=0)
                       .reindex(index=users, columns=STATES, fill_value=0).to_numpy(float))
    return mk("size"), mk("sum")


def contrast_stats(N, X, idx):
    kt, ki = STATES.index(TARGET), STATES.index("Lesson")
    other = [i for i in range(len(STATES)) if i not in (kt, ki)]
    nT, kT = N[idx][..., kt].sum(1), X[idx][..., kt].sum(1)
    nO, kO = N[idx][..., other].sum((1, 2)), X[idx][..., other].sum((1, 2))
    a, b = kT / nT, kO / nO
    return np.stack([a, b, a - b, a / b], 1)


def boot_contrast(d, col="is_exit", B=N_BOOT):
    N, X = student_matrices(d, col)
    point = contrast_stats(N, X, np.arange(len(N))[None, :])[0]
    bs = cluster_bootstrap(lambda idx: contrast_stats(N, X, idx), len(N), B=B, chunk=100)
    lo, hi = pct_ci(bs)
    return point, lo, hi


def icc_counts(n, k, w=None, min_n=2):
    """ANOVA intraclass correlation (binary outcome) from per-group event counts n and exit counts k.
    w = optional group multiplicities (used by the student bootstrap)."""
    w = np.ones(len(n)) if w is None else w
    keep = (n >= min_n) & (w > 0); n, k, w = n[keep], k[keep], w[keep]
    N, G = (w * n).sum(), w.sum(); gm = (w * k).sum() / N
    ssb = (w * n * (k / n - gm) ** 2).sum(); ssw = (w * (k - k ** 2 / n)).sum()
    msb, msw = ssb / (G - 1), ssw / (N - G); n0 = (N - (w * n ** 2).sum() / N) / (G - 1)
    return (msb - msw) / (msb + (n0 - 1) * msw)


import time
_t0 = time.time()
def T(msg): print(f'[{time.time()-_t0:5.0f}s] {msg}', flush=True)


def main():
    ev = load_events().sort_values(["QuizSessionId", "Timestamp"], kind="mergesort").reset_index(drop=True)
    sc = ev[ev.in_scope == 1].copy()
    out = {}

    # ---------- 1. exit rate by event type
    N, X = student_matrices(sc); U = len(N)
    R = np.divide(X, N, out=np.zeros_like(X), where=N > 0); M = (N > 0).astype(float)
    def st(idx):
        Nb, Xb = N[idx].sum(1), X[idx].sum(1)
        return np.stack([Xb / Nb, R[idx].sum(1) / M[idx].sum(1)], 1)
    bs = cluster_bootstrap(st, U, B=N_BOOT, chunk=100)
    ew, sw = X.sum(0) / N.sum(0), R.sum(0) / M.sum(0)
    lo_e, hi_e = pct_ci(bs[:, 0, :]); lo_s, hi_s = pct_ci(bs[:, 1, :])
    tab = pd.DataFrame({"state": STATES, "label": [LABELS[s] for s in STATES], "events": N.sum(0).astype(int), "students": M.sum(0).astype(int),
                        "exits": X.sum(0).astype(int), "event_rate": ew, "event_lo": lo_e, "event_hi": hi_e,
                        "student_rate": sw, "student_lo": lo_s, "student_hi": hi_s}).sort_values("event_rate", ascending=False)
    tab.to_csv(TAB / "q2_state_rates.csv", index=False)

    T('state rates')
    # ---------- 2. headline contrast (+ by calendar period)
    p, lo, hi = boot_contrast(sc)
    out["contrast"] = {k: {"est": float(p[i]), "lo": float(lo[i]), "hi": float(hi[i])} for i, k in enumerate(["rate_failed_retry", "rate_other", "difference", "ratio"])}
    for per in ("before", "after"):
        p2, lo2, hi2 = boot_contrast(sc[sc.period == per], B=500)
        out[f"contrast_{per}_break"] = {k: {"est": float(p2[i]), "lo": float(lo2[i]), "hi": float(hi2[i])} for i, k in enumerate(["rate_failed_retry", "rate_other", "difference", "ratio"])}
    kt = STATES.index(TARGET)
    out["share_of_all_exits_after_failed_retry"] = float(X[:, kt].sum() / X.sum())
    out["n_events"], out["n_students"], out["n_exits"] = int(N.sum()), int(U), int(X.sum())

    T('contrast')
    # ---------- 3. process structure: what happens next after each state
    sc["next_type"] = ev.groupby("QuizSessionId", sort=False).Type.shift(-1).reindex(sc.index)
    sc["next_node"] = np.select([sc.is_exit == 1, sc.next_type == "Checkin", sc.next_type == "CheckinRetry", sc.next_type == "Lesson",
                                 sc.next_type == "Checkout", sc.next_type == "CheckoutRetry"],
                                ["Quit", "Next question", "Retry", "Lesson", "Check-out", "Check-out retry"], "Other")
    tr = pd.crosstab(sc.state, sc.next_node)
    trp = tr.div(tr.sum(1), axis=0)
    long = tr.stack().rename("n").reset_index().merge(trp.stack().rename("prob").reset_index(), on=["state", "next_node"])
    long.to_csv(TAB / "q2_transitions.csv", index=False)
    fr = sc[sc.state == TARGET]
    out["after_failed_retry"] = {k: float((fr.next_node == k).mean()) for k in ["Lesson", "Check-out", "Quit", "Next question", "Other"]}
    lsn = sc[sc.state == "Lesson"]
    out["after_lesson"] = {"quit": float(lsn.is_exit.mean()), "n": int(len(lsn))}

    T('transitions')
    # ---------- 4. commitment after the lesson: exit after a failed check-out, by whether this question had a lesson
    lk = ev[ev.Type == "Lesson"][["QuizSessionId", "QuestionSequence"]].drop_duplicates().assign(had_lesson=1)
    co = sc[sc.state.isin(["Checkout_fail", "CheckoutRetry_fail"])].merge(lk, on=["QuizSessionId", "QuestionSequence"], how="left")
    co["had_lesson"] = co.had_lesson.fillna(0).astype(int)
    com = co.groupby(["state", "had_lesson"]).is_exit.agg(events="size", exit_rate="mean").reset_index()
    com.to_csv(TAB / "q2_commitment.csv", index=False)

    T('commitment')
    # ---------- 5. sensitivity of the headline ratio to how "exit" is defined, and to who is included
    variants = {"Primary: last event of a session that never reached Q5": (sc, "is_exit"),
                "Also counting any pause of over 1 hour": (sc, "is_exit_pause"),
                "Exit and no new session within 24 hours": (sc, "is_exit_noreturn"),
                "Primary, dropping sessions in the last 24 hours of data": (sc[sc.censor_risk == 0], "is_exit"),
                "Primary, excluding each student's first session": (sc[sc.user_session_idx > 1], "is_exit"),
                "Primary, school-hours sessions only": (sc[sc.school_hours == 1], "is_exit"),
                "Primary, Years 5-11 only": (sc[sc.YearGroup.between(5, 11)], "is_exit")}
    rows = []
    for name, (d, col) in variants.items():
        p, lo, hi = boot_contrast(d, col, B=300)
        rows.append({"definition": name, "exits": int(d[col].sum()), "rate_failed_retry": p[0], "rate_other": p[1], "ratio": p[3], "ratio_lo": lo[3], "ratio_hi": hi[3]})
    pd.DataFrame(rows).to_csv(TAB / "q2_sensitivity.csv", index=False)

    T('sensitivity')
    # ---------- 6. who versus what: share of variation in exit after a failed retry (ANOVA intraclass correlation)
    f = sc[sc.state == TARGET]
    users = np.sort(f.UserId.unique()); uidx = pd.Series(np.arange(len(users)), index=users)
    def mats(col, d):
        gid = pd.factorize(d[col])[0]; ng = gid.max() + 1; ui = uidx.reindex(d.UserId).values
        Nm = np.zeros((len(users), ng)); Km = np.zeros_like(Nm)
        np.add.at(Nm, (ui, gid), 1); np.add.at(Km, (ui, gid), d.is_exit.values)
        return Nm, Km
    Ns, Ks = np.bincount(uidx.reindex(f.UserId).values, minlength=len(users)).astype(float), np.bincount(uidx.reindex(f.UserId).values, weights=f.is_exit.values, minlength=len(users))
    Nt, Kt = mats("SubjectId", f[f.SubjectId.notna()].reset_index(drop=True)); Nc, Kc = mats("ConstructId", f.reset_index(drop=True))
    def all_icc(w):
        return [icc_counts(Ns, Ks, w), icc_counts(w @ Nt, w @ Kt), icc_counts(w @ Nc, w @ Kc)]
    pt = all_icc(np.ones(len(users)))
    rng = np.random.default_rng(SEED); bsI = np.array([all_icc(np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users)).astype(float)) for _ in range(300)])
    se_i = bsI.std(axis=0, ddof=1)       # a percentile interval is biased for topic-level ICCs (resampling duplicates events), so use est +/- 1.96 bootstrap SE
    # permutation test: shuffle topic / concept labels across events (destroys any topic effect); p = share of null ICCs >= observed
    perm_p = []
    for col, d in (("SubjectId", f[f.SubjectId.notna()]), ("ConstructId", f)):
        gid = pd.factorize(d[col])[0]; y = d.is_exit.values.astype(float); prng = np.random.default_rng(SEED + 7); obs = icc_counts(np.bincount(gid).astype(float), np.bincount(gid, weights=y))
        null = [icc_counts(np.bincount(pg).astype(float), np.bincount(pg, weights=y)) for pg in (prng.permutation(gid) for _ in range(1000))]
        perm_p.append(float((1 + np.sum(np.array(null) >= obs)) / (1 + len(null)))); 
    pd.DataFrame({"level": ["student", "topic (subject)", "concept (construct)"], "icc": pt, "lo": np.array(pt) - 1.96 * se_i, "hi": np.array(pt) + 1.96 * se_i,
                  "perm_p": [np.nan] + perm_p}).to_csv(TAB / "q2_icc.csv", index=False)
    T("ICC")

    json.dump(out, open(TAB / "q2_summary.json", "w"), indent=2, default=float)
    print(json.dumps(out, indent=1, default=float))
    for f_ in ["q2_state_rates", "q2_sensitivity", "q2_commitment", "q2_icc"]:
        print("\n" + f_); print(pd.read_csv(TAB / f"{f_}.csv").round(4).to_string(index=False))
    print("\n transitions (row probs):"); print(trp.round(3).to_string())


if __name__ == "__main__":
    main()
