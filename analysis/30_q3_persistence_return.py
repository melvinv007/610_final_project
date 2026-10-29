"""RQ3 - Is quitting persistent, and do quitters return?
Unit = quiz session (ordered within student). Exit = session ended before question 5 (is_exit).
Part A persistence: P(exit | previous session was an exit) vs not; by gap since previous session; student-clustered bootstrap;
        trait-vs-state: leave-one-out student rate control, within-student (fixed-effects) LPM, split-half reliability.
Part B return: time from session end to the student's next session; right-censored at end of data; weighted Kaplan-Meier by
        previous-session type (exit after a failed retry / other exit / completed); student bootstrap; Cox PH (clustered).
Writes results/tables/q3_*.csv, q3_summary.json
"""
import json, time, warnings
import numpy as np, pandas as pd
from common import TAB, SEED, load_events, weighted_km
warnings.filterwarnings("ignore")
_t0 = time.time()
def T(m): print(f"[{time.time()-_t0:5.0f}s] {m}", flush=True)
H = 3600.0


def sessions():
    ev = load_events().sort_values(["QuizSessionId", "Timestamp"], kind="mergesort")
    g = ev.groupby("QuizSessionId", sort=False)
    s = g.agg(UserId=("UserId", "first"), QuizId=("QuizId", "first"), start=("Timestamp", "min"), end=("Timestamp", "max"),
              n_ev=("Timestamp", "size"), scope=("in_scope", "max"), period=("period", "first"), wkend=("wkend", "first"))
    ex = ev[ev.is_exit == 1].drop_duplicates("QuizSessionId").set_index("QuizSessionId")
    s["exit"] = s.index.isin(ex.index).astype(int)
    s["exit_fr"] = (s.index.isin(ex.index[ex.state == "CheckinRetry_fail"])).astype(int)
    s["kind"] = np.where(s.exit_fr == 1, "exit after failed retry", np.where(s.exit == 1, "other exit", "completed"))
    s = s[s.scope == 1].reset_index().sort_values(["UserId", "start"], kind="mergesort").reset_index(drop=True)
    s["k"] = s.groupby("UserId").cumcount()
    nxt = s.groupby("UserId").start.shift(-1); s["next_gap_s"] = (nxt - s.end).dt.total_seconds().clip(lower=0)
    s["next_same_quiz"] = (s.groupby("UserId").QuizId.shift(-1) == s.QuizId).astype(float).where(nxt.notna())
    s["prev_exit"] = s.groupby("UserId").exit.shift(1); s["prev_kind"] = s.groupby("UserId").kind.shift(1)
    s["prev_gap_s"] = s.groupby("UserId").next_gap_s.shift(1)
    return s


def rate_ci(s, col, boot_ids, B=400, rng=None):
    pass


def main():
    s = sessions(); T(f"{len(s):,} sessions, {s.UserId.nunique():,} students, exit rate {s.exit.mean():.4f}")
    data_end = load_events().Timestamp.max()
    uid, ucode = pd.factorize(s.UserId); s["u"] = uid; nU = len(ucode)
    rng = np.random.default_rng(SEED)
    out = {"sessions": int(len(s)), "students": int(nU), "exit_rate": float(s.exit.mean())}

    # ---------------- Part A: persistence
    P = s[s.prev_exit.notna()].copy()
    P["gap_bin"] = pd.cut(P.prev_gap_s / H, [-1, 1, 24, 24 * 7, 1e9], labels=["<1 h (same sitting)", "1-24 h", "1-7 d", ">7 d"])
    def cell(df):
        return pd.Series({"n": len(df), "rate_after_exit": df.exit[df.prev_exit == 1].mean(), "rate_after_complete": df.exit[df.prev_exit == 0].mean(),
                          "n_after_exit": int((df.prev_exit == 1).sum())})
    # student-level sufficient stats for fast cluster bootstrap
    def stats(df, key=None):
        d = df if key is None else df[key]
        a = d.assign(pe=d.prev_exit.astype(int)).groupby(["u", "pe"]).exit.agg(["sum", "size"]).unstack(fill_value=0)
        a = a.reindex(range(nU), fill_value=0); return a
    rows = []
    groups = [("all", np.ones(len(P), bool))] + [(str(b), (P.gap_bin == b).values) for b in P.gap_bin.cat.categories]
    for name, m in groups:
        A = stats(P[m]); k1, n1 = A[("sum", 1)].values, A[("size", 1)].values; k0, n0 = A[("sum", 0)].values, A[("size", 0)].values
        def est(w): 
            return (w @ k1) / max(w @ n1, 1), (w @ k0) / max(w @ n0, 1)
        r1, r0 = est(np.ones(nU)); bs = []
        for _ in range(500):
            w = rng.multinomial(nU, np.ones(nU) / nU).astype(float); a, b = est(w); bs.append((a, b, a - b, a / b if b > 0 else np.nan))
        bs = np.array(bs)
        rows.append({"gap": name, "n_pairs": int(m.sum()), "n_after_exit": int(n1.sum()), "rate_after_exit": r1, "rate_after_complete": r0, "diff": r1 - r0,
                     "diff_lo": np.percentile(bs[:, 2], 2.5), "diff_hi": np.percentile(bs[:, 2], 97.5), "ratio": r1 / r0,
                     "ratio_lo": np.nanpercentile(bs[:, 3], 2.5), "ratio_hi": np.nanpercentile(bs[:, 3], 97.5)})
    # by previous-session kind
    for kd in ["exit after failed retry", "other exit"]:
        m = ((P.prev_kind == kd) | (P.prev_exit == 0)).values; Q = P[m]
        A = stats(Q); k1, n1, k0, n0 = A[("sum", 1)].values, A[("size", 1)].values, A[("sum", 0)].values, A[("size", 0)].values
        r1, r0 = k1.sum() / n1.sum(), k0.sum() / n0.sum(); bs = []
        for _ in range(300):
            w = rng.multinomial(nU, np.ones(nU) / nU).astype(float); bs.append((w @ k1) / (w @ n1) - (w @ k0) / (w @ n0))
        rows.append({"gap": "prev: " + kd, "n_pairs": int(len(Q)), "n_after_exit": int(n1.sum()), "rate_after_exit": r1, "rate_after_complete": r0, "diff": r1 - r0,
                     "diff_lo": np.percentile(bs, 2.5), "diff_hi": np.percentile(bs, 97.5), "ratio": r1 / r0, "ratio_lo": np.nan, "ratio_hi": np.nan})
    persist = pd.DataFrame(rows); persist.to_csv(TAB / "q3_persistence.csv", index=False); print(persist.round(4).to_string(index=False)); T("persistence")

    # trait vs state
    tot = s.groupby("u").exit.transform("sum"); cnt = s.groupby("u").exit.transform("size")
    s["loo"] = ((tot - s.exit) / (cnt - 1)).where(cnt > 1)
    P = s[s.prev_exit.notna() & s.loo.notna()].copy(); P["after"] = (P.period == "after").astype(float)
    P["loo"] = P.loo.astype(float)
    P["gap_same"] = (P.prev_gap_s < H).astype(float); P["pe"] = P.prev_exit.astype(float)
    def ols_cluster(y, X, g):
        X = np.column_stack([np.ones(len(y)), X]); b = np.linalg.lstsq(X, y, rcond=None)[0]; u = y - X @ b
        XtXi = np.linalg.inv(X.T @ X); S = np.zeros((X.shape[1],) * 2)
        sc = pd.DataFrame(X * u[:, None]).groupby(g).sum().values; S = sc.T @ sc
        V = XtXi @ S @ XtXi; return b, np.sqrt(np.diag(V))
    tv = []
    b, se = ols_cluster(P.exit.values.astype(float), P[["pe"]].values, P.u.values); tv.append(("raw (no control)", b[1], se[1]))
    b, se = ols_cluster(P.exit.values.astype(float), P[["pe", "loo", "after"]].values, P.u.values); tv.append(("controlling student's usual rate (leave-one-out) and period", b[1], se[1]))
    # within-student LPM
    Wc = P[["exit", "pe", "after", "gap_same"]].astype(float); Wd = Wc - Wc.groupby(P.u.values).transform("mean")
    keep = P.groupby("u").exit.transform("size") >= 5
    Wd, Pk = Wd[keep], P[keep]
    X = Wd[["pe", "after", "gap_same"]].values; y = Wd.exit.values
    XtXi = np.linalg.inv(X.T @ X); bw = XtXi @ X.T @ y; uu = y - X @ bw
    sc = pd.DataFrame(X * uu[:, None]).groupby(Pk.u.values).sum().values; Vw = XtXi @ (sc.T @ sc) @ XtXi
    tv.append(("within student (fixed effects)", bw[0], np.sqrt(Vw[0, 0])))
    trait = pd.DataFrame([{"specification": n, "effect_of_previous_exit": b_, "se": s_, "lo": b_ - 1.96 * s_, "hi": b_ + 1.96 * s_} for n, b_, s_ in tv])
    trait.to_csv(TAB / "q3_trait_state.csv", index=False); print(trait.round(4).to_string(index=False)); T("trait/state")
    # split-half reliability
    ss = s[s.k >= 0].copy(); cnt = ss.groupby("u").exit.transform("size"); ss = ss[cnt >= 10]; rel = []
    for _ in range(100):
        ss["half"] = rng.integers(0, 2, len(ss)); t = ss.groupby(["u", "half"]).exit.mean().unstack().dropna()
        rel.append(np.corrcoef(t[0], t[1])[0, 1])
    r = float(np.mean(rel)); out.update({"splithalf_r": r, "spearman_brown": 2 * r / (1 + r), "splithalf_students": int(ss.u.nunique())})
    # exit share concentration
    st = s.groupby("u").exit.agg(["sum", "size"]); st = st[st["size"] >= 5]
    out.update({"students_ge5": int(len(st)), "share_never_exit_ge5": float((st["sum"] == 0).mean()),
                "share_of_exits_by_top10pct": float(st["sum"].sort_values(ascending=False).head(int(len(st) * .1)).sum() / st["sum"].sum())})
    T("reliability")

    # ---------------- Part B: return
    R = s[s.exit.notna()].copy()
    R["event"] = R.next_gap_s.notna().astype(int)
    R["dur"] = np.where(R.event == 1, R.next_gap_s, (data_end - R.end).dt.total_seconds().clip(lower=0))
    grid = np.array([0.25, 1, 3, 6, 12, 24, 48, 24 * 7, 24 * 14, 24 * 30]) * H
    kinds = ["exit after failed retry", "other exit", "completed"]; res = []; curves = []
    egrid = np.unique(np.concatenate([np.linspace(0, 30 * 24 * H, 181), grid]))
    d_all, e_all, u_all, k_all = R.dur.values, R.event.values, R.u.values, R.kind.values
    point = {k: 1 - weighted_km(d_all[k_all == k], e_all[k_all == k].astype(float), np.ones((k_all == k).sum()), egrid) for k in kinds}
    boots = {k: [] for k in kinds}
    for _ in range(200):
        wu = rng.multinomial(nU, np.ones(nU) / nU).astype(float); w = wu[u_all]
        for k in kinds:
            m = k_all == k; boots[k].append(1 - weighted_km(d_all[m], e_all[m].astype(float), w[m], egrid))
    for k in kinds:
        B = np.array(boots[k]); lo, hi = np.nanpercentile(B, 2.5, axis=0), np.nanpercentile(B, 97.5, axis=0)
        for t, p_, l_, h_ in zip(egrid, point[k], lo, hi): curves.append({"kind": k, "hours": t / H, "p_return": p_, "lo": l_, "hi": h_})
        for gt in grid:
            i = np.searchsorted(egrid, gt); res.append({"kind": k, "within_hours": gt / H, "p_return": point[k][i], "lo": lo[i], "hi": hi[i], "n_sessions": int((k_all == k).sum())})
    pd.DataFrame(curves).to_csv(TAB / "q3_return_curves.csv", index=False)
    rt = pd.DataFrame(res); rt.to_csv(TAB / "q3_return_probs.csv", index=False)
    print(rt[rt.within_hours.isin([1, 24, 168, 720])].round(3).to_string(index=False)); T("KM")

    # same-quiz return, median time among returners, never-return share
    ret = R[R.event == 1]
    q = ret.groupby("kind").agg(returners=("event", "size"), same_quiz=("next_same_quiz", "mean"), median_gap_h=("dur", lambda x: np.median(x) / H)).reset_index()
    q["censored_share"] = q.kind.map(R.groupby("kind").event.apply(lambda x: 1 - x.mean())); q.to_csv(TAB / "q3_return_summary.csv", index=False); print(q.round(3).to_string(index=False))

    # Adjusted odds of returning within 24 h / 7 d (sessions with full follow-up), student-clustered logit
    import statsmodels.api as sm
    rows_ = []
    for hrs in (24, 168):
        Rc = R[(data_end - R.end).dt.total_seconds() >= hrs * H].copy(); y = ((Rc.event == 1) & (Rc.dur <= hrs * H)).astype(float)
        X = pd.DataFrame({"exit after failed retry": (Rc.kind == "exit after failed retry").astype(float), "other exit": (Rc.kind == "other exit").astype(float),
                          "after break": (Rc.period == "after").astype(float), "weekend": Rc.wkend.astype(float), "session number (log)": np.log1p(Rc.k)})
        m = sm.Logit(y.values, sm.add_constant(X)).fit(disp=0, cov_type="cluster", cov_kwds={"groups": Rc.u.values}); ci = m.conf_int()
        for t in X.columns: rows_.append({"horizon_h": hrs, "term": t, "OR": np.exp(m.params[t]), "lo": np.exp(ci.loc[t, 0]), "hi": np.exp(ci.loc[t, 1]), "p": m.pvalues[t], "n": len(Rc)})
    cox = pd.DataFrame(rows_); cox.to_csv(TAB / "q3_return_or.csv", index=False); print(cox.round(3).to_string(index=False)); T("return logit")

    out.update({"last_session_censored_share": float(1 - R.event.mean())})
    json.dump(out, open(TAB / "q3_summary.json", "w"), indent=2); print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
