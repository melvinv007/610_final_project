"""RQ1 - Is the lesson step linked to learning?

Situation  : a student has just failed a check-in question twice (check-in and its retry). The platform then offers a
             lesson; some students take it, some go straight to the check-out question (a fresh question on the same
             concept), and some quit (quitters have no outcome and are not part of this question; see RQ2).
Exposure   : lesson taken (and for how long: the gap between the Lesson event and the check-out) vs lesson skipped.
Outcome    : first-attempt check-out correct (1/0).
Caution    : observational. Students are not randomised to lessons, so we (a) compare the SAME students with and without
             lessons, (b) adjust for context, (c) check covariate balance, (d) run a placebo outcome that the lesson
             cannot have caused, and (e) report an E-value (how strong hidden confounding would have to be).

Writes  results/tables/q1_*.csv|json
"""
import json, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from common import TAB, SEED, load_events, cluster_bootstrap, pct_ci

warnings.filterwarnings("ignore")
N_BOOT = 1000
GLANCE, READ = 10, 60           # lesson-time cut-points in seconds: glance <=10 s < read <=60 s < long


def build():
    ev = load_events().sort_values(["QuizSessionId", "Timestamp"], kind="mergesort").reset_index(drop=True)
    g = ev.groupby("QuizSessionId", sort=False)
    for k in (1, 2):
        ev[f"n{k}_type"], ev[f"n{k}_ok"], ev[f"n{k}_ts"] = g.Type.shift(-k), g.IsCorrect.shift(-k), g.Timestamp.shift(-k)
    # first-attempt check-in accuracy history of each student (excluding the current question)
    ci = ev[ev.Type == "Checkin"].sort_values(["UserId", "Timestamp"]).copy()
    ci["hist_n"] = ci.groupby("UserId").cumcount()
    ci["prior_acc"] = (ci.groupby("UserId").IsCorrect.cumsum() - ci.IsCorrect) / ci.hist_n.replace(0, np.nan)
    ci["sess_prior_acc"] = ((ci.groupby("QuizSessionId").IsCorrect.cumsum() - ci.IsCorrect) /
                            ci.groupby("QuizSessionId").cumcount().replace(0, np.nan))
    ci["constr_acc"] = ci.groupby("ConstructId").IsCorrect.transform("mean")
    keys = ci[["QuizSessionId", "QuestionSequence", "prior_acc", "hist_n", "sess_prior_acc", "constr_acc"]]
    fr = ev[ev.state == "CheckinRetry_fail"].merge(keys, on=["QuizSessionId", "QuestionSequence"], how="left")
    lesson = (fr.n1_type == "Lesson") & (fr.n2_type == "Checkout")
    skip = fr.n1_type == "Checkout"
    L = fr[lesson | skip].copy()
    L["lesson"] = lesson[lesson | skip].astype(int)
    L["y"] = np.where(L.lesson == 1, L.n2_ok, L.n1_ok)
    L["dwell"] = np.where(L.lesson == 1, (L.n2_ts - L.n1_ts).dt.total_seconds(), np.nan)
    L["cat"] = np.select([L.lesson == 0, L.dwell <= GLANCE, L.dwell <= READ], ["skip", "glance", "read"], "long")
    L["lat"] = np.log1p(L.latency_s.clip(0, 600).fillna(0))
    L["pfail"] = L.n_prior_fail.clip(upper=4)
    L["yg"] = L.YearGroup.fillna(8).clip(5, 11)
    L["after"] = (L.period == "after").astype(int)
    L["sess_log"] = np.log1p(L.user_session_idx)
    L["no_hist"] = L.prior_acc.isna().astype(int)
    L["prior_acc"] = L.prior_acc.fillna(L.prior_acc.mean())
    L["sess_prior_acc_f"] = L.sess_prior_acc.fillna(L.sess_prior_acc.mean())
    return L, len(fr), int((fr.n1_type.isna() | ((fr.n1_type != "Lesson") & (fr.n1_type != "Checkout"))).sum())


def design(L, with_student_const=True):
    X = pd.DataFrame({"lat": L.lat, "after": L.after, "school_hours": L.school_hours, "wkend": L.wkend,
                      "prior_acc": L.prior_acc, "no_hist": L.no_hist, "constr_acc": L.constr_acc, "sess_log": L.sess_log,
                      "sess_prior_acc": L.sess_prior_acc_f}, index=L.index)
    X = X.join(pd.get_dummies(L.QuestionSequence, prefix="q", drop_first=True).astype(float))
    X = X.join(pd.get_dummies(L.pfail, prefix="pf", drop_first=True).astype(float))
    X = X.join(pd.get_dummies(L.yg, prefix="yg", drop_first=True).astype(float))
    return X


def fe_ols(y, X, groups, cols):
    """Linear probability model with student fixed effects (within transformation) and student-clustered SEs."""
    y, X, groups = y.reset_index(drop=True), X.reset_index(drop=True), pd.Series(np.asarray(groups))
    d = pd.concat([y.rename("y"), X[cols]], axis=1)
    dm = d - d.groupby(groups.values).transform("mean")
    dm = dm.loc[:, dm.abs().sum() > 0]
    res = sm.OLS(dm.y, dm.drop(columns="y")).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    return res


def cp(res, name):
    b, ci = res.params[name], res.conf_int().loc[name]
    return {"est": float(b), "lo": float(ci[0]), "hi": float(ci[1]), "p": float(res.pvalues[name])}


def smd(a, b):
    return float((a.mean() - b.mean()) / np.sqrt((a.var() + b.var()) / 2))


def main():
    L, n_fr, n_noexp = build()
    users = np.sort(L.UserId.unique()); uidx = pd.Series(np.arange(len(users)), index=users); U = len(users)
    ui = uidx.reindex(L.UserId).values
    out = {"failed_retry_events": n_fr, "events_with_checkout": int(len(L)), "students": int(U),
           "n_lesson": int(L.lesson.sum()), "n_skip": int((1 - L.lesson).sum()),
           "skip_share_of_checkout_route": float(1 - L.lesson.mean())}

    # ---------- 1. descriptives by exposure category
    def agg(mask_cat):
        m = np.zeros((U, 2)); np.add.at(m, (ui[mask_cat], 0), L.y.values[mask_cat]); np.add.at(m, (ui[mask_cat], 1), 1); return m
    rows = []
    for c in ["skip", "glance", "read", "long"]:
        m = agg((L.cat == c).values)
        bs = cluster_bootstrap(lambda idx: m[idx, 0].sum(1) / m[idx, 1].sum(1), U, B=N_BOOT, chunk=250)
        lo, hi = pct_ci(bs)
        rows.append({"category": c, "events": int(m[:, 1].sum()), "students": int((m[:, 1] > 0).sum()), "success": m[:, 0].sum() / m[:, 1].sum(), "lo": lo, "hi": hi,
                     "median_dwell_s": float(L.dwell[L.cat == c].median()) if c != "skip" else np.nan})
    pd.DataFrame(rows).to_csv(TAB / "q1_by_category.csv", index=False)
    # fine dwell bins for the figure
    edges = [0, 5, 10, 15, 20, 30, 45, 60, 90, 150, 300, 1e9]
    lab = ["<=5", "5-10", "10-15", "15-20", "20-30", "30-45", "45-60", "60-90", "90-150", "150-300", ">300"]
    L["bin"] = pd.cut(L.dwell, edges, labels=lab)
    rows = []
    for b in lab:
        msk = (L.bin == b).values; m = agg(msk)
        bs = cluster_bootstrap(lambda idx: m[idx, 0].sum(1) / np.maximum(m[idx, 1].sum(1), 1), U, B=500, chunk=250)
        lo, hi = pct_ci(bs); rows.append({"bin": b, "events": int(m[:, 1].sum()), "success": m[:, 0].sum() / m[:, 1].sum(), "lo": lo, "hi": hi})
    pd.DataFrame(rows).to_csv(TAB / "q1_dose_fine.csv", index=False)

    # ---------- 2. covariate balance (lesson vs skip)
    cov = {"Time to fail the retry (log s)": "lat", "Earlier failures in session": "n_prior_fail", "Question position": "QuestionSequence",
           "Student prior check-in accuracy": "prior_acc", "Accuracy earlier in this session": "sess_prior_acc_f", "Concept difficulty (mean accuracy)": "constr_acc",
           "Year group": "yg", "After calendar break": "after", "School-hours session": "school_hours", "Student's session number (log)": "sess_log"}
    bal = pd.DataFrame([{"covariate": k, "mean_lesson": L.loc[L.lesson == 1, v].mean(), "mean_skip": L.loc[L.lesson == 0, v].mean(),
                         "smd": smd(L.loc[L.lesson == 1, v], L.loc[L.lesson == 0, v])} for k, v in cov.items()])
    bal.to_csv(TAB / "q1_balance.csv", index=False)

    # ---------- 3. effect estimates, lesson (any) vs skip
    nl, kl, ns, ks = [np.zeros(U) for _ in range(4)]
    np.add.at(nl, ui[L.lesson == 1], 1); np.add.at(kl, ui[L.lesson == 1], L.y[L.lesson == 1]); np.add.at(ns, ui[L.lesson == 0], 1); np.add.at(ks, ui[L.lesson == 0], L.y[L.lesson == 0])
    est = []
    raw = lambda idx: (kl[idx].sum(1) / nl[idx].sum(1) - ks[idx].sum(1) / ns[idx].sum(1))
    bs = cluster_bootstrap(raw, U, B=N_BOOT, chunk=250); lo, hi = pct_ci(bs)
    est.append({"estimator": "Raw difference (all events)", "est": float(raw(np.arange(U)[None, :])[0]), "lo": lo, "hi": hi, "n": int(len(L))})
    both = (nl > 0) & (ns > 0); db = kl[both] / nl[both] - ks[both] / ns[both]; nb = int(both.sum())
    rng = np.random.default_rng(SEED); bsd = [db[rng.integers(0, nb, nb)].mean() for _ in range(N_BOOT)]
    est.append({"estimator": "Same students, with vs without lesson", "est": float(db.mean()), "lo": float(np.percentile(bsd, 2.5)), "hi": float(np.percentile(bsd, 97.5)), "n": nb})
    X = design(L); groups = L.UserId.reset_index(drop=True); X = X.reset_index(drop=True); y = L.y.reset_index(drop=True)
    X["lesson"] = L.lesson.values
    cols_tv = list(X.columns)
    fe = fe_ols(y, X, groups, cols_tv); c = cp(fe, "lesson")
    est.append({"estimator": "Student fixed effects + context controls", "est": c["est"], "lo": c["lo"], "hi": c["hi"], "n": int(both.sum()), "p": c["p"]})
    # adjusted logit, average marginal effect
    Xl = sm.add_constant(X)
    lg = sm.Logit(y, Xl).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]})
    me = lg.get_margeff(at="overall").summary_frame().loc["lesson"]
    est.append({"estimator": "Adjusted logistic regression (marginal effect)", "est": float(me["dy/dx"]), "lo": float(me["Conf. Int. Low"]), "hi": float(me["Cont. Int. Hi."]), "n": int(len(L))})
    out["logit_or_lesson"] = float(np.exp(lg.params["lesson"])); out["logit_or_ci"] = [float(np.exp(v)) for v in lg.conf_int().loc["lesson"]]
    # IPW (stabilised, trimmed), bootstrap over students
    Z = sm.add_constant(X.drop(columns="lesson")); t = L.lesson.values.astype(float); yy = y.values
    def ipw(wst):
        pm = sm.GLM(t, Z, family=sm.families.Binomial(), freq_weights=wst).fit()
        ps = np.clip(pm.predict(Z), 0.02, 0.98)
        w1 = wst * t / ps; w0 = wst * (1 - t) / (1 - ps)
        return (w1 * yy).sum() / w1.sum() - (w0 * yy).sum() / w0.sum()
    point = ipw(np.ones(len(L)))
    rng = np.random.default_rng(SEED); bi = []
    for _ in range(200):
        wst = np.bincount(rng.integers(0, U, U), minlength=U).astype(float)[ui]
        bi.append(ipw(wst))
    est.append({"estimator": "Inverse-probability weighting", "est": float(point), "lo": float(np.percentile(bi, 2.5)), "hi": float(np.percentile(bi, 97.5)), "n": int(len(L))})
    pd.DataFrame(est).to_csv(TAB / "q1_effects.csv", index=False)

    # ---------- 4. dose-response with student fixed effects (reference = skipped the lesson)
    Xd = design(L).reset_index(drop=True)
    for cat in ["glance", "read", "long"]:
        Xd[cat] = (L.cat.values == cat).astype(float)
    fe2 = fe_ols(y, Xd, groups, list(Xd.columns))
    dose = pd.DataFrame([{"category": cat, **cp(fe2, cat)} for cat in ["glance", "read", "long"]]); dose.to_csv(TAB / "q1_dose_fe.csv", index=False)

    # ---------- 5. sensitivity
    sens = {}
    # (a) placebo outcome: accuracy EARLIER in the same session (cannot be caused by a lesson that comes later)
    Xp = design(L).reset_index(drop=True).drop(columns="sess_prior_acc"); Xp["lesson"] = L.lesson.values
    has = L.sess_prior_acc.notna().values
    pl = fe_ols(L.sess_prior_acc_f.reset_index(drop=True)[has], Xp[has].reset_index(drop=True), groups[has].reset_index(drop=True), list(Xp.columns))
    sens["placebo_session_accuracy_FE"] = cp(pl, "lesson")
    sens["placebo_session_accuracy_raw_diff"] = float(L.loc[(L.lesson == 1) & has, "sess_prior_acc"].mean() - L.loc[(L.lesson == 0) & has, "sess_prior_acc"].mean())
    # (b) alternative samples
    def fe_effect(mask):
        r = fe_ols(y[mask.values], X[mask.values].reset_index(drop=True), groups[mask.values].reset_index(drop=True), cols_tv); return cp(r, "lesson")
    sens["fe_before_break"] = fe_effect(L.after == 0); sens["fe_after_break"] = fe_effect(L.after == 1)
    sens["fe_excl_long_gaps_gt_600s"] = fe_effect(~(L.dwell > 600)); sens["fe_questions_1_to_4_only"] = fe_effect(L.QuestionSequence <= 4)
    # (c) E-value (VanderWeele & Ding 2017) for the adjusted marginal effect
    p_les = L.y[L.lesson == 1].mean(); ate = est[3]["est"]; p0 = p_les - ate
    def evalue(rr): return float(rr + np.sqrt(rr * (rr - 1))) if rr > 1 else 1.0
    rr = p_les / p0; rr_lo = (p0 + est[3]["lo"]) / p0
    sens["risk_ratio_adjusted"] = float(rr); sens["e_value_point"] = evalue(rr); sens["e_value_ci_limit"] = evalue(rr_lo)
    out["sensitivity"] = sens
    out["effects"] = est
    json.dump(out, open(TAB / "q1_summary.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "effects"}, indent=1, default=float))
    print(pd.DataFrame(est).round(4).to_string(index=False))
    print(pd.DataFrame(rows).round(3).to_string(index=False) if False else "")
    print(pd.read_csv(TAB / "q1_by_category.csv").round(3).to_string(index=False)); print(dose.round(4).to_string(index=False)); print(bal.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
