"""RQ2 (part 2) - Discrete-time hazard models of quitting.

A quiz session is a sequence of steps (events). At every step the student either continues or quits. The probability of
quitting at a step GIVEN that the student has reached it is the *hazard*; a logistic regression on the step-level data is the
standard discrete-time survival model. Time-varying covariates are what make it temporal: the event type, the position in the
quiz, the failures accumulated so far, the time taken to answer and the time elapsed in the session.

Part A  nested models M0..M4 (does each temporal ingredient add predictive power for UNSEEN students? 5-fold CV grouped by student)
Part B  failed-retry steps only: adjusted odds ratios and standardised (g-computation) quit probabilities by answering speed,
        earlier failed retries and question position, with student-clustered bootstrap intervals
Part C  the same model with STUDENT FIXED EFFECTS (conditional logit): does the pattern survive when every student is only
        compared with themselves?  If the falling hazard were just 'persistent students stay', it would vanish here.
Writes  results/tables/q2_hazard_models.csv, q2_hazard_or_full.csv, q2_fr_or.csv, q2_fr_marginal.csv, q2_fr_fe.csv, q2_hazard_summary.json
"""
import json, time, warnings
import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from common import TAB, SEED, load_events

warnings.filterwarnings("ignore")
_t0 = time.time()
def T(m): print(f"[{time.time()-_t0:5.0f}s] {m}", flush=True)
LAT_EDGES = [0, 3, 6, 12, 25, 60, 1e9]; LAT_LAB = ["<=3 s", "3-6 s", "6-12 s", "12-25 s", "25-60 s", ">60 s"]
LAT_REF = "12-25 s"


def prepare():
    ev = load_events().sort_values(["QuizSessionId", "Timestamp"], kind="mergesort").reset_index(drop=True)
    ev["fr"] = (ev.state == "CheckinRetry_fail").astype(int)
    ev["prior_fr"] = (ev.groupby("QuizSessionId", sort=False).fr.cumsum() - ev.fr).clip(upper=3)
    d = ev[(ev.in_scope == 1) & (ev.state != "Lesson")].copy()
    d["y"] = d.is_exit.astype(float)
    d["first_event"] = d.latency_s.isna().astype(int)
    d["lat"] = np.log1p(d.latency_s.clip(0, 600).fillna(0))
    d["lat_bin"] = pd.cut(d.latency_s.fillna(1e-3), LAT_EDGES, labels=LAT_LAB)
    d["mins"] = np.log1p(d.since_start_s / 60)
    d["after"] = (d.period == "after").astype(int)
    d["sess_log"] = np.log1p(d.user_session_idx)
    d["yg"] = d.YearGroup.fillna(8).clip(5, 11).astype(int)
    return d


def block(d, kind):
    if kind == "state":
        return pd.get_dummies(d.state, prefix="st").drop(columns="st_Checkin_ok").astype(float)
    if kind == "position":
        return pd.get_dummies(d.QuestionSequence, prefix="q", drop_first=True).astype(float)
    if kind == "cumulative":
        return pd.get_dummies(d.prior_fr, prefix="pfr", drop_first=True).astype(float)
    if kind == "timing":
        return pd.DataFrame({"lat": d.lat, "first_event": d.first_event, "mins": d.mins}, index=d.index)
    if kind == "context":
        return pd.DataFrame({"after": d.after, "school_hours": d.school_hours, "wkend": d.wkend, "sess_log": d.sess_log}, index=d.index).join(
            pd.get_dummies(d.yg, prefix="yg", drop_first=True).astype(float))
    raise KeyError(kind)


def main():
    d = prepare(); T(f"data {len(d):,} steps, {d.UserId.nunique():,} students, {int(d.y.sum()):,} exits")
    y = d.y.values; groups = d.UserId.values
    spec = {"M0 event type": ["state"], "M1 + position in quiz": ["state", "position"], "M2 + failures so far": ["state", "position", "cumulative"],
            "M3 + answering speed, time in session": ["state", "position", "cumulative", "timing"],
            "M4 + calendar and student context": ["state", "position", "cumulative", "timing", "context"]}
    folds = list(GroupKFold(n_splits=5).split(d, y, groups))
    rows, base_ll = [], None
    for name, blocks in spec.items():
        X = pd.concat([block(d, b) for b in blocks], axis=1).values.astype(float)
        mk = lambda: LogisticRegression(penalty=None, solver="newton-cholesky", max_iter=100)
        full = mk().fit(X, y); p = full.predict_proba(X)[:, 1]
        ll = float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))); k = X.shape[1] + 1
        auc, lls = [], []
        for tr, te in folds:
            m = mk().fit(X[tr], y[tr]); pt = m.predict_proba(X[te])[:, 1]
            auc.append(roc_auc_score(y[te], pt)); lls.append(log_loss(y[te], pt))
        p0 = y.mean(); ll0 = float(np.sum(y * np.log(p0) + (1 - y) * np.log(1 - p0)))
        rows.append({"model": name, "parameters": k, "loglik": ll, "aic": 2 * k - 2 * ll, "mcfadden_r2": 1 - ll / ll0, "cv_auc": np.mean(auc), "cv_auc_sd": np.std(auc),
                     "cv_logloss": np.mean(lls)})
        T(name)
    mod = pd.DataFrame(rows); mod.to_csv(TAB / "q2_hazard_models.csv", index=False); print(mod.round(4).to_string(index=False))

    # full model with student-clustered standard errors (M4)
    X4 = pd.concat([block(d, b) for b in spec["M4 + calendar and student context"]], axis=1)
    res = sm.Logit(y, sm.add_constant(X4.astype(float))).fit(disp=0, cov_type="cluster", cov_kwds={"groups": pd.factorize(groups)[0]}, maxiter=100)
    ci = res.conf_int()
    full = pd.DataFrame({"term": res.params.index, "OR": np.exp(res.params.values), "lo": np.exp(ci[0].values), "hi": np.exp(ci[1].values), "p": res.pvalues.values})
    full.to_csv(TAB / "q2_hazard_or_full.csv", index=False); T("full cluster model")

    # ---------- Part B: failed-retry steps
    F = d[d.state == "CheckinRetry_fail"].copy().reset_index(drop=True)
    F["lat_bin"] = pd.Categorical(F.lat_bin, categories=LAT_LAB)
    def XF(df, with_student_const=True):
        df = df.assign(prior_fr=pd.Categorical(df.prior_fr, categories=[0, 1, 2, 3]),
                       QuestionSequence=pd.Categorical(df.QuestionSequence, categories=[1, 2, 3, 4]),
                       yg=pd.Categorical(df.yg, categories=list(range(5, 12))))
        parts = [pd.get_dummies(df.lat_bin, prefix="lat").drop(columns=f"lat_{LAT_REF}").astype(float),
                 pd.get_dummies(df.prior_fr, prefix="pfr", drop_first=True).astype(float),
                 pd.get_dummies(df.QuestionSequence, prefix="q", drop_first=True).astype(float),
                 df[["after", "school_hours", "wkend"]].astype(float)]
        if with_student_const:
            parts += [df[["sess_log"]], pd.get_dummies(df.yg, prefix="yg", drop_first=True).astype(float)]
        return pd.concat(parts, axis=1)
    XB = sm.add_constant(XF(F)); yb = F.y.values; gb = pd.factorize(F.UserId)[0]
    rb = sm.Logit(yb, XB).fit(disp=0, cov_type="cluster", cov_kwds={"groups": gb}, maxiter=100); cb = rb.conf_int()
    orB = pd.DataFrame({"term": rb.params.index, "OR": np.exp(rb.params.values), "lo": np.exp(cb[0].values), "hi": np.exp(cb[1].values), "p": rb.pvalues.values})
    T("failed-retry logit")

    # standardised (g-computation) quit probability for each level of the focal variables, with student-bootstrap CIs
    def gcomp(df, X, params):
        out = {}
        for var, levels, setter in [("latency", LAT_LAB, lambda dd, v: dd.assign(lat_bin=pd.Categorical([v] * len(dd), categories=LAT_LAB))),
                                    ("failures so far", [0, 1, 2, 3], lambda dd, v: dd.assign(prior_fr=v)),
                                    ("position", [1, 2, 3, 4], lambda dd, v: dd.assign(QuestionSequence=v))]:
            for v in levels:
                Xv = sm.add_constant(XF(setter(df, v)), has_constant="add").reindex(columns=X.columns, fill_value=0.0)
                out[(var, str(v))] = float((1 / (1 + np.exp(-(Xv.values @ params)))).mean())
        return out
    point = gcomp(F, XB, rb.params.values)
    rng = np.random.default_rng(SEED); users = F.UserId.unique(); by_user = {u: g.index.values for u, g in F.groupby("UserId")}
    boots = []
    for b in range(120):
        pick = rng.choice(users, len(users)); idx = np.concatenate([by_user[u] for u in pick]); Fb = F.loc[idx].reset_index(drop=True)
        Xb_ = sm.add_constant(XF(Fb)).reindex(columns=XB.columns, fill_value=0.0)
        try:
            pb = sm.GLM(Fb.y.values, Xb_, family=sm.families.Binomial()).fit(maxiter=50).params.values
            boots.append(gcomp(Fb, XB, pb))
        except Exception:
            continue
    marg = pd.DataFrame([{"variable": k[0], "level": k[1], "adj_prob": v, "lo": np.percentile([bb[k] for bb in boots], 2.5), "hi": np.percentile([bb[k] for bb in boots], 97.5)} for k, v in point.items()])
    # raw rates for the same levels
    raw = []
    for var, col in [("latency", "lat_bin"), ("failures so far", "prior_fr"), ("position", "QuestionSequence")]:
        t = F.groupby(col).y.agg(["size", "mean"]).reset_index()
        for _, r in t.iterrows(): raw.append({"variable": var, "level": str(r.iloc[0]), "raw_rate": r["mean"], "events": int(r["size"])})
    marg = marg.merge(pd.DataFrame(raw), on=["variable", "level"], how="left"); marg.to_csv(TAB / "q2_fr_marginal.csv", index=False)
    orB.to_csv(TAB / "q2_fr_or.csv", index=False); T(f"g-computation ({len(boots)} bootstraps)")

    # ---------- Part C: student fixed effects (conditional logit)
    from statsmodels.discrete.conditional_models import ConditionalLogit
    XC = XF(F, with_student_const=False)
    ex = F.groupby("UserId").y.agg(["min", "max"]); informative = ex[(ex["min"] == 0) & (ex["max"] == 1)].index
    msk = F.UserId.isin(informative).values
    cl = ConditionalLogit(F.y.values[msk], XC.values[msk], groups=F.UserId.values[msk]).fit(disp=0, maxiter=200)
    cc = cl.conf_int()
    fe = pd.DataFrame({"term": XC.columns, "OR_within_student": np.exp(cl.params), "lo": np.exp(cc[:, 0]), "hi": np.exp(cc[:, 1]), "p": cl.pvalues})
    fe = fe.merge(orB.rename(columns={"OR": "OR_pooled", "lo": "lo_pooled", "hi": "hi_pooled", "p": "p_pooled"}), on="term", how="left")
    fe.to_csv(TAB / "q2_fr_fe.csv", index=False); T("conditional logit")

    summ = {"steps": int(len(d)), "students": int(d.UserId.nunique()), "failed_retry_steps": int(len(F)),
            "fe_students_informative": int(len(informative)), "fe_events": int(msk.sum())}
    json.dump(summ, open(TAB / "q2_hazard_summary.json", "w"), indent=2)
    print("\nFailed-retry logit ORs (pooled, clustered):"); print(orB.round(3).to_string(index=False))
    print("\nWithin-student vs pooled:"); print(fe.round(3).to_string(index=False))
    print("\nStandardised quit probabilities:"); print(marg.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
