"""RQ1 - When do students quit?  Exit rate after each event type, and the failed-retry contrast."""
import numpy as np, pandas as pd, json
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import WORK, TAB, FIG, SEED, wilson, cluster_bootstrap, pct_ci

ev = pd.read_pickle(WORK / "events.pkl")
sc = ev[ev.in_scope == 1].copy()
STATES = ["Checkin_ok", "Checkin_fail", "CheckinRetry_ok", "CheckinRetry_fail", "Lesson",
          "Checkout_ok", "Checkout_fail", "CheckoutRetry_ok", "CheckoutRetry_fail"]
LABEL = {"Checkin_ok": "Check-in, correct", "Checkin_fail": "Check-in, wrong",
         "CheckinRetry_ok": "Check-in retry, correct", "CheckinRetry_fail": "Check-in retry, wrong",
         "Lesson": "Lesson", "Checkout_ok": "Check-out, correct", "Checkout_fail": "Check-out, wrong",
         "CheckoutRetry_ok": "Check-out retry, correct", "CheckoutRetry_fail": "Check-out retry, wrong"}

users = np.sort(sc.UserId.unique()); uidx = {u: i for i, u in enumerate(users)}
sc["u"] = sc.UserId.map(uidx)
M = len(users)

def per_user(df, col="is_exit"):
    """K[u,s] exits and N[u,s] events of state s for student u."""
    K = np.zeros((M, len(STATES))); N = np.zeros((M, len(STATES)))
    g = df.groupby(["u", "state"]).agg(k=(col, "sum"), n=(col, "size")).reset_index()
    for s_i, s in enumerate(STATES):
        gg = g[g.state == s]
        K[gg.u.values, s_i] = gg.k.values; N[gg.u.values, s_i] = gg.n.values
    return K, N

def rates(K, N, B=2000):
    ew = K.sum(0) / N.sum(0)
    R = np.where(N > 0, K / np.where(N > 0, N, 1), np.nan)
    sw = np.nanmean(R, 0)
    def stat(idx):
        e = K[idx].sum(1) / np.maximum(N[idx].sum(1), 1)
        s = np.nanmean(np.where(N[idx] > 0, K[idx] / np.where(N[idx] > 0, N[idx], 1), np.nan), 1)
        return np.stack([e, s], 1)
    bs = cluster_bootstrap(stat, M, B=B, chunk=100)
    ew_lo, ew_hi = pct_ci(bs[:, 0, :]); sw_lo, sw_hi = pct_ci(bs[:, 1, :])
    return ew, ew_lo, ew_hi, sw, sw_lo, sw_hi, (N > 0).sum(0)

# ---------------------------------------------------------------- 1. exit rate by event type
K, N = per_user(sc)
ew, ew_lo, ew_hi, sw, sw_lo, sw_hi, nstud = rates(K, N)
t1 = pd.DataFrame({"state": STATES, "label": [LABEL[s] for s in STATES], "events": N.sum(0).astype(int),
                   "exits": K.sum(0).astype(int), "students": nstud.astype(int),
                   "exit_rate_event_weighted": ew, "ew_ci_lo": ew_lo, "ew_ci_hi": ew_hi,
                   "exit_rate_student_weighted": sw, "sw_ci_lo": sw_lo, "sw_ci_hi": sw_hi})
t1.to_csv(TAB / "rq1_state_rates.csv", index=False)
print(t1.round(4).to_string())

# ---------------------------------------------------------------- 2. contrast: failed check-in retry vs other answers
iT = STATES.index("CheckinRetry_fail"); iL = STATES.index("Lesson")
iO = [i for i in range(len(STATES)) if i not in (iT, iL)]          # all other answered attempts
kT, nT = K[:, iT], N[:, iT]; kO, nO = K[:, iO].sum(1), N[:, iO].sum(1)
def contrast(kT, nT, kO, nO, idx=None):
    if idx is None: idx = np.arange(len(kT))[None, :]
    a = kT[idx].sum(1) / nT[idx].sum(1); b = kO[idx].sum(1) / nO[idx].sum(1)
    return a, b
a, b = contrast(kT, nT, kO, nO)
bsa, bsb = [], []
def stat_c(idx):
    a_, b_ = contrast(kT, nT, kO, nO, idx); return np.stack([a_, b_, a_ - b_, a_ / b_], 1)
bs = cluster_bootstrap(stat_c, M, B=2000, chunk=200)
rd_lo, rd_hi = pct_ci(bs[:, 2]); rr_lo, rr_hi = pct_ci(bs[:, 3])
# paired, student-weighted: students having both event kinds
both = (nT > 0) & (nO > 0)
dT = kT[both] / nT[both]; dO = kO[both] / nO[both]
diff = dT - dO
w = stats.wilcoxon(diff)
mb = np.flatnonzero(both)
def stat_p(idx):                                                  # idx indexes into 'both' students
    return (diff[idx].mean(1))[:, None]
bs_p = cluster_bootstrap(stat_p, both.sum(), B=2000, chunk=200)[:, 0]
con = {"rate_failed_checkin_retry_event_weighted": a[0], "rate_other_answers_event_weighted": b[0],
       "risk_difference": a[0] - b[0], "rd_ci": [rd_lo, rd_hi], "risk_ratio": a[0] / b[0], "rr_ci": [rr_lo, rr_hi],
       "paired_students": int(both.sum()), "paired_mean_rate_failed_retry": float(dT.mean()),
       "paired_mean_rate_other": float(dO.mean()), "paired_mean_diff": float(diff.mean()),
       "paired_diff_ci": list(pct_ci(bs_p)), "wilcoxon_stat": float(w.statistic), "wilcoxon_p": float(w.pvalue),
       "share_of_all_exits_after_failed_checkin_retry": float(kT.sum() / K.sum()),
       "share_of_answer_events_that_are_failed_checkin_retry": float(nT.sum() / N[:, [i for i in range(9) if i != iL]].sum())}
json.dump(con, open(TAB / "rq1_contrast.json", "w"), indent=2, default=float)
print(json.dumps(con, indent=2, default=float))

# ---------------------------------------------------------------- 3. logistic regression, student-clustered SEs
reg = sc[sc.state != "Lesson"].copy()                              # Lesson has ~0 exits (separation) -> excluded
reg["state"] = pd.Categorical(reg.state, categories=[s for s in STATES if s != "Lesson"])
m = smf.logit("is_exit ~ C(state, Treatment('Checkin_ok'))", data=reg).fit(
    disp=0, cov_type="cluster", cov_kwds={"groups": reg.UserId.values})
ci = m.conf_int()
orr = pd.DataFrame({"term": [t.split("[T.")[1].rstrip("]") for t in m.params.index[1:]],
                    "OR": np.exp(m.params.values[1:]), "lo": np.exp(ci.values[1:, 0]), "hi": np.exp(ci.values[1:, 1]),
                    "p": m.pvalues.values[1:]})
orr.to_csv(TAB / "rq1_logit_or.csv", index=False); print(orr.round(4).to_string())

# ---------------------------------------------------------------- 4. sensitivity to the definition of "exit"
rows = []
variants = {"primary (last event, never reached Q5)": (sc, "is_exit"),
            "pause (also any >1h pause in the session)": (sc, "is_exit_pause"),
            "no return (no new session within 24h)": (sc, "is_exit_noreturn"),
            "primary, drop sessions ending in last 24h of data": (sc[sc.censor_risk == 0], "is_exit")}
for name, (d, col) in variants.items():
    d = d.copy(); Kv = np.zeros((M, 9)); Nv = np.zeros((M, 9))
    g = d.groupby(["u", "state"]).agg(k=(col, "sum"), n=(col, "size")).reset_index()
    for s_i, s in enumerate(STATES):
        gg = g[g.state == s]; Kv[gg.u.values, s_i] = gg.k.values; Nv[gg.u.values, s_i] = gg.n.values
    kT_, nT_ = Kv[:, iT], Nv[:, iT]; kO_, nO_ = Kv[:, iO].sum(1), Nv[:, iO].sum(1)
    def st(idx):
        a_ = kT_[idx].sum(1) / nT_[idx].sum(1); b_ = kO_[idx].sum(1) / nO_[idx].sum(1); return np.stack([a_ / b_], 1)
    bsv = cluster_bootstrap(st, M, B=2000, chunk=200)[:, 0]
    lo, hi = pct_ci(bsv)
    rows.append({"definition": name, "exits_total": int(Kv.sum()), "rate_failed_retry": kT_.sum() / nT_.sum(),
                 "rate_other_answers": kO_.sum() / nO_.sum(), "risk_ratio": (kT_.sum() / nT_.sum()) / (kO_.sum() / nO_.sum()),
                 "rr_lo": lo, "rr_hi": hi})
sens = pd.DataFrame(rows); sens.to_csv(TAB / "rq1_sensitivity.csv", index=False); print(sens.round(4).to_string())

# ---------------------------------------------------------------- 5. by question position (descriptive, Wilson CI)
rows = []
for q in [1, 2, 3, 4]:
    d = sc[sc.QuestionSequence == q]
    for label, mask in [("failed check-in retry", d.state == "CheckinRetry_fail"),
                        ("other answers", (d.state != "CheckinRetry_fail") & (d.state != "Lesson"))]:
        k, n = d[mask].is_exit.sum(), mask.sum(); lo, hi = wilson(k, n)
        rows.append({"question": q, "group": label, "events": int(n), "exits": int(k), "rate": k / n, "lo": float(lo), "hi": float(hi)})
pos = pd.DataFrame(rows); pos.to_csv(TAB / "rq1_by_position.csv", index=False); print(pos.round(4).to_string())
# (figures are drawn by 05_make_figures.py from the saved tables)
