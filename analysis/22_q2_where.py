"""RQ2 (descriptive 'where' tables for Figures): exit rate by position, time in session and hour of day, plus how sessions thin out.
Student-cluster bootstrap CIs (seed 610). Writes results/tables/q2_where.csv, q2_reach.csv, data_monthly.csv"""
import importlib.util, numpy as np, pandas as pd
from common import TAB, SEED, load_events

spec = importlib.util.spec_from_file_location("hz", __file__.replace("22_q2_where.py", "21_q2_hazard.py")); hz = importlib.util.module_from_spec(spec); spec.loader.exec_module(hz)
d = hz.prepare()
uid, ucode = pd.factorize(d.UserId); d["u"] = uid; nU = len(ucode)
d["is_fr"] = d.state == "CheckinRetry_fail"
d["mins"] = d.since_start_s / 60
d["time_bin"] = pd.cut(d.mins, [-1, 0.5, 1, 2, 4, 8, 1e9], labels=["<30 s", "0.5-1", "1-2", "2-4", "4-8", ">8 min"])
d["hour_grp"] = pd.cut(d.hour, [-1, 6, 8, 10, 12, 14, 16, 18, 20, 23], labels=["0-6", "7-8", "9-10", "11-12", "13-14", "15-16", "17-18", "19-20", "21-23"])
d["step"] = np.where(d.is_fr, "after a failed retry", "any other step")
rng = np.random.default_rng(SEED); W = rng.multinomial(nU, np.ones(nU) / nU, size=400).astype(float)
rows = []
def run(var, col, sub=None, by_step=True):
    dd = d if sub is None else d[sub]
    for stp, g in (dd.groupby("step") if by_step else [("all", dd)]):
        for lv, h in g.groupby(col, observed=True):
            k = np.bincount(h.u, weights=h.y, minlength=nU); n = np.bincount(h.u, minlength=nU).astype(float)
            r = k.sum() / n.sum(); bs = (W @ k) / np.maximum(W @ n, 1)
            rows.append({"variable": var, "step": stp, "level": str(lv), "n": int(n.sum()), "exits": int(k.sum()), "rate": r, "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
run("position", "QuestionSequence"); run("time in session", "time_bin"); run("hour of day", "hour_grp", sub=d.is_fr & (d.period == "after"), by_step=False)
run("hour of day (before break)", "hour_grp", sub=d.is_fr & (d.period == "before"), by_step=False)
pd.DataFrame(rows).to_csv(TAB / "q2_where.csv", index=False); print(pd.DataFrame(rows).round(3).to_string(index=False))

ev = load_events()
s = ev.groupby("QuizSessionId").agg(maxq=("QuestionSequence", "max"), start=("Timestamp", "min"), u=("UserId", "first"))
reach = pd.DataFrame({"question": [1, 2, 3, 4, 5]}); reach["share_sessions_reaching"] = [(s.maxq >= k).mean() for k in reach.question]; reach["sessions"] = [int((s.maxq >= k).sum()) for k in reach.question]
reach.to_csv(TAB / "q2_reach.csv", index=False); print(reach)
s["month"] = s.start.dt.strftime("%Y-%m"); m = s.groupby("month").agg(sessions=("u", "size"), students=("u", "nunique")).reset_index(); m.to_csv(TAB / "data_monthly.csv", index=False); print(m)
