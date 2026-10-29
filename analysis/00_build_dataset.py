"""Step 0 - build the analysis table.
Reads the raw Eedi CausalEdu files, orders events inside each quiz session by timestamp,
flags the exit event (last event of the session), adds covariates and writes data_work/events.pkl."""
import json
import numpy as np
import pandas as pd
from common import DATA, WORK, TAB

ev = pd.read_csv(DATA / "checkins_lessons_checkouts_training.csv", parse_dates=["Timestamp"])
n_raw = len(ev)
ev = ev.sort_values(["QuizSessionId", "Timestamp"]).reset_index(drop=True)
g = ev.groupby("QuizSessionId")

# --- exit flag: an event is an exit if it is the last event of its quiz session --------------
ev["is_last"] = (~ev.duplicated("QuizSessionId", keep="last")).astype(int)
ev["max_qs"] = g.QuestionSequence.transform("max")
# PRIMARY exit: final event of a session that never reached question 5 (a student who finished the five
# check-ins and later revisits old questions has not quit).
ev["is_exit"] = ((ev.is_last == 1) & (ev.max_qs < 5)).astype(int)
ev["next_gap_s"] = (g.Timestamp.shift(-1) - ev.Timestamp).dt.total_seconds()
# SENSITIVITY exit (pause): also count an event as an exit if the session is not touched again for >1 hour
ev["is_exit_pause"] = ((ev.is_exit == 1) | (ev.next_gap_s > 3600)).astype(int)
ev["event_idx"] = g.cumcount()
ev["t_start"] = g.Timestamp.transform("min")
ev["t_last"] = g.Timestamp.transform("max")
ev["latency_s"] = g.Timestamp.diff().dt.total_seconds()          # time since previous event in session
ev["since_start_s"] = (ev.Timestamp - ev.t_start).dt.total_seconds()
ev["fail"] = (ev.IsCorrect == 0).astype(int)
ev["n_prior_fail"] = g["fail"].cumsum() - ev["fail"]               # failed answers earlier in the session

res = ev.IsCorrect.map({1.0: "ok", 0.0: "fail"})
ev["state"] = np.where(ev.Type == "Lesson", "Lesson", ev.Type + "_" + res.fillna(""))

# --- session-level features (position in the student's history, strict-exit sensitivity) -------
sess = g.agg(UserId=("UserId", "first"), t_start=("Timestamp", "min"), t_last=("Timestamp", "max"),
             max_qs=("QuestionSequence", "max"), n_events=("Timestamp", "size")).reset_index()
sess = sess.sort_values(["UserId", "t_start"]).reset_index(drop=True)
sess["user_session_idx"] = sess.groupby("UserId").cumcount() + 1
sess["next_start"] = sess.groupby("UserId").t_start.shift(-1)
data_end = ev.Timestamp.max()
# 'returned_24h' = the same student starts another session within 24h after this session's last event
sess["returned_24h"] = ((sess.next_start - sess.t_last) <= pd.Timedelta("24h")).astype(int)
# sessions ending in the last 24h of the data cannot show a return -> right-censoring risk
sess["censor_risk"] = ((data_end - sess.t_last) < pd.Timedelta("24h")).astype(int)
ev = ev.merge(sess[["QuizSessionId", "user_session_idx", "returned_24h", "censor_risk"]], on="QuizSessionId")
# SENSITIVITY exit (no return): exit AND the student did not start another session within 24h
ev["is_exit_noreturn"] = ((ev.is_exit == 1) & (ev.returned_24h == 0)).astype(int)

# --- student metadata ---------------------------------------------------------------------------
st = pd.read_csv(DATA / "student_metadata.csv")
ev = ev.merge(st[["UserId", "Gender", "IsPupilPremium", "YearGroup"]], on="UserId", how="left")

# --- construct -> subject (topic) hierarchy -------------------------------------------------------
tp = pd.read_csv(DATA / "topic_pathway_metadata.csv")
mode = lambda s: s.dropna().mode().iat[0] if s.notna().any() else np.nan
cmap = tp.groupby("ConstructId").agg(SubjectId=("SubjectId", mode), SubjectName=("SubjectName", mode),
                                     PathYear=("YearGroup", mode)).reset_index()
sm = pd.read_csv(DATA / "subject_metadata.csv")
name = dict(zip(sm.SubjectId, sm.Name)); parent = dict(zip(sm.SubjectId, sm.ParentId)); level = dict(zip(sm.SubjectId, sm.Level))

def ancestor(sid, lvl):
    while sid == sid and sid is not None:
        sid = int(sid)
        if level.get(sid) == lvl:
            return sid
        sid = parent.get(sid, np.nan)
    return np.nan

cmap["StrandId"] = cmap.SubjectId.apply(lambda s: ancestor(s, 1))
cmap["Strand"] = cmap.StrandId.map(name)
ev = ev.merge(cmap[["ConstructId", "SubjectId", "SubjectName", "PathYear", "Strand"]], on="ConstructId", how="left")

ev["in_scope"] = (ev.QuestionSequence <= 4).astype(int)            # Q5 excluded: quiz ends there naturally
ev["hour"] = ev.Timestamp.dt.hour
ev["weekday"] = ev.Timestamp.dt.dayofweek
ev.to_pickle(WORK / "events.pkl")

sc = ev[ev.in_scope == 1]
summary = {
    "raw_events": int(n_raw), "sessions": int(ev.QuizSessionId.nunique()), "students": int(ev.UserId.nunique()),
    "date_min": str(ev.Timestamp.min()), "date_max": str(ev.Timestamp.max()),
    "in_scope_events": int(len(sc)), "in_scope_students": int(sc.UserId.nunique()),
    "in_scope_exits": int(sc.is_exit.sum()), "in_scope_exits_pause": int(sc.is_exit_pause.sum()),
    "in_scope_exits_noreturn": int(sc.is_exit_noreturn.sum()), "sessions_not_reaching_Q5": int((sess.max_qs < 5).sum()),
    "students_with_year_group": int(ev.drop_duplicates("UserId").YearGroup.notna().sum()),
    "constructs_mapped_to_subject": float(ev.drop_duplicates("ConstructId").SubjectName.notna().mean()),
    "within_session_gaps_gt_24h": int((ev.latency_s > 86400).sum()),
}
json.dump(summary, open(TAB / "dataset_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))
print(sc.groupby("state").is_exit.agg(["size", "sum", "mean"]))
print(ev.Strand.value_counts(dropna=False).head(12))
