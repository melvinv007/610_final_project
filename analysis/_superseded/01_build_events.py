"""Step 1 - build one clean, analysis-ready event table.

Reads  : checkins_lessons_checkouts_training.csv, student_metadata.csv,
         topic_pathway_metadata.csv, subject_metadata.csv
Writes : work/events.pkl            (one row per quiz event, with derived columns)
         results/01_data_qa.txt     (counts to quote in the report's data section)

Key derived columns
  is_last        True if the event is the final one recorded in its quiz session
  in_scope       True if QuestionSequence <= 4 (question 5 is excluded: quizzes end there)
  exit           1 if in_scope and is_last, else 0  -> "the student quit after this event"
  state          event type + outcome, e.g. CheckinRetry_fail, Checkout_ok, Lesson
  prior_fails    number of wrong answers earlier in the same session
  user_session_no  1 = the student's first ever session in the data
  days_to_end    days between this event and the last timestamp in the whole dataset
  Strand         top-level maths strand of the construct (Number, Algebra, ...)
"""
import numpy as np
import pandas as pd
from common import DATA, EVENTS, RESULTS, SCOPE_MAX_QS


def build_subject_map():
    sm = pd.read_csv(DATA / "subject_metadata.csv")
    parent = {int(a): (int(b) if pd.notna(b) else None) for a, b in zip(sm.SubjectId, sm.ParentId)}
    level = dict(zip(sm.SubjectId.astype(int), sm.Level))
    name = dict(zip(sm.SubjectId.astype(int), sm.Name))

    def strand(sid):
        while sid is not None and level.get(sid, 0) > 1:
            sid = parent.get(sid)
        return name.get(sid) if sid is not None and level.get(sid) == 1 else None

    tp = pd.read_csv(DATA / "topic_pathway_metadata.csv").dropna(subset=["SubjectId"])
    c2s = tp.groupby("ConstructId").agg(SubjectId=("SubjectId", "first"),
                                        Subject=("SubjectName", "first")).reset_index()
    c2s["SubjectId"] = c2s.SubjectId.astype(int)
    c2s["Strand"] = c2s.SubjectId.map(strand)
    return c2s


def main():
    ev = pd.read_csv(DATA / "checkins_lessons_checkouts_training.csv", parse_dates=["Timestamp"])
    n_raw = len(ev)
    ev = ev.sort_values(["QuizSessionId", "Timestamp"], kind="mergesort").reset_index(drop=True)
    assert not ev.duplicated(["QuizSessionId", "Timestamp"]).any(), "timestamp ties inside a session"
    assert ev.IsCorrect.isna().sum() == (ev.Type == "Lesson").sum(), "unexpected missing IsCorrect"

    g = ev.groupby("QuizSessionId", sort=False)
    ev["event_idx"] = g.cumcount()
    ev["session_len"] = g["Timestamp"].transform("size")
    ev["is_last"] = ev.event_idx == ev.session_len - 1
    ev["gap_prev_s"] = g["Timestamp"].diff().dt.total_seconds()
    ev["in_scope"] = ev.QuestionSequence <= SCOPE_MAX_QS
    ev["exit"] = (ev.in_scope & ev.is_last).astype(int)

    outcome = ev.IsCorrect.map({1.0: "ok", 0.0: "fail"})
    ev["state"] = np.where(ev.Type == "Lesson", "Lesson", ev.Type + "_" + outcome.fillna("na"))
    ev["fail"] = (ev.IsCorrect == 0).astype(int)
    ev["prior_fails"] = ev.groupby("QuizSessionId", sort=False)["fail"].cumsum() - ev["fail"]

    # each student's n-th session
    ss = ev.groupby("QuizSessionId").agg(UserId=("UserId", "first"), start=("Timestamp", "min"))
    ss["user_session_no"] = ss.groupby("UserId").start.rank(method="first").astype(int)
    ev = ev.merge(ss[["user_session_no"]], left_on="QuizSessionId", right_index=True, how="left")
    ev["days_to_end"] = (ev.Timestamp.max() - ev.Timestamp).dt.total_seconds() / 86400

    # student attributes
    st = pd.read_csv(DATA / "student_metadata.csv")[["UserId", "Gender", "IsPupilPremium", "YearGroup"]]
    ev = ev.merge(st, on="UserId", how="left")
    ev["YearGroup"] = ev.YearGroup.astype("Int64")

    # topic attributes
    ev = ev.merge(build_subject_map(), on="ConstructId", how="left")

    ev.to_pickle(EVENTS)

    # ---- QA report -------------------------------------------------------------
    s = ev[ev.in_scope]
    lines = [
        f"raw rows                         : {n_raw:,}",
        f"students                         : {ev.UserId.nunique():,}",
        f"quiz sessions                    : {ev.QuizSessionId.nunique():,}",
        f"constructs                       : {ev.ConstructId.nunique():,}",
        f"date range                       : {ev.Timestamp.min():%Y-%m-%d} to {ev.Timestamp.max():%Y-%m-%d}",
        f"students with a YearGroup        : {ev.drop_duplicates('UserId').YearGroup.notna().sum():,}",
        f"constructs with a subject label  : {ev.drop_duplicates('ConstructId').Subject.notna().sum():,} of {ev.ConstructId.nunique():,}",
        "",
        f"events in scope (question 1-{SCOPE_MAX_QS})   : {len(s):,}",
        f"sessions with an exit in scope   : {int(s.exit.sum()):,} of {ev.QuizSessionId.nunique():,} "
        f"({s.exit.sum() / ev.QuizSessionId.nunique():.1%})",
        f"gaps > 1 day inside a session    : {(ev.gap_prev_s > 86400).sum():,}  (sessions that were resumed)",
        "",
        "events and exit rate by state (in scope):",
        s.groupby("state").exit.agg(events="size", exits="sum", rate="mean").round(4).to_string(),
    ]
    txt = "\n".join(lines)
    (RESULTS / "01_data_qa.txt").write_text(txt, encoding="utf-8")
    print(txt)
    print(f"\nSaved {EVENTS}")


if __name__ == "__main__":
    main()
