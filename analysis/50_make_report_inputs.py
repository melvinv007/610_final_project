"""Turn results/tables into LaTeX macros (report/auto_numbers.tex) and tables (report/tab_*.tex). Every number in the report comes from here."""
import json
import numpy as np, pandas as pd
from common import TAB, ROOT

REP = ROOT / "report"; REP.mkdir(exist_ok=True)
rd = lambda n: pd.read_csv(TAB / n); js = lambda n: json.load(open(TAB / n))
M = {}
def put(name, val): M[name] = val
def pct(x, d=1): return f"{100 * x:.{d}f}\\%"
def pp(x, d=1): return f"{100 * x:.{d}f}"
def n_(x): return f"{int(round(x)):,}".replace(",", "{,}")
def f_(x, d=2): return f"{x:.{d}f}"
def ci(lo, hi, d=1, mult=100, suf=""):
    fmt = lambda v: (f"{mult * v:.{d}f}").replace("-", "$-$")
    return f"{fmt(lo)} to {fmt(hi)}{suf}" if lo < 0 else f"{fmt(lo)} to {fmt(hi)}{suf}"

ds = js("dataset_summary.json"); q1 = js("q1_summary.json"); q2 = js("q2_summary.json"); cp = js("cp_summary.json"); q3 = js("q3_summary.json"); hz = js("q2_hazard_summary.json")
# ---- data
put("nEventsScope", n_(q2["n_events"])); put("nStudents", n_(q2["n_students"])); put("nExits", n_(q2["n_exits"])); put("nSessions", n_(q3["sessions"]))
put("sessionExitRate", pct(q3["exit_rate"]))
for k, v in ds.items():
    if isinstance(v, (int, float)) and k.isalpha(): put("ds" + k[0].upper() + k[1:], n_(v) if v > 1000 else f_(v, 3))
put("nEventsRaw", n_(ds["raw_events"])); put("nSessionsMulti", n_(ds["within_session_gaps_gt_24h"]))
# ---- Q1
put("qOneFailedRetries", n_(q1["failed_retry_events"])); put("qOneRoute", n_(q1["events_with_checkout"])); put("qOneStudents", n_(q1["students"]))
put("qOneLesson", n_(q1["n_lesson"])); put("qOneSkip", n_(q1["n_skip"])); put("qOneSkipShare", pct(q1["skip_share_of_checkout_route"]))
cat = rd("q1_by_category.csv").set_index("category")
for c, nm in [("skip", "Skip"), ("glance", "Glance"), ("read", "Read"), ("long", "Long")]:
    put("qOneSucc" + nm, pct(cat.loc[c, "success"])); put("qOneSucc" + nm + "CI", ci(cat.loc[c, "lo"], cat.loc[c, "hi"]))
    put("qOneEv" + nm, n_(cat.loc[c, "events"]))
put("qOneMedRead", f_(cat.loc["read", "median_dwell_s"], 0)); put("qOneMedGlance", f_(cat.loc["glance", "median_dwell_s"], 0))
eff = rd("q1_effects.csv")
for key, nm in zip(eff.estimator, ["Raw", "Same", "FE", "Logit", "IPW"]):
    r = eff[eff.estimator == key].iloc[0]; put("qOneEff" + nm, pp(r.est)); put("qOneEff" + nm + "CI", ci(r.lo, r.hi))
put("qOneOR", f_(q1["logit_or_lesson"])); put("qOneORCI", f"{q1['logit_or_ci'][0]:.2f} to {q1['logit_or_ci'][1]:.2f}")
sn = q1["sensitivity"]
put("qOnePlacebo", pp(sn["placebo_session_accuracy_FE"]["est"])); put("qOnePlaceboCI", ci(sn["placebo_session_accuracy_FE"]["lo"], sn["placebo_session_accuracy_FE"]["hi"]))
put("qOnePlaceboRaw", pp(sn["placebo_session_accuracy_raw_diff"]))
for k, nm in [("fe_before_break", "Before"), ("fe_after_break", "After"), ("fe_excl_long_gaps_gt_600s", "NoLong"), ("fe_questions_1_to_4_only", "QFour")]:
    put("qOneFE" + nm, pp(sn[k]["est"])); put("qOneFE" + nm + "CI", ci(sn[k]["lo"], sn[k]["hi"]))
put("qOneRR", f_(sn["risk_ratio_adjusted"])); put("qOneEvalue", f_(sn["e_value_point"])); put("qOneEvalueCI", f_(sn["e_value_ci_limit"]))
bal = rd("q1_balance.csv").set_index("covariate")
put("qOneSMDfail", f_(abs(bal.loc["Earlier failures in session", "smd"]))); put("qOneSMDacc", f_(abs(bal.loc["Accuracy earlier in this session", "smd"])))
put("qOneSkipEarlierFail", f_(bal.loc["Earlier failures in session", "mean_skip"], 1)); put("qOneLessonEarlierFail", f_(bal.loc["Earlier failures in session", "mean_lesson"], 1))
dfe = rd("q1_dose_fe.csv").set_index("category")
for c, nm in [("glance", "Glance"), ("read", "Read"), ("long", "Long")]:
    put("qOneDoseFE" + nm, pp(dfe.loc[c, "est"])); put("qOneDoseFE" + nm + "CI", ci(dfe.loc[c, "lo"], dfe.loc[c, "hi"]))
# ---- Q2
c = q2["contrast"]
put("qTwoRateFR", pct(c["rate_failed_retry"]["est"])); put("qTwoRateFRCI", ci(c["rate_failed_retry"]["lo"], c["rate_failed_retry"]["hi"]))
put("qTwoRateOther", pct(c["rate_other"]["est"], 2)); put("qTwoRatio", f_(c["ratio"]["est"], 1)); put("qTwoRatioCI", f"{c['ratio']['lo']:.1f} to {c['ratio']['hi']:.1f}")
put("qTwoShareFR", pct(q2["share_of_all_exits_after_failed_retry"], 0))
a = q2["after_failed_retry"]; put("qTwoToLesson", pct(a["Lesson"])); put("qTwoToCheckout", pct(a["Check-out"])); put("qTwoToQuit", pct(a["Quit"])); put("qTwoToNext", pct(a["Next question"], 1))
put("qTwoQuitAfterLesson", f"{100 * q2['after_lesson']['quit']:.3f}\\%")
st = rd("q2_state_rates.csv").set_index("state")
put("qTwoStudentRateFR", pct(st.loc["CheckinRetry_fail", "student_rate"], 0)); put("qTwoStudentRateFRCI", ci(st.loc["CheckinRetry_fail", "student_lo"], st.loc["CheckinRetry_fail", "student_hi"], 0))
put("qTwoNStepsFR", n_(st.loc["CheckinRetry_fail", "events"]))
cm = rd("q2_commitment.csv"); r = cm[(cm.state == "Checkout_fail") & (cm.had_lesson == 0)].iloc[0]; put("qTwoCommitNoLesson", pct(r.exit_rate, 0))
r = cm[(cm.state == "Checkout_fail") & (cm.had_lesson == 1)].iloc[0]; put("qTwoCommitLesson", pct(r.exit_rate, 1))
for k, nm in [("contrast_before_break", "Before"), ("contrast_after_break", "After")]:
    put("qTwoRate" + nm, pct(q2[k]["rate_failed_retry"]["est"])); put("qTwoRatio" + nm, f_(q2[k]["ratio"]["est"], 1))
icc = rd("q2_icc.csv").set_index("level")
for lv, nm in [("student", "Student"), ("topic (subject)", "Topic"), ("concept (construct)", "Concept")]:
    put("qTwoICC" + nm, f_(icc.loc[lv, "icc"], 3)); put("qTwoICC" + nm + "CI", f"{icc.loc[lv, 'lo']:.3f} to {icc.loc[lv, 'hi']:.3f}")
sens = rd("q2_sensitivity.csv")
mod = rd("q2_hazard_models.csv")
for i, nm in enumerate(["A", "B", "C", "D", "E"]): put("hzAUC" + nm, f_(mod.cv_auc[i], 3)); put("hzR" + nm, f_(mod.mcfadden_r2[i], 3))
mg = rd("q2_fr_marginal.csv")
for _, r in mg.iterrows():
    key = {"latency": "Lat", "failures so far": "Fail", "position": "Pos"}[r.variable]; lv = str(r.level).replace("<=3 s", "A").replace("3-6 s", "B").replace("6-12 s", "C").replace("12-25 s", "D").replace("25-60 s", "E").replace(">60 s", "F")
    put(f"hzMarg{key}{ {'0':'Zero','1':'One','2':'Two','3':'Three','4':'Four'}.get(lv, lv)}", pct(r.adj_prob))
fe = rd("q2_fr_fe.csv").set_index("term")
for t, nm in [("pfr_1", "FailOne"), ("pfr_2", "FailTwo"), ("q_4", "PosFour"), ("after", "After")]:
    put("hzFE" + nm, f_(fe.loc[t, "OR_within_student"])); put("hzFE" + nm + "CI", f"{fe.loc[t, 'lo']:.2f} to {fe.loc[t, 'hi']:.2f}"); put("hzPool" + nm, f_(fe.loc[t, "OR_pooled"]))
put("hzFEstudents", n_(hz["fe_students_informative"]))
# ---- break
put("cpDate", pd.Timestamp(cp["break_date"]).strftime("%-d %B")); put("cpLo", pd.Timestamp(cp["break_ci_days"][0] if False else cp["failed_retry"]["break_ci_days"][0]).strftime("%-d %B")); put("cpHi", pd.Timestamp(cp["failed_retry"]["break_ci_days"][1]).strftime("%-d %B"))
put("cpBefore", pct(cp["failed_retry"]["rate_before"])); put("cpAfter", pct(cp["failed_retry"]["rate_after"])); put("cpLR", f"{cp['failed_retry']['lr_stat']:.0f}")
pr = cp["paired_students"]; put("cpPairN", n_(pr["n_students"])); put("cpPairDiff", pp(pr["mean_diff"])); put("cpPairCI", ci(*pr["diff_ci"]))
put("cpOtherBefore", pct(cp["other_answers"]["rate_before"], 2)); put("cpOtherAfter", pct(cp["other_answers"]["rate_after"], 2))
# ---- Q3
ps = rd("q3_persistence.csv").set_index("gap")
a_ = ps.loc["all"]; put("qThreeAfterExit", pct(a_.rate_after_exit)); put("qThreeAfterComplete", pct(a_.rate_after_complete)); put("qThreeRatio", f_(a_.ratio, 1)); put("qThreeRatioCI", f"{a_.ratio_lo:.1f} to {a_.ratio_hi:.1f}")
put("qThreeDiff", pp(a_["diff"])); put("qThreeDiffCI", ci(a_.diff_lo, a_.diff_hi)); put("qThreePairs", n_(a_.n_pairs)); put("qThreeAfterQuits", n_(a_.n_after_exit))
put("qThreeSame", pct(ps.loc["<1 h (same sitting)", "rate_after_exit"])); put("qThreeWeek", f_(ps.loc[">7 d", "ratio"], 1))
ts = rd("q3_trait_state.csv")
for i, nm in enumerate(["Raw", "Loo", "Within"]): put("qThree" + nm, pp(ts.effect_of_previous_exit[i])); put("qThree" + nm + "CI", ci(ts.lo[i], ts.hi[i]))
put("qThreeSB", f_(q3["spearman_brown"])); put("qThreeSplitN", n_(q3["splithalf_students"])); put("qThreeNever", pct(q3["share_never_exit_ge5"], 0)); put("qThreeTop", pct(q3["share_of_exits_by_top10pct"], 0)); put("qThreeGeFive", n_(q3["students_ge5"]))
put("qThreeCensored", pct(q3["last_session_censored_share"], 0))
rp = rd("q3_return_probs.csv"); rs = rd("q3_return_summary.csv").set_index("kind")
for k, nm in [("exit after failed retry", "FR"), ("other exit", "Oth"), ("completed", "Done")]:
    for h, hn in [(1, "H"), (24, "D"), (168, "W"), (720, "M")]:
        r = rp[(rp.kind == k) & (rp.within_hours == h)].iloc[0]; put(f"qThreeRet{nm}{hn}", pct(r.p_return, 0 if False else 1)); put(f"qThreeRet{nm}{hn}CI", ci(r.lo, r.hi))
    put("qThreeMed" + nm, f"{rs.loc[k, 'median_gap_h'] * 60:.0f}~min" if rs.loc[k, "median_gap_h"] < 1.5 else f"{rs.loc[k, 'median_gap_h']:.0f}~h")
    put("qThreeSameQuiz" + nm, pct(rs.loc[k, "same_quiz"], 1)); put("qThreeNQuit" + nm, n_(rp[(rp.kind == k)].n_sessions.iloc[0]))
ro = rd("q3_return_or.csv")
for hz_, hn in [(24, "D"), (168, "W")]:
    for t, nm in [("exit after failed retry", "FR"), ("other exit", "Oth"), ("after break", "Brk")]:
        r = ro[(ro.horizon_h == hz_) & (ro.term == t)].iloc[0]; put(f"qThreeOR{nm}{hn}", f_(r.OR)); put(f"qThreeOR{nm}{hn}CI", f"{r.lo:.2f} to {r.hi:.2f}")

# ---- extra figures
w = rd("q2_where.csv")
def wv(var, stp, lv): return w[(w.variable == var) & (w.step == stp) & (w.level == lv)].iloc[0]
put("whereFROne", pct(wv("position", "after a failed retry", "1").rate)); put("whereFRFour", pct(wv("position", "after a failed retry", "4").rate))
put("whereOthOne", pct(wv("position", "any other step", "1").rate)); put("whereOthFour", pct(wv("position", "any other step", "4").rate))
put("whereTimeFirst", pct(wv("time in session", "after a failed retry", "<30 s").rate)); put("whereTimeLast", pct(wv("time in session", "after a failed retry", ">8 min").rate))
put("whereOthTimeMid", pct(wv("time in session", "any other step", "1-2").rate)); put("whereOthTimeLast", pct(wv("time in session", "any other step", ">8 min").rate))
put("whereHourPeak", pct(wv("hour of day", "all", "13-14").rate)); put("whereHourDip", pct(wv("hour of day", "all", "15-16").rate))
put("whereHourPeakBefore", pct(wv("hour of day (before break)", "all", "13-14").rate)); put("whereHourDipBefore", pct(wv("hour of day (before break)", "all", "15-16").rate))
rch = rd("q2_reach.csv")
for i, nm in enumerate(["One", "Two", "Three", "Four", "Five"]): put("reach" + nm, pct(rch.share_sessions_reaching[i]))
cc = js("q3_concentration.json"); put("qThreeGini", f_(cc["gini"])); put("qThreeMedRate", pct(cc["median_student_rate"])); put("qThreeHalfShare", pct(cc["share_ge_half"]))
sk = rd("q3_streak.csv")
for i, nm in enumerate(["Zero", "One", "Two", "Three"]): put("streak" + nm, pct(sk.rate[i], 0)); put("streak" + nm + "CI", ci(sk.lo[i], sk.hi[i], 0))
mon = rd("data_monthly.csv"); put("monthSepStudents", n_(mon.students.iloc[-1])); put("monthAugStudents", n_(mon.students.iloc[-2]))
# ---------------- write macros
with open(REP / "auto_numbers.tex", "w", encoding="utf-8") as f:
    f.write("% AUTO-GENERATED by analysis/50_make_report_inputs.py - do not edit by hand\n")
    for k, v in M.items():
        assert k.isalpha(), k
        f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---------------- tables
def esc(s): return s.replace("%", "\\%").replace("&", "\\&")
rows = []
for _, r in mod.iterrows():
    rows.append(f"{esc(r.model)} & {int(r.parameters)} & {r.mcfadden_r2:.3f} & {r.cv_auc:.3f} & {r.cv_logloss:.4f} \\\\")
open(REP / "tab_hazard.tex", "w").write("\\begin{tabular}{lrrrr}\n\\toprule\nHazard model & Par. & McFadden $R^2$ & CV AUC & CV log-loss \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

lab = {"exit after failed retry": "Quit after failed retry", "other exit": "Other quit", "completed": "Completed"}
rows = []
for k in lab:
    cells = []
    for h in (1, 24, 168, 720):
        r = rp[(rp.kind == k) & (rp.within_hours == h)].iloc[0]; cells.append(f"{100 * r.p_return:.1f} ({100 * r.lo:.1f} to {100 * r.hi:.1f})")
    rows.append(f"{lab[k]} & {int(rp[rp.kind == k].n_sessions.iloc[0]):,} & " + " & ".join(cells) + " \\\\")
open(REP / "tab_return.tex", "w").write("\\begin{tabular}{lrcccc}\n\\toprule\nPrevious session & $n$ & 1 h & 24 h & 7 d & 30 d \\\\\n\\midrule\n" + "\n".join(rows).replace(",", "{,}") + "\n\\bottomrule\n\\end{tabular}\n")

rows = []
short = ["Primary (session ended before Q5)", "+ any pause over 1 h counts as exit", "Exit and no return within 24 h", "Drop sessions in last 24 h of data", "Exclude first session of each student", "School-hours sessions only", "Years 5 to 11 only"]
for s_, (_, r) in zip(short, sens.iterrows()):
    rows.append(f"{s_} & {int(r.exits):,} & {100 * r.rate_failed_retry:.1f} & {100 * r.rate_other:.2f} & {r.ratio:.1f} ({r.ratio_lo:.1f} to {r.ratio_hi:.1f}) \\\\")
open(REP / "tab_sens.tex", "w").write("\\begin{tabular}{lrrrr}\n\\toprule\nExit definition / sample & Exits & After failed retry (\\%) & Elsewhere (\\%) & Ratio (95\\% CI) \\\\\n\\midrule\n" + "\n".join(rows).replace(",", "{,}") + "\n\\bottomrule\n\\end{tabular}\n")
print(len(M), "macros;", "tables written to", REP)
