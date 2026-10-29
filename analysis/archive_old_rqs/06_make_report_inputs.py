"""Step 6 - turn saved results into LaTeX macros (report/auto_numbers.tex) and tables (report/tab_*.tex),
so that every number in the report text comes straight from the analysis outputs."""
import json
import numpy as np, pandas as pd
from common import ROOT, TAB

REP = ROOT / "report"; REP.mkdir(exist_ok=True)
J = lambda f: json.load(open(TAB / f))
ds, c1, r2, r3 = J("dataset_summary.json"), J("rq1_contrast.json"), J("rq2_summary.json"), J("rq3_summary.json")
st = pd.read_csv(TAB / "rq1_state_rates.csv").set_index("state"); orr = pd.read_csv(TAB / "rq1_logit_or.csv").set_index("term")
sens = pd.read_csv(TAB / "rq1_sensitivity.csv"); pos = pd.read_csv(TAB / "rq1_by_position.csv")
full = pd.read_csv(TAB / "rq2_topic_full.csv", index_col=0); hb = pd.read_csv(TAB / "rq3_hotspots_by_band.csv", index_col=0)
yr = pd.read_csv(TAB / "rq3_by_year.csv", index_col=0); mcv = pd.read_csv(TAB / "model_cv.csv")

num = lambda n: f"{int(round(n)):,}".replace(",", "{,}")
pc = lambda x, d=1: f"{100 * x:.{d}f}"
f2 = lambda x, d=2: (f"{x:.{d}f}".replace("-", "$-$"))
def f2z(x, d=2): s = f"{x:.{d}f}"; return s.replace("0.", ".", 1) if s.startswith("0.") else s.replace("-0.", "$-$.", 1) if s.startswith("-0.") else s
def pstr(p): return "$p<.001$" if p < .001 else f"$p={p:.3f}$".replace("=0.", "=.")
M = {}
def m(name, val): M[name] = val

# ---- data
m("nEvents", num(ds["raw_events"])); m("nSessions", num(ds["sessions"])); m("nStudents", num(ds["students"]))
m("nInScope", num(ds["in_scope_events"])); m("nExitSess", num(ds["in_scope_exits"])); m("pExitSess", pc(ds["in_scope_exits"] / ds["sessions"]))
m("nAnswers", num(mcv.n_events[0])); m("nYG", num(ds["students_with_year_group"]))
m("nPause", num(ds["in_scope_exits_pause"])); m("nNoReturn", num(ds["in_scope_exits_noreturn"])); m("nGapDay", num(ds["within_session_gaps_gt_24h"]))
m("pMapped", pc(r2["events"] / 40787)); 
# ---- RQ1
t = st.loc["CheckinRetry_fail"]
m("nFR", num(t.events)); m("nFRexits", num(t.exits)); m("nFRstud", num(t.students))
m("rFR", pc(t.exit_rate_event_weighted)); m("rFRlo", pc(t.ew_ci_lo)); m("rFRhi", pc(t.ew_ci_hi))
m("swFR", pc(t.exit_rate_student_weighted)); m("swFRlo", pc(t.sw_ci_lo)); m("swFRhi", pc(t.sw_ci_hi))
m("rOth", pc(c1["rate_other_answers_event_weighted"], 2)); m("rdFR", pc(c1["risk_difference"])); m("rdLo", pc(c1["rd_ci"][0])); m("rdHi", pc(c1["rd_ci"][1]))
m("rrFR", f2(c1["risk_ratio"], 1)); m("rrLo", f2(c1["rr_ci"][0], 1)); m("rrHi", f2(c1["rr_ci"][1], 1))
m("pdMean", pc(c1["paired_mean_diff"])); m("pdLo", pc(c1["paired_diff_ci"][0])); m("pdHi", pc(c1["paired_diff_ci"][1])); m("pdN", num(c1["paired_students"]))
m("swOth", pc(c1["paired_mean_rate_other"])); m("shExits", pc(c1["share_of_all_exits_after_failed_checkin_retry"])); m("shAns", pc(c1["share_of_answer_events_that_are_failed_checkin_retry"]))
o = orr.loc["CheckinRetry_fail"]; m("orFR", f2(o.OR, 1)); m("orFRlo", f2(o.lo, 1)); m("orFRhi", f2(o.hi, 1))
m("orOthMax", f2(orr.drop("CheckinRetry_fail").OR.max(), 1)); m("orOthMin", f2(orr.drop("CheckinRetry_fail").OR.min(), 1))
pf = pos[(pos.group == "failed check-in retry")].set_index("question"); po = pos[(pos.group == "other answers")].set_index("question")
m("posFa", pc(pf.rate[1])); m("posFd", pc(pf.rate[4])); m("posOa", pc(po.rate[1], 2)); m("posOd", pc(po.rate[4], 2))
# ---- RQ2
tl = r2["topic_level"]; cl = r2["construct_level"]
m("nFRtopic", num(r2["events"])); m("nStudTopic", num(r2["students"])); m("pooledRate", pc(r2["pooled_rate"])); m("nTopics", num(r2["topics_tested"]))
m("nHsRaw", num(r2["hotspots_raw_q05"])); m("nHsAdj", num(r2["hotspots_adjusted_q05"])); m("nHsBoth", num(r2["raw_hotspots_surviving_adjustment"])); m("zsd", f2(r2["z_sd_across_topics"]))
m("rhoMean", f2z(tl["rho"]["mean"])); m("rhoLo", f2z(tl["rho"]["lo"])); m("rhoHi", f2z(tl["rho"]["hi"]))
m("sbMean", f2z(tl["sb"]["mean"])); m("sbLo", f2z(tl["sb"]["lo"])); m("sbHi", f2z(tl["sb"]["hi"]))
m("rhoNull", f2z(tl["rho_null_permuted"]["mean"])); m("rhoNullMax", f2z(tl["rho_null_permuted"]["max"]))
m("topA", f2(tl["top10_oe_in_A"]["mean"])); m("topB", f2(tl["top10_oe_in_B"]["mean"])); m("topBlo", f2(tl["top10_oe_in_B"]["lo"])); m("topBhi", f2(tl["top10_oe_in_B"]["hi"]))
m("restB", f2(tl["rest_oe_in_B"]["mean"])); m("pTopSig", pc(tl["top10_share_of_runs_p_lt_05_in_B"], 0)); m("pTopAbove", pc(tl["top10_share_of_runs_above_expected_in_B"], 0))
m("rhoCon", f2z(cl["rho_mean"])); m("rhoConLo", f2z(cl["rho_lo"])); m("rhoConHi", f2z(cl["rho_hi"])); m("nConTested", num(cl["tested"])); m("nConHs", num(cl["hotspots_q05"]))
m("nHsNoLat", num(r2["sensitivity_no_latency"]["hotspots_without_latency"]))
# ---- RQ3
tr = r3["trend"]; ha = r3["hotspots_young_vs_old"]
m("nYearEv", num(yr.events.sum())); m("nYearStud", num(yr.students.sum())); m("nExclEv", num(r3["excluded_failed_retry_events_outside_Y5_11"])); m("nExclStud", num(r3["excluded_students_outside_Y5_11"]))
a = tr["adjusted_event_weighted"]; u = tr["unadjusted_event_weighted"]
m("orYr", f2(a["OR_per_year"], 3)); m("orYrLo", f2(a["lo"], 3)); m("orYrHi", f2(a["hi"], 3)); m("orYrP", pstr(a["p"]))
m("orYrU", f2(u["OR_per_year"], 3)); m("orYrUp", pstr(u["p"]))
s = tr["adjusted_student_level_pp_per_year"]; su = tr["unadjusted_student_level_pp_per_year"]
m("ppYr", f2(s["slope_pp"])); m("ppYrLo", f2(s["lo"])); m("ppYrHi", f2(s["hi"])); m("ppYrP", pstr(s["p"])); m("ppYrU", f2(su["slope_pp"])); m("ppYrUp", pstr(su["p"]))
m("rYoung", pc(r3["rate_Y5_7"])); m("rOld", pc(r3["rate_Y9_11"]))
ov = r3["older_vs_younger_adjusted_OR"]; m("orOld", f2(ov["OR"])); m("orOldLo", f2(ov["lo"])); m("orOldHi", f2(ov["hi"])); m("orOldP", pstr(ov["p"]))
ot = r3["trend_other_answers_unadjusted"]; m("orOthYr", f2(ot["OR_per_year"], 3)); m("orOthYrLo", f2(ot["lo"], 3)); m("orOthYrHi", f2(ot["hi"], 3))
it = r3["age_slope_interaction_failed_retry_vs_other"]; m("orInt", f2(it["OR_ratio"])); m("orIntLo", f2(it["lo"])); m("orIntHi", f2(it["hi"])); m("orIntP", pstr(it["p"]))
m("quadP", pstr(r3["quadratic_term_p"]))
m("rhoAge", f2z(ha["rho"])); m("rhoAgeLo", f2z(ha["rho_ci"][0])); m("rhoAgeHi", f2z(ha["rho_ci"][1])); m("nTopAge", num(ha["topics_compared"]))
m("relY", f2z(ha["reliability_young"])); m("relO", f2z(ha["reliability_old"])); m("topOverlap", num(ha["top10_overlap"]))
m("nDiff", num(ha["n_topics_age_differential_q05"])); m("nDiffTested", num(ha["n_tested_for_difference"]))
m("yrFirst", pc(yr.ew.iloc[0])); m("yrLast", pc(yr.ew.iloc[-1]))
with open(REP / "auto_numbers.tex", "w") as f:
    f.write("% AUTO-GENERATED by analysis/06_make_report_inputs.py - do not edit by hand\n")
    for k, v in M.items(): f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

# ---- tables
# Table: sensitivity of RQ1 to the definition of exit
nm = {"primary (last event, never reached Q5)": "Primary: last event of a session that never reached Q5",
      "pause (also any >1h pause in the session)": "Also counting any pause of more than 1 hour",
      "no return (no new session within 24h)": "Exit and no new session within 24 hours",
      "primary, drop sessions ending in last 24h of data": "Primary, dropping sessions in the last 24 hours of data"}
rows = [f"{nm[r.definition]} & {num(r.exits_total)} & {pc(r.rate_failed_retry)} & {pc(r.rate_other_answers, 2)} & {r.risk_ratio:.1f} [{r.rr_lo:.1f}, {r.rr_hi:.1f}] \\\\" for r in sens.itertuples()]
open(REP / "tab_sens.tex", "w").write("\\begin{tabular}{@{}lrrrr@{}}\\toprule\nExit definition & Exits & After failed retry (\\%) & After other answers (\\%) & Risk ratio [95\\% CI] \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
qfmt = lambda q: "$<$.001" if q < .001 else f"{q:.3f}".lstrip("0")
# Table: confirmed hotspots
h = full[full.q < 0.05].sort_values("oe", ascending=False)
rows = []
for sid, r in h.iterrows():
    b = hb.loc[sid]
    rows.append(f"{r.topic} & {int(r.n)} & {int(r.students)} & {pc(r.raw_rate)} & {r.oe:.2f} [{r.oe_lo:.2f}, {r.oe_hi:.2f}] & {qfmt(r.q)} & {b.oe_young:.2f} ({int(b.n_young)}) & {b.oe_old:.2f} ({int(b.n_old)}) \\\\")
open(REP / "tab_hot.tex", "w").write("\\begin{tabular}{@{}lrrrcrcc@{}}\\toprule\nTopic & Events & Students & Exit (\\%) & O/E [95\\% CI] & $q$ & O/E Y5--7 ($n$) & O/E Y9--11 ($n$) \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
# Table: predictive validation
TIMES = "$\\times$"
rows = [f"{r.model.replace(' x ', ' ' + TIMES + ' ')} & {r.auc_mean:.3f} ({r.auc_sd:.3f}) & {r.avg_precision:.3f} ({r.ap_sd:.3f}) & {r.log_loss:.4f} ({r.ll_sd:.4f}) \\\\" for r in mcv.itertuples()]
open(REP / "tab_model.tex", "w").write("\\begin{tabular}{@{}lccc@{}}\\toprule\nModel & AUC & Average precision & Log-loss \\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n")
print(open(REP / "auto_numbers.tex").read()); print(open(REP / "tab_hot.tex").read()); print(open(REP / "tab_model.tex").read()); print(open(REP / "tab_sens.tex").read())
