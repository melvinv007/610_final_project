"""RQ3 - Does frustration change with age?  (a) exit rate after a failed check-in retry by year group,
(b) trend tests, (c) do younger and older students share the same topic hotspots?"""
import json
import numpy as np, pandas as pd
import statsmodels.api as sm, statsmodels.formula.api as smf
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

ev = pd.read_pickle(WORK / "events.pkl")
sc = ev[ev.in_scope == 1]
F = add_covariates(sc[sc.state == "CheckinRetry_fail"])
F = F[F.YearGroup.between(5, 11)].copy()
OTH = sc[(sc.state != "CheckinRetry_fail") & (sc.state != "Lesson") & sc.YearGroup.between(5, 11)]
print("failed-retry events Y5-11:", len(F), " students:", F.UserId.nunique())
out = {}

# ------------------------------------------------------------------ (a) rates by year group
def boot_rate(df, B=2000):
    g = df.groupby("UserId").is_exit.agg(["sum", "size"]); k, n = g["sum"].values.astype(float), g["size"].values.astype(float)
    def st(idx): return np.stack([k[idx].sum(1) / n[idx].sum(1), (k[idx] / n[idx]).mean(1)], 1)
    bs = cluster_bootstrap(st, len(k), B=B, chunk=250)
    lo, hi = pct_ci(bs)
    return dict(events=int(n.sum()), students=len(k), ew=k.sum() / n.sum(), ew_lo=lo[0], ew_hi=hi[0], sw=(k / n).mean(), sw_lo=lo[1], sw_hi=hi[1])
rows = []
for yg in range(5, 12):
    r = boot_rate(F[F.YearGroup == yg]); r["year"] = yg
    o = boot_rate(OTH[OTH.YearGroup == yg], B=500); r["other_ew"] = o["ew"]; r["other_lo"] = o["ew_lo"]; r["other_hi"] = o["ew_hi"]
    rows.append(r)
yt = pd.DataFrame(rows).set_index("year"); yt.to_csv(TAB / "rq3_by_year.csv"); print(yt.round(4).to_string())
excl = sc[(sc.state == "CheckinRetry_fail") & ~sc.YearGroup.between(5, 11)]
out["excluded_failed_retry_events_outside_Y5_11"] = int(len(excl)); out["excluded_students_outside_Y5_11"] = int(excl.UserId.nunique())

# ------------------------------------------------------------------ (b) trend tests (student-clustered SEs)
F["ygc"] = F.YearGroup - 8
cov = "C(qs) + C(sess) + C(pfail) + C(block) + wkend + lat"
def trend(formula):
    m = smf.logit(formula, data=F).fit(disp=0, cov_type="cluster", cov_kwds={"groups": F.UserId.values})
    b, ci = m.params["ygc"], m.conf_int().loc["ygc"]
    return {"OR_per_year": float(np.exp(b)), "lo": float(np.exp(ci[0])), "hi": float(np.exp(ci[1])), "p": float(m.pvalues["ygc"])}
out["trend"] = {"unadjusted_event_weighted": trend("is_exit ~ ygc"), "adjusted_event_weighted": trend(f"is_exit ~ ygc + {cov}")}
# student-weighted trend: one row per student (their own exit proportion), OLS with HC3 robust SE
S = F.groupby("UserId").agg(rate=("is_exit", "mean"), n=("is_exit", "size"), ygc=("ygc", "first"), qs=("qs", "mean"), sess=("sess", "mean"),
                            pfail=("pfail", "mean"), lat=("lat", "mean"), wkend=("wkend", "mean"), school=("block", lambda s: (s == "school").mean())).reset_index()
for nm, fm in [("unadjusted_student_level", "rate ~ ygc"), ("adjusted_student_level", "rate ~ ygc + qs + sess + pfail + lat + wkend + school")]:
    mm = smf.ols(fm, data=S).fit(cov_type="HC3"); ci = mm.conf_int().loc["ygc"]
    out["trend"][nm + "_pp_per_year"] = {"slope_pp": float(mm.params["ygc"] * 100), "lo": float(ci[0] * 100), "hi": float(ci[1] * 100), "p": float(mm.pvalues["ygc"]), "students": int(len(S))}
# curvature check
F["ygc2"] = F.ygc ** 2
mq = smf.logit(f"is_exit ~ ygc + ygc2 + {cov}", data=F).fit(disp=0, cov_type="cluster", cov_kwds={"groups": F.UserId.values})
out["quadratic_term_p"] = float(mq.pvalues["ygc2"])
# younger (Y5-7) vs older (Y9-11) contrast
yb = F[F.YearGroup.isin([5, 6, 7, 9, 10, 11])].copy(); yb["old"] = yb.YearGroup.isin([9, 10, 11]).astype(int)
mo = smf.logit(f"is_exit ~ old + {cov}", data=yb).fit(disp=0, cov_type="cluster", cov_kwds={"groups": yb.UserId.values})
ci = mo.conf_int().loc["old"]; out["older_vs_younger_adjusted_OR"] = {"OR": float(np.exp(mo.params["old"])), "lo": float(np.exp(ci[0])), "hi": float(np.exp(ci[1])), "p": float(mo.pvalues["old"])}
out["rate_Y5_7"] = float(F[F.YearGroup.isin([5, 6, 7])].is_exit.mean()); out["rate_Y9_11"] = float(F[F.YearGroup.isin([9, 10, 11])].is_exit.mean())
# is the decline specific to frustration?  trend in exit after OTHER answers
O2 = OTH.copy(); O2["ygc"] = O2.YearGroup - 8
mo2 = smf.logit("is_exit ~ ygc", data=O2).fit(disp=0, cov_type="cluster", cov_kwds={"groups": O2.UserId.values})
ci = mo2.conf_int().loc["ygc"]; out["trend_other_answers_unadjusted"] = {"OR_per_year": float(np.exp(mo2.params["ygc"])), "lo": float(np.exp(ci[0])), "hi": float(np.exp(ci[1])), "p": float(mo2.pvalues["ygc"])}
P = pd.concat([F.assign(fr=1)[["UserId", "ygc", "is_exit", "fr"]], O2.assign(fr=0)[["UserId", "ygc", "is_exit", "fr"]]])
mi = smf.logit("is_exit ~ ygc * fr", data=P).fit(disp=0, cov_type="cluster", cov_kwds={"groups": P.UserId.values})
ci = mi.conf_int().loc["ygc:fr"]; out["age_slope_interaction_failed_retry_vs_other"] = {"OR_ratio": float(np.exp(mi.params["ygc:fr"])), "lo": float(np.exp(ci[0])), "hi": float(np.exp(ci[1])), "p": float(mi.pvalues["ygc:fr"])}
# compositional check: are younger students simply earlier in their platform history?
print(F.groupby("YearGroup").agg(first_session_share=("sess", lambda s: (s == 1).mean()), mean_prior_fail=("n_prior_fail", "mean")).round(3))

# ------------------------------------------------------------------ (c) shared hotspots, younger vs older
G = F[F.SubjectId.notna()].copy(); G["SubjectId"] = G.SubjectId.astype(int)
name_of = G.drop_duplicates("SubjectId").set_index("SubjectId").SubjectName

def band_tab(df, min_n=15):
    X = design_matrix(df); y = df.is_exit.values.astype(float)
    p = expected_prob(X.values, y, np.ones(len(df), bool))
    return oe_table(df, p, "SubjectId", min_n=min_n), df.assign(p=p)

def matrices(d, topics):
    """users x topics matrices of observed exits and expected exits."""
    O = d.pivot_table(index="UserId", columns="SubjectId", values="is_exit", aggfunc="sum", fill_value=0).reindex(columns=topics, fill_value=0)
    E = d.pivot_table(index="UserId", columns="SubjectId", values="p", aggfunc="sum", fill_value=0).reindex(columns=topics, fill_value=0)
    return O.values.astype(float), E.reindex(O.index).values.astype(float)

def compare(young_years, old_years, B=500, R=200):
    dy, do = G[G.YearGroup.isin(young_years)], G[G.YearGroup.isin(old_years)]
    ty, dyp = band_tab(dy); to, dop = band_tab(do)
    topics = ty.index[ty.n >= 15].intersection(to.index[to.n >= 15])
    ly, lo_ = np.log(ty.loc[topics, "oe"].clip(lower=.05)), np.log(to.loc[topics, "oe"].clip(lower=.05))
    rho = stats.spearmanr(ly, lo_)[0]
    Oy, Ey = matrices(dyp, topics); Oo, Eo = matrices(dop, topics)
    rng = np.random.default_rng(11)
    boots = []
    for _ in range(B):
        iy = rng.integers(0, len(Oy), len(Oy)); io = rng.integers(0, len(Oo), len(Oo))
        a = np.log(np.clip(Oy[iy].sum(0) / np.maximum(Ey[iy].sum(0), 1e-9), .05, None)); b = np.log(np.clip(Oo[io].sum(0) / np.maximum(Eo[io].sum(0), 1e-9), .05, None))
        boots.append(stats.spearmanr(a, b)[0])
    # within-band reliability: split-half over students, Spearman-Brown
    def rel(O, E):
        rs = []
        for _ in range(R):
            perm = rng.permutation(len(O)); h = len(O) // 2; A, Bh = perm[:h], perm[h:]
            a = np.log(np.clip(O[A].sum(0) / np.maximum(E[A].sum(0), 1e-9), .05, None)); b = np.log(np.clip(O[Bh].sum(0) / np.maximum(E[Bh].sum(0), 1e-9), .05, None))
            r = stats.spearmanr(a, b)[0]; rs.append(2 * r / (1 + r))
        return float(np.mean(rs))
    ry, ro = rel(Oy, Ey), rel(Oo, Eo)
    dis = rho / np.sqrt(max(ry, 1e-6) * max(ro, 1e-6))
    # per-topic age difference
    hs = pd.read_csv(TAB / "rq2_topic_full.csv", index_col=0); hs = hs[hs.q < 0.05].index
    hb = pd.DataFrame({"topic": name_of.reindex(hs).values, "n_young": ty.reindex(hs).n, "oe_young": ty.reindex(hs).oe, "n_old": to.reindex(hs).n, "oe_old": to.reindex(hs).oe}, index=hs)
    if len(young_years) == 3: hb.to_csv(TAB / "rq3_hotspots_by_band.csv"); print(hb.round(2).to_string())
    se_y = ty.loc[topics, "se"] / ty.loc[topics, "O"].clip(lower=1) * 1.0; se_o = to.loc[topics, "se"] / to.loc[topics, "O"].clip(lower=1)
    ok = (ty.loc[topics, "O"] >= 5) & (to.loc[topics, "O"] >= 5)
    zdiff = (np.log(ty.loc[topics, "oe"].clip(lower=.05)) - np.log(to.loc[topics, "oe"].clip(lower=.05)))[ok] / np.sqrt(se_y[ok] ** 2 + se_o[ok] ** 2)
    pd_ = 2 * (1 - stats.norm.cdf(np.abs(zdiff))); qd = bh_qvalues(np.asarray(pd_))
    topy = set(ty.loc[topics, "z"].sort_values(ascending=False).head(10).index); topo = set(to.loc[topics, "z"].sort_values(ascending=False).head(10).index)
    res = {"young": list(young_years), "old": list(old_years), "events_young": int(len(dy)), "events_old": int(len(do)),
           "topics_compared": int(len(topics)), "rho": float(rho), "rho_ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
           "reliability_young": ry, "reliability_old": ro, "rho_disattenuated": float(min(dis, 1.0)),
           "top10_overlap": len(topy & topo), "n_topics_age_differential_q05": int((qd < 0.05).sum()), "n_tested_for_difference": int(ok.sum()),
           "hotspots_young_q05": int((ty.q < 0.05).sum()), "hotspots_old_q05": int((to.q < 0.05).sum())}
    tab = pd.DataFrame({"topic": name_of.reindex(topics).values, "n_young": ty.loc[topics, "n"], "oe_young": ty.loc[topics, "oe"], "z_young": ty.loc[topics, "z"],
                        "n_old": to.loc[topics, "n"], "oe_old": to.loc[topics, "oe"], "z_old": to.loc[topics, "z"]}, index=topics)
    return res, tab

res_main, tab_main = compare([5, 6, 7], [9, 10, 11]); res_alt, _ = compare([5, 6, 7, 8], [9, 10, 11])
out["hotspots_young_vs_old"] = res_main; out["hotspots_young_vs_old_sensitivity_Y5_8_vs_Y9_11"] = res_alt
tab_main.sort_values("z_young", ascending=False).to_csv(TAB / "rq3_topics_young_vs_old.csv")
json.dump(out, open(TAB / "rq3_summary.json", "w"), indent=2, default=float); print(json.dumps(out, indent=2, default=float))
# (figures are drawn by 05_make_figures.py from the saved tables)
