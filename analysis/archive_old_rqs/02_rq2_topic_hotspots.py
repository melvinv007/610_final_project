"""RQ2 - Which topics are frustration hotspots (exit after a failed check-in retry well above expected)?
Unit: subject (topic).  Secondary: construct.  Replication: repeated split-half over students."""
import json
import numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from common import *

ev = pd.read_pickle(WORK / "events.pkl")
f = ev[(ev.in_scope == 1) & (ev.state == "CheckinRetry_fail")].copy()
f = add_covariates(f)
n_all = len(f)
f = f[f.SubjectId.notna()].copy(); f["SubjectId"] = f.SubjectId.astype(int)
print(f"failed check-in retry events (Q1-4): {n_all}; with a mapped topic: {len(f)} ({len(f)/n_all:.1%})")
name_of = f.drop_duplicates("SubjectId").set_index("SubjectId").SubjectName
strand_of = f.drop_duplicates("SubjectId").set_index("SubjectId").Strand

X = design_matrix(f); y = f.is_exit.values.astype(float)
all_rows = np.ones(len(f), bool)
p_hat = expected_prob(X.values, y, all_rows)
pooled = y.mean()
print("pooled exit rate after failed retry:", round(pooled, 4))

# ---------------------------------------------------------------- full-sample topic table (raw vs adjusted)
adj = oe_table(f, p_hat, "SubjectId", min_n=30)
raw = oe_table(f, np.full(len(f), pooled), "SubjectId", min_n=30)
full = adj[["n", "students", "O", "E", "oe", "oe_lo", "oe_hi", "raw_rate", "z", "p_one", "q"]].copy()
full["adj_rate"] = full.oe * pooled
full["raw_p"] = raw.p_one; full["raw_q"] = raw.q
full.insert(0, "topic", name_of.reindex(full.index).values); full.insert(1, "strand", strand_of.reindex(full.index).values)
full = full[full.n >= 30].sort_values("oe", ascending=False)
full.to_csv(TAB / "rq2_topic_full.csv")
hs_adj = full[full.q < 0.05]; hs_raw = full[full.raw_q < 0.05]
print(f"topics tested (n>=30): {len(full)};  hotspots raw q<.05: {len(hs_raw)};  adjusted q<.05: {len(hs_adj)};  "
      f"raw hotspots still significant after adjustment: {len(set(hs_raw.index) & set(hs_adj.index))}")
print(hs_adj[["topic", "n", "students", "raw_rate", "oe", "oe_lo", "oe_hi", "q"]].round(3).to_string())
# heterogeneity across topics: chi-square on residuals (are topics different at all?)
Z = full.z.values; print("topic heterogeneity: mean z", Z.mean().round(3), " sd z", Z.std().round(3), "(sd>1 means more spread than chance)")

# ---------------------------------------------------------------- replication: repeated split-half over STUDENTS
users = f.UserId.unique()
def set_test(sub, p):
    """Pooled observed-vs-expected test (one-sided, student-clustered) for a SET of topics in a sample."""
    d = sub.assign(p=p); r = d.is_exit - d.p
    V = (d.p * (1 - d.p)).sum(); R2 = r.groupby(d.UserId).sum().pow(2).sum(); S2 = (r ** 2).sum()
    se = np.sqrt(V * max(1.0, R2 / S2)); O, E = d.is_exit.sum(), d.p.sum()
    return O / E, 1 - stats.norm.cdf((O - E) / se)

def split_stats(seed, key="SubjectId", min_half=15, k_top=10):
    rng = np.random.default_rng(seed)
    inA = set(rng.choice(users, size=len(users) // 2, replace=False))
    mA = f.UserId.isin(inA).values
    tabs, pr = [], []
    for m in (mA, ~mA):
        p = expected_prob(Xm[key], y, m)                           # covariate model fitted within each half
        pr.append(p); tabs.append(oe_table(f[m], p[m], key, min_n=min_half))
    tA, tB = tabs
    idx = tA.index[(tA.n >= min_half)].intersection(tB.index[(tB.n >= min_half)])
    a, b = tA.loc[idx], tB.loc[idx]
    la, lb = np.log(a.oe.clip(lower=.05)), np.log(b.oe.clip(lower=.05))
    rho = stats.spearmanr(la, lb)[0]
    big = (a.n >= 30) & (b.n >= 30)
    rho30 = stats.spearmanr(la[big], lb[big])[0] if big.sum() > 10 else np.nan
    fl = a.q < 0.05
    top = a.z.sort_values(ascending=False).head(k_top).index          # strongest evidence in discovery half A
    inB = f[~mA & f[key].isin(top).values]
    oe_top_B, p_top_B = set_test(inB, pr[1][~mA & f[key].isin(top).values])
    rest = f[~mA & ~f[key].isin(top).values & f[key].isin(idx).values]
    out = {"rho": rho, "sb": 2 * rho / (1 + rho), "rho_n30": rho30, "n_topics": len(idx), "n_flagged_A": int(fl.sum()),
           "top10_oe_in_B": oe_top_B, "top10_p_in_B": p_top_B, "top10_oe_in_A": float(a.loc[top, "oe"].mean()),
           "rest_oe_in_B": float(b.loc[b.index.difference(top), "oe"].mean())}
    if fl.sum() >= 1:
        out["oe_B_flagged"] = float(b.loc[fl, "oe"].mean()); out["oe_B_unflagged"] = float(b.loc[~fl, "oe"].mean())
        out["share_flagged_above_expected_B"] = float((b.loc[fl, "oe"] > 1).mean())
        out["share_flagged_sig_B_p05"] = float((b.loc[fl, "p_one"] < 0.05).mean())
    return out, a, b, fl, top

Xm = {"SubjectId": X.values, "ConstructId": X.values}
R = 100
res = [split_stats(1000 + r)[0] for r in range(R)]
rs = pd.DataFrame(res); rs.to_csv(TAB / "rq2_split_half_runs.csv", index=False)
def ms(c): v = rs[c].dropna(); return {"mean": float(v.mean()), "lo": float(v.quantile(.025)), "hi": float(v.quantile(.975)), "n_runs": int(len(v))}
summary = {"topic_level": {c: ms(c) for c in ["rho", "sb", "rho_n30", "n_topics", "n_flagged_A", "oe_B_flagged", "oe_B_unflagged",
                                              "share_flagged_above_expected_B", "share_flagged_sig_B_p05", "top10_oe_in_A", "top10_oe_in_B",
                                              "rest_oe_in_B"]}}
summary["topic_level"]["top10_share_of_runs_p_lt_05_in_B"] = float((rs.top10_p_in_B < 0.05).mean())
summary["topic_level"]["top10_share_of_runs_above_expected_in_B"] = float((rs.top10_oe_in_B > 1).mean())
# permutation null for rho: shuffle topic labels among events (destroys topic effect), same pipeline
rng = np.random.default_rng(7); null = []
for r in range(30):
    f2 = f.copy(); f2["SubjectId"] = rng.permutation(f2.SubjectId.values)
    rngs = np.random.default_rng(5000 + r); inA = set(rngs.choice(users, size=len(users) // 2, replace=False)); mA = f2.UserId.isin(inA).values
    tabs = []
    for m in (mA, ~mA):
        p = expected_prob(X.values, y, m); tabs.append(oe_table(f2[m], p[m], "SubjectId", min_n=15))
    idx = tabs[0].index[tabs[0].n >= 15].intersection(tabs[1].index[tabs[1].n >= 15])
    null.append(stats.spearmanr(np.log(tabs[0].loc[idx, "oe"].clip(lower=.05)), np.log(tabs[1].loc[idx, "oe"].clip(lower=.05)))[0])
summary["topic_level"]["rho_null_permuted"] = {"mean": float(np.mean(null)), "max": float(np.max(null))}

# construct level (secondary)
fc = f.copy()
res_c = []
for r in range(50):
    rng = np.random.default_rng(2000 + r); inA = set(rng.choice(users, size=len(users) // 2, replace=False)); mA = fc.UserId.isin(inA).values
    tabs = []
    for m in (mA, ~mA):
        p = expected_prob(X.values, y, m); tabs.append(oe_table(fc[m], p[m], "ConstructId", min_n=10))
    idx = tabs[0].index[tabs[0].n >= 10].intersection(tabs[1].index[tabs[1].n >= 10])
    rho = stats.spearmanr(np.log(tabs[0].loc[idx, "oe"].clip(lower=.05)), np.log(tabs[1].loc[idx, "oe"].clip(lower=.05)))[0]
    res_c.append({"rho": rho, "n_constructs": len(idx)})
rc = pd.DataFrame(res_c)
summary["construct_level"] = {"rho_mean": float(rc.rho.mean()), "rho_lo": float(rc.rho.quantile(.025)), "rho_hi": float(rc.rho.quantile(.975)),
                              "n_constructs_mean": float(rc.n_constructs.mean())}
# construct-level full-sample hotspots (n>=30)
adjc = oe_table(f, p_hat, "ConstructId", min_n=30)
adjc = adjc[adjc.n >= 30].sort_values("oe", ascending=False)
cn = f.drop_duplicates("ConstructId").set_index("ConstructId").SubjectName
adjc.insert(0, "topic", cn.reindex(adjc.index).values); adjc["adj_rate"] = adjc.oe * pooled
adjc[["topic", "n", "students", "O", "E", "oe", "oe_lo", "oe_hi", "raw_rate", "adj_rate", "p_one", "q"]].to_csv(TAB / "rq2_construct_full.csv")
summary["construct_level"]["tested"] = int(len(adjc)); summary["construct_level"]["hotspots_q05"] = int((adjc.q < 0.05).sum())
summary.update({"pooled_rate": float(pooled), "events": int(len(f)), "students": int(f.UserId.nunique()), "topics_tested": int(len(full)),
                "hotspots_adjusted_q05": int(len(hs_adj)), "hotspots_raw_q05": int(len(hs_raw)),
                "raw_hotspots_surviving_adjustment": int(len(set(hs_raw.index) & set(hs_adj.index))),
                "z_sd_across_topics": float(Z.std())})
json.dump(summary, open(TAB / "rq2_summary.json", "w"), indent=2)
print(json.dumps(summary, indent=2))

# ---------------------------------------------------------------- primary split used for the figure / table
out, a, b, fl, top = split_stats(610)
print("primary split:", out)
ps = pd.DataFrame({"topic": name_of.reindex(a.index).values, "nA": a.n, "oeA": a.oe, "zA": a.z, "qA": a.q, "nB": b.n, "oeB": b.oe, "pB": b.p_one}, index=a.index)
ps["top10_A"] = ps.index.isin(top); ps.sort_values("zA", ascending=False).to_csv(TAB / "rq2_primary_split.csv")
print(ps[ps.top10_A].sort_values("zA", ascending=False).round(3).to_string())
fl = pd.Series(ps.index.isin(top), index=ps.index)

# sensitivity: expected-rate model without the retry-latency covariate
X2 = design_matrix(f, "C(qs) + C(sess) + C(pfail) + C(yg) + C(block) + wkend")
adj2 = oe_table(f, expected_prob(X2.values, y, all_rows), "SubjectId", min_n=30)
sens = {"hotspots_without_latency": int((adj2.q < 0.05).sum()), "overlap_with_main": int(len(set(adj2.index[adj2.q < 0.05]) & set(hs_adj.index)))}
summary["sensitivity_no_latency"] = sens; json.dump(summary, open(TAB / "rq2_summary.json", "w"), indent=2); print(sens)
# (figures are drawn by 05_make_figures.py from the saved tables)
