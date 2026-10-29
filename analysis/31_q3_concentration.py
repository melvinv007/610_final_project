"""RQ3 (extra): how concentrated is quitting among students, and does a run of recent quits predict the next one?
Writes results/tables/q3_student_rates.csv, q3_lorenz.csv, q3_streak.csv, q3_concentration.json"""
import importlib.util, json, numpy as np, pandas as pd
from common import TAB, SEED
spec = importlib.util.spec_from_file_location("q3", __file__.replace("31_q3_concentration.py", "30_q3_persistence_return.py")); q3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(q3)
s = q3.sessions(); u, _ = pd.factorize(s.UserId); s["u"] = u; nU = u.max() + 1
st = s.groupby("u").exit.agg(["sum", "size"]); st = st[st["size"] >= 5]; st["rate"] = st["sum"] / st["size"]; st.to_csv(TAB / "q3_student_rates.csv")
x = st["sum"].sort_values().values; cum = np.cumsum(x) / x.sum(); frac = np.arange(1, len(x) + 1) / len(x)
pd.DataFrame({"share_students": np.r_[0, frac], "share_exits": np.r_[0, cum]}).to_csv(TAB / "q3_lorenz.csv", index=False)
gini = 1 - 2 * np.trapz(np.r_[0, cum], np.r_[0, frac])
# streak: number of exits in previous 3 sessions
g = s.groupby("u"); s["prev3"] = sum(g.exit.shift(i).fillna(0) for i in (1, 2, 3)); s["has3"] = g.cumcount() >= 3
P = s[s.has3].copy(); rng = np.random.default_rng(SEED); W = rng.multinomial(nU, np.ones(nU) / nU, size=400).astype(float); rows = []
for k, h in P.groupby("prev3"):
    kk = np.bincount(h.u, weights=h.exit, minlength=nU); nn = np.bincount(h.u, minlength=nU).astype(float); bs = (W @ kk) / np.maximum(W @ nn, 1)
    rows.append({"quits_in_previous_3": int(k), "sessions": int(nn.sum()), "rate": kk.sum() / nn.sum(), "lo": np.percentile(bs, 2.5), "hi": np.percentile(bs, 97.5)})
pd.DataFrame(rows).to_csv(TAB / "q3_streak.csv", index=False); print(pd.DataFrame(rows).round(3))
json.dump({"gini": float(gini), "median_student_rate": float(st.rate.median()), "share_ge_half": float((st.rate >= .5).mean()), "n": int(len(st))}, open(TAB / "q3_concentration.json", "w"), indent=1)
print(gini, st.rate.median(), (st.rate >= .5).mean())
