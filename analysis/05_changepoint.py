"""Step 05 - temporal structure in calendar time: is there a break in the exit rate after a failed retry?

Method : binomial likelihood-ratio scan. For every candidate break date, compare "one exit rate overall" with
         "one rate before, another after"; the break date is the maximiser of the likelihood ratio.
         Uncertainty: bootstrap over STUDENTS (their whole day-by-day history is resampled together).
Also   : the same students observed on both sides of the break (paired, so a change in who uses the platform cannot
         explain it), the same scan for exits after other answers, and exit rates by weekday and hour.
Writes : results/tables/cp_summary.json, cp_scan.csv, cp_weekday_hour.csv, cp_monthly.csv
"""
import numpy as np
import pandas as pd
from common import WORK, TAB, SEED, pct_ci

N_BOOT = 500
TRIM = 0.10


def scan(k_day, n_day):
    """k_day, n_day: (days,) totals. Returns LR statistic for a break placed BEFORE each day index (1..D-1)."""
    ck, cn = np.cumsum(k_day), np.cumsum(n_day)
    K, N = ck[-1], cn[-1]
    k1, n1 = ck[:-1], cn[:-1]
    k2, n2 = K - k1, N - n1
    def ll(k, n):
        p = np.where(n > 0, k / np.maximum(n, 1), 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(k * np.log(p) + (n - k) * np.log(1 - p))
    lr = 2 * (ll(k1, n1) + ll(k2, n2) - ll(K, N))
    lo, hi = int(TRIM * len(n_day)), int((1 - TRIM) * len(n_day))
    mask = np.zeros(len(lr), bool); mask[lo:hi] = True
    return np.where(mask, lr, -np.inf)


def main():
    ev = pd.read_pickle(WORK / "events.pkl")
    ev["date"] = ev.Timestamp.dt.normalize()
    sc = ev[ev.in_scope == 1]
    days = pd.date_range(sc.date.min(), sc.date.max(), freq="D")
    out = {}
    for name, d in [("failed_retry", sc[sc.state == "CheckinRetry_fail"]),
                    ("other_answers", sc[(sc.state != "CheckinRetry_fail") & (sc.state != "Lesson")])]:
        users = np.sort(d.UserId.unique()); uidx = pd.Series(np.arange(len(users)), index=users)
        di = pd.Series(np.arange(len(days)), index=days)
        ui, dd = uidx.reindex(d.UserId).values, di.reindex(d.date).values
        Kmat = np.zeros((len(users), len(days))); Nmat = np.zeros_like(Kmat)
        np.add.at(Kmat, (ui, dd), d.is_exit.values); np.add.at(Nmat, (ui, dd), 1)
        lr = scan(Kmat.sum(0), Nmat.sum(0)); best = int(np.argmax(lr)) + 1     # break placed before day index `best`
        rng = np.random.default_rng(SEED); boots = []
        for _ in range(N_BOOT):
            w = np.bincount(rng.integers(0, len(users), len(users)), minlength=len(users)).astype(float)
            boots.append(int(np.argmax(scan(w @ Kmat, w @ Nmat))) + 1)
        lo, hi = pct_ci(np.array(boots, float))
        bd = days[best]
        pre, post = d[d.date < bd], d[d.date >= bd]
        out[name] = {"break_date": str(bd.date()), "break_ci_days": [str(days[int(lo)].date()), str(days[int(hi)].date())],
                     "lr_stat": float(lr[best - 1]), "rate_before": float(pre.is_exit.mean()), "rate_after": float(post.is_exit.mean()),
                     "n_before": int(len(pre)), "n_after": int(len(post))}
        if name == "failed_retry":
            scan_df = pd.DataFrame({"date": days[1:], "lr": lr}); scan_df.to_csv(TAB / "cp_scan.csv", index=False)
            # same students on both sides (paired)
            a, b = pre.groupby("UserId").is_exit.agg(["mean", "size"]), post.groupby("UserId").is_exit.agg(["mean", "size"])
            both = a.join(b, lsuffix="_pre", rsuffix="_post", how="inner"); both = both[(both.size_pre >= 3) & (both.size_post >= 3)]
            rng = np.random.default_rng(SEED + 1); diffs = both.mean_post.values - both.mean_pre.values
            bs = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(2000)]
            keep = d[d.UserId.isin(both.index)]
            out["paired_students"] = {"n_students": int(len(both)), "mean_rate_before": float(both.mean_pre.mean()), "mean_rate_after": float(both.mean_post.mean()),
                                      "mean_diff": float(diffs.mean()), "diff_ci": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                                      "pooled_rate_before": float(keep[keep.date < bd].is_exit.mean()), "pooled_rate_after": float(keep[keep.date >= bd].is_exit.mean())}
            yg = d.assign(per=np.where(d.date >= bd, "after", "before"), yg=d.YearGroup.clip(5, 11))
            out["by_year_group"] = (yg.groupby(["yg", "per"]).is_exit.mean().unstack().round(4)).reset_index().to_dict("records")
            wk = d.assign(per=np.where(d.date >= bd, "after", "before"), wkend=d.weekday >= 5).groupby(["per", "wkend"]).is_exit.agg(["size", "mean"]).reset_index()
            wk.to_csv(TAB / "cp_weekend.csv", index=False)
            hr = d.groupby(pd.cut(d.hour, [-1, 6, 8, 10, 12, 14, 16, 18, 20, 23])).is_exit.agg(["size", "mean"]).reset_index()
            hr["hour"] = hr.hour.astype(str); hr.to_csv(TAB / "cp_hour.csv", index=False)
            wd = d.groupby("weekday").is_exit.agg(["size", "mean"]).reset_index(); wd.to_csv(TAB / "cp_weekday.csv", index=False)
            mo = d.groupby(d.Timestamp.dt.to_period("M")).agg(events=("is_exit", "size"), students=("UserId", "nunique"), rate=("is_exit", "mean")).reset_index()
            mo["month"] = mo.Timestamp.astype(str); mo.drop(columns="Timestamp").to_csv(TAB / "cp_monthly.csv", index=False)
    out["break_date"] = out["failed_retry"]["break_date"]
    # ratio of failed-retry exit to other-answer exit on each side of the break (does the gap itself change?)
    bd = pd.Timestamp(out["break_date"])
    f, o = sc[sc.state == "CheckinRetry_fail"], sc[(sc.state != "CheckinRetry_fail") & (sc.state != "Lesson")]
    out["ratio_before"] = float(f[f.date < bd].is_exit.mean() / o[o.date < bd].is_exit.mean())
    out["ratio_after"] = float(f[f.date >= bd].is_exit.mean() / o[o.date >= bd].is_exit.mean())
    import json
    json.dump(out, open(TAB / "cp_summary.json", "w"), indent=2, default=float)
    print(json.dumps({k: v for k, v in out.items() if k != "by_year_group"}, indent=2, default=float))
    print(pd.DataFrame(out["by_year_group"]).to_string(index=False))


if __name__ == "__main__":
    main()
