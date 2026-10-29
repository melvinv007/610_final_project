"""Step 4 - does event type (and topic) predict exits on UNSEEN students?  5-fold cross-validation grouped by student."""
import json, warnings
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss
import patsy
from common import *
warnings.filterwarnings("ignore")

ev = pd.read_pickle(WORK / "events.pkl")
d = ev[(ev.in_scope == 1) & (ev.state != "Lesson")].copy()
d = add_covariates(d)
d["lat"] = np.log1p(d.latency_s.clip(0, 600).fillna(0))
d["topic"] = d.SubjectId.fillna(-1).astype(int).astype(str)
y = d.is_exit.values; groups = d.UserId.values
base = "C(qs) + C(sess) + C(pfail) + C(yg) + C(block) + wkend + lat"
Xb = sparse.csr_matrix(patsy.dmatrix(base, d, return_type="dataframe").values)
Xs = sparse.csr_matrix(patsy.dmatrix("0 + C(state)", d, return_type="dataframe").values)
Xt = OneHotEncoder(handle_unknown="ignore").fit_transform(d[["topic"]])
# state x question-position interaction (the exit risk after a failed retry shrinks across the quiz)
Xi = sparse.csr_matrix(patsy.dmatrix("0 + C(state):C(qs)", d, return_type="dataframe").values)
models = {"M0 context only": Xb, "M1 + event type": sparse.hstack([Xb, Xs]).tocsr(),
          "M2 + event type x position": sparse.hstack([Xb, Xs, Xi]).tocsr(),
          "M3 + topic": sparse.hstack([Xb, Xs, Xi, Xt]).tocsr()}
rows = []
folds = list(GroupKFold(n_splits=5).split(Xb, y, groups))
for name, X in models.items():
    auc, ap, ll = [], [], []
    for tr, te in folds:
        m = LogisticRegression(C=1.0, max_iter=300, solver="lbfgs").fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        auc.append(roc_auc_score(y[te], p)); ap.append(average_precision_score(y[te], p)); ll.append(log_loss(y[te], p))
    rows.append({"model": name, "auc_mean": np.mean(auc), "auc_sd": np.std(auc), "avg_precision": np.mean(ap), "ap_sd": np.std(ap),
                 "log_loss": np.mean(ll), "ll_sd": np.std(ll)})
    print(rows[-1])
res = pd.DataFrame(rows); res["baseline_rate"] = y.mean(); res["n_events"] = len(y); res["n_students"] = len(np.unique(groups))
res.to_csv(TAB / "model_cv.csv", index=False); print(res.round(4).to_string())
