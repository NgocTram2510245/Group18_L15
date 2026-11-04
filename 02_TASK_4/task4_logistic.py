"""
Task 4 - Hồi quy Logistic nhị phân (Normal=0 / Attack=1) trên NSL-KDD.


Quy ước chống rò rỉ (theo 00_DU_LIEU_CHUNG/QUY_UOC_DU_LIEU.md):
  * Mọi bước có học tham số (one-hot, scaler, chọn C) chỉ fit trên Train.
  * Test chỉ được dùng MỘT lần ở giai đoạn đánh giá cuối.
  * attack_name, attack_category, label_binary, difficulty không bao giờ là input.
    difficulty / attack_category / attack_name chỉ dùng SAU dự đoán để phân tích lỗi.

Đầu ra:
  02_TASK_4/outputs/*.csv, summary.json
  02_TASK_4/figures/task4_*.pdf (+ .png để xem nhanh)
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 0. Đường dẫn và hằng số
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "00_DU_LIEU_CHUNG" / "data" / "processed"
OUT = Path(__file__).resolve().parent / "outputs"
FIG = Path(__file__).resolve().parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

SEED = 18
ALPHA = 0.05
THRESHOLD = 0.5
CATEGORY_ORDER = ["Normal", "DoS", "Probe", "R2L", "U2R"]

TARGET = "label_binary"
NON_PREDICTORS = ["attack_name", "difficulty", "attack_category", "label_binary"]
CATEGORICAL = ["protocol_type", "service", "flag"]
DROPPED_CONSTANT = ["num_outbound_cmds"]  # hằng số ở cả Train và Test

BINARY = ["land", "logged_in", "root_shell", "su_attempted", "is_host_login", "is_guest_login"]
RATE = [
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]
COUNT = [
    "duration", "src_bytes", "dst_bytes", "wrong_fragment", "urgent", "hot",
    "num_failed_logins", "num_compromised", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "count", "srv_count",
    "dst_host_count", "dst_host_srv_count",
]

# Mô hình A (diễn giải): tập biến gọn, chọn theo ý nghĩa nghiên cứu, không có cặp |r|>0,95.
MODEL_A_CONT = [
    "src_bytes", "dst_bytes", "count", "dst_host_srv_count",   # log1p + chuẩn hóa
    "serror_rate", "rerror_rate", "same_srv_rate", "diff_srv_rate",
    "dst_host_same_src_port_rate",                              # chuẩn hóa
]
# logged_in đã thử đưa vào nhưng VIF = 12,4 (kéo serror_rate lên 12,1) -> loại để max VIF < 10.
MODEL_A_BIN: list[str] = []                                     # biến 0/1 giữ nguyên thang
MODEL_A_PROTO_REF = "tcp"                                       # mức tham chiếu



# Tiện ích
def wilson_ci(k: int, n: int, alpha: float = ALPHA) -> tuple[float, float]:
    """Khoảng tin cậy Wilson cho tỷ lệ k/n."""
    if n == 0:
        return (np.nan, np.nan)
    z = stats.norm.ppf(1 - alpha / 2)
    p = k / n
    den = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (centre - half, centre + half)


def metrics_block(y_true, y_pred, y_prob, name: str) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = len(y_true)
    acc_ci = wilson_ci(tp + tn, n)
    prec_ci = wilson_ci(tp, tp + fp)
    rec_ci = wilson_ci(tp, tp + fn)
    spec_ci = wilson_ci(tn, tn + fp)
    return {
        "set": name, "n": int(n), "TP": int(tp), "TN": int(tn), "FP": int(fp), "FN": int(fn),
        "accuracy": accuracy_score(y_true, y_pred),
        "accuracy_ci_low": acc_ci[0], "accuracy_ci_high": acc_ci[1],
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "precision_ci_low": prec_ci[0], "precision_ci_high": prec_ci[1],
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "recall_ci_low": rec_ci[0], "recall_ci_high": rec_ci[1],
        "specificity": tn / (tn + fp) if (tn + fp) else np.nan,
        "specificity_ci_low": spec_ci[0], "specificity_ci_high": spec_ci[1],
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "fpr": fp / (fp + tn) if (fp + tn) else np.nan,
    }


def log1p_frame(x):
    return np.log1p(x)


# --------------------------------------------------------------------------- #
# 1. Đọc dữ liệu và kiểm tra hợp đồng dữ liệu
# --------------------------------------------------------------------------- #
train = pd.read_csv(DATA / "train_clean.csv")
test = pd.read_csv(DATA / "test_clean.csv")
assert train.shape == (125973, 45) and test.shape == (22544, 45), "Sai kích thước dữ liệu"
for df in (train, test):
    assert df[TARGET].isin([0, 1]).all()
    assert df.isna().sum().sum() == 0
    assert set(df["attack_category"]).issubset(CATEGORY_ORDER)

FEATURES = [c for c in train.columns if c not in NON_PREDICTORS + DROPPED_CONSTANT]
assert set(FEATURES) == set(CATEGORICAL + BINARY + RATE + COUNT), "Phân nhóm biến chưa khớp"
assert len(FEATURES) == 40

X_train, y_train = train[FEATURES], train[TARGET].to_numpy()
X_test, y_test = test[FEATURES], test[TARGET].to_numpy()

seen_attacks = set(train.loc[train[TARGET] == 1, "attack_name"])
test_unseen_mask = (test[TARGET] == 1) & ~test["attack_name"].isin(seen_attacks)

summary: dict = {
    "n_train": int(len(train)), "n_test": int(len(test)),
    "train_attack": int(y_train.sum()), "train_normal": int((1 - y_train).sum()),
    "test_attack": int(y_test.sum()), "test_normal": int((1 - y_test).sum()),
    "n_features_raw": len(FEATURES),
    "test_unseen_attack_names": sorted(set(test.loc[test_unseen_mask, "attack_name"])),
    "test_unseen_attack_rows": int(test_unseen_mask.sum()),
    "seed": SEED, "threshold": THRESHOLD, "alpha": ALPHA,
}

# Cặp tương quan |r| > 0,95 trên Train (sau log1p) - để minh bạch đa cộng tuyến
num_cols = BINARY + RATE + COUNT
num_tr = X_train[num_cols].copy()
num_tr[COUNT] = np.log1p(num_tr[COUNT])
corr = num_tr.corr()
pairs = []
for i, a in enumerate(num_cols):
    for b in num_cols[i + 1:]:
        r = corr.loc[a, b]
        if abs(r) > 0.95:
            pairs.append({"var1": a, "var2": b, "r": r})
pd.DataFrame(pairs).sort_values("r", key=abs, ascending=False).to_csv(
    OUT / "high_corr_pairs_train.csv", index=False)
summary["n_high_corr_pairs"] = len(pairs)

# --------------------------------------------------------------------------- #
# 2. Mô hình B - dự đoán (toàn bộ 40 đặc trưng, L2, chọn C bằng CV trên Train)
# --------------------------------------------------------------------------- #
preprocess = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", drop=None, sparse_output=False), CATEGORICAL),
        ("cnt", Pipeline([("log", FunctionTransformer(log1p_frame, feature_names_out="one-to-one")),
                          ("sc", StandardScaler())]), COUNT),
        ("num", StandardScaler(), RATE + BINARY),
    ],
    verbose_feature_names_out=False,
)
pipe = Pipeline([
    ("prep", preprocess),
    ("clf", LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000)),
])
grid = {"clf__C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
N_SPLITS = cv.get_n_splits()


def one_se_rule(cv_results) -> int:
    """Chọn C nhỏ nhất (điều chuẩn mạnh nhất) có F1-CV trung bình >= F1 tốt nhất - 1 SE.

    F1-CV đạt "bình nguyên" khi C >= 10; quy tắc 1-SE tránh việc C tốt nhất
    luôn trôi về biên lưới và ưu tiên mô hình đơn giản hơn. Chỉ dùng dữ liệu Train.
    """
    mean = np.asarray(cv_results["mean_test_f1"])
    se = np.asarray(cv_results["std_test_f1"]) / np.sqrt(N_SPLITS)
    best = int(np.argmax(mean))
    ok = np.where(mean >= mean[best] - se[best])[0]
    Cs = np.asarray(cv_results["param_clf__C"], dtype=float)
    return int(ok[np.argmin(Cs[ok])])


search = GridSearchCV(pipe, grid, scoring={"f1": "f1", "accuracy": "accuracy", "roc_auc": "roc_auc"},
                      refit=one_se_rule, cv=cv, n_jobs=-1, return_train_score=False)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    search.fit(X_train, y_train)

cv_res = pd.DataFrame(search.cv_results_)[[
    "param_clf__C", "mean_test_f1", "std_test_f1", "mean_test_accuracy",
    "std_test_accuracy", "mean_test_roc_auc", "std_test_roc_auc"]]
cv_res.to_csv(OUT / "cv_results_modelB.csv", index=False)
best_idx = int(search.best_index_)
best_C = float(cv_res["param_clf__C"].iloc[best_idx])
model_b = search.best_estimator_
feat_names = list(model_b.named_steps["prep"].get_feature_names_out())
summary["modelB_best_C"] = best_C
summary["modelB_n_inputs"] = len(feat_names)
summary["modelB_cv_f1_selected"] = float(cv_res["mean_test_f1"].iloc[best_idx])
summary["modelB_cv_f1_max"] = float(cv_res["mean_test_f1"].max())
summary["modelB_cv_C_argmax"] = float(cv_res["param_clf__C"].iloc[int(cv_res["mean_test_f1"].idxmax())])

coef_b = pd.DataFrame({
    "feature": feat_names,
    "beta_std": model_b.named_steps["clf"].coef_.ravel(),
})
coef_b["odds_ratio"] = np.exp(coef_b["beta_std"])
coef_b = coef_b.sort_values("beta_std", key=abs, ascending=False)
coef_b.to_csv(OUT / "coefficients_modelB.csv", index=False)
summary["modelB_intercept"] = float(model_b.named_steps["clf"].intercept_[0])

# --------------------------------------------------------------------------- #
# 3. Mô hình A - diễn giải (statsmodels Logit, tập biến gọn)
# --------------------------------------------------------------------------- #
def design_A(df: pd.DataFrame, scaler_stats: dict | None = None):
    Z = pd.DataFrame(index=df.index)
    for c in MODEL_A_CONT:
        Z[c] = np.log1p(df[c]) if c in COUNT else df[c].astype(float)
    if scaler_stats is None:
        scaler_stats = {c: (Z[c].mean(), Z[c].std(ddof=0)) for c in MODEL_A_CONT}
    for c in MODEL_A_CONT:
        mu, sd = scaler_stats[c]
        Z[c] = (Z[c] - mu) / sd
    for c in MODEL_A_BIN:
        Z[c] = df[c].astype(float)
    for lvl in sorted(train["protocol_type"].unique()):
        if lvl != MODEL_A_PROTO_REF:
            Z[f"protocol_type_{lvl}"] = (df["protocol_type"] == lvl).astype(float)
    return sm.add_constant(Z, has_constant="add"), scaler_stats


XA_tr, A_stats = design_A(train)          # tham số chuẩn hóa học từ Train
XA_te, _ = design_A(test, A_stats)

vif = pd.DataFrame({
    "variable": XA_tr.columns[1:],
    "VIF": [variance_inflation_factor(XA_tr.values, i) for i in range(1, XA_tr.shape[1])],
})
vif.to_csv(OUT / "vif_modelA.csv", index=False)
summary["modelA_max_vif"] = float(vif["VIF"].max())

logit_A = sm.Logit(y_train, XA_tr).fit(method="newton", maxiter=200, disp=False)
z = stats.norm.ppf(1 - ALPHA / 2)
coef_a = pd.DataFrame({
    "variable": logit_A.params.index,
    "beta": logit_A.params.values,
    "se": logit_A.bse.values,
    "z": logit_A.tvalues.values,
    "p_value": logit_A.pvalues.values,
})
coef_a["ci_low_beta"] = coef_a["beta"] - z * coef_a["se"]
coef_a["ci_high_beta"] = coef_a["beta"] + z * coef_a["se"]
coef_a["odds_ratio"] = np.exp(coef_a["beta"])
coef_a["or_ci_low"] = np.exp(coef_a["ci_low_beta"])
coef_a["or_ci_high"] = np.exp(coef_a["ci_high_beta"])
coef_a.to_csv(OUT / "coefficients_modelA.csv", index=False)

summary.update({
    "modelA_converged": bool(logit_A.mle_retvals["converged"]),
    "modelA_n_params": int(len(logit_A.params)),
    "modelA_llf": float(logit_A.llf), "modelA_llnull": float(logit_A.llnull),
    "modelA_LR_stat": float(logit_A.llr), "modelA_LR_df": int(logit_A.df_model),
    "modelA_LR_pvalue": float(logit_A.llr_pvalue),
    "modelA_pseudo_R2_McFadden": float(logit_A.prsquared),
    "modelA_AIC": float(logit_A.aic), "modelA_BIC": float(logit_A.bic),
})

# --------------------------------------------------------------------------- #
# 4. Đánh giá (Test dùng một lần cho mỗi mô hình đã chốt)
# --------------------------------------------------------------------------- #
prob_b_tr = model_b.predict_proba(X_train)[:, 1]
prob_b_te = model_b.predict_proba(X_test)[:, 1]
pred_b_tr = (prob_b_tr >= THRESHOLD).astype(int)
pred_b_te = (prob_b_te >= THRESHOLD).astype(int)

prob_a_tr = logit_A.predict(XA_tr)
prob_a_te = logit_A.predict(XA_te)
pred_a_tr = (prob_a_tr >= THRESHOLD).astype(int)
pred_a_te = (prob_a_te >= THRESHOLD).astype(int)

metrics = pd.DataFrame([
    {"model": "B_full_L2", **metrics_block(y_train, pred_b_tr, prob_b_tr, "train")},
    {"model": "B_full_L2", **metrics_block(y_test, pred_b_te, prob_b_te, "test")},
    {"model": "A_interpretive", **metrics_block(y_train, pred_a_tr, prob_a_tr, "train")},
    {"model": "A_interpretive", **metrics_block(y_test, pred_a_te, prob_a_te, "test")},
])
metrics.to_csv(OUT / "metrics.csv", index=False)

# --------------------------------------------------------------------------- #
# 5. Phân tích lỗi (mô hình B trên Test). Nhãn/difficulty chỉ dùng sau dự đoán.
# --------------------------------------------------------------------------- #
te = test[["attack_name", "attack_category", "difficulty", "service", "flag", TARGET]].copy()
te["pred"] = pred_b_te
te["prob"] = prob_b_te
te["correct"] = (te["pred"] == te[TARGET]).astype(int)
tr = train[["attack_name", "attack_category", TARGET]].copy()
tr["pred"] = pred_b_tr


def by_category(df):
    rows = []
    for cat in CATEGORY_ORDER:
        d = df[df["attack_category"] == cat]
        n = len(d)
        if cat == "Normal":
            k = int((d["pred"] == 0).sum())   # đúng = dự đoán Normal
            metric = "specificity"
        else:
            k = int((d["pred"] == 1).sum())   # đúng = phát hiện Attack
            metric = "recall"
        lo, hi = wilson_ci(k, n)
        rows.append({"category": cat, "n": n, "correct": k, "missed": n - k,
                     "metric": metric, "rate": k / n if n else np.nan, "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


cat_te = by_category(te); cat_te.insert(0, "set", "test")
cat_tr = by_category(tr); cat_tr.insert(0, "set", "train")
pd.concat([cat_tr, cat_te]).to_csv(OUT / "recall_by_category.csv", index=False)

att = te[te[TARGET] == 1].copy()
att["seen_in_train"] = att["attack_name"].isin(seen_attacks)
seen_rows = []
for flag_seen, label in [(True, "Đã có trong Train"), (False, "Chỉ có trong Test")]:
    d = att[att["seen_in_train"] == flag_seen]
    k = int(d["pred"].sum())
    lo, hi = wilson_ci(k, len(d))
    seen_rows.append({"group": label, "n": len(d), "detected": k, "missed": len(d) - k,
                      "recall": k / len(d), "ci_low": lo, "ci_high": hi})
seen_df = pd.DataFrame(seen_rows)
seen_df.to_csv(OUT / "recall_seen_vs_unseen.csv", index=False)
fn_total = int(((te[TARGET] == 1) & (te["pred"] == 0)).sum())
summary["fn_total_test"] = fn_total
summary["fn_from_unseen"] = int(seen_df.loc[seen_df["group"] == "Chỉ có trong Test", "missed"].iloc[0])

name_tab = (att.groupby(["attack_name", "attack_category"])
            .agg(n=("pred", "size"), detected=("pred", "sum"))
            .reset_index())
name_tab["missed"] = name_tab["n"] - name_tab["detected"]
name_tab["recall"] = name_tab["detected"] / name_tab["n"]
name_tab["seen_in_train"] = name_tab["attack_name"].isin(seen_attacks)
name_tab = name_tab.sort_values("missed", ascending=False)
name_tab.to_csv(OUT / "missed_by_attack_name.csv", index=False)

fp_df = te[(te[TARGET] == 0) & (te["pred"] == 1)]
normal_te = te[te[TARGET] == 0]
fp_service = (fp_df["service"].value_counts().rename("fp").to_frame()
              .join(normal_te["service"].value_counts().rename("n_normal")))
fp_service["fp_rate"] = fp_service["fp"] / fp_service["n_normal"]
fp_service.head(15).to_csv(OUT / "fp_by_service.csv")
fp_flag = (fp_df["flag"].value_counts().rename("fp").to_frame()
           .join(normal_te["flag"].value_counts().rename("n_normal")))
fp_flag["fp_rate"] = fp_flag["fp"] / fp_flag["n_normal"]
fp_flag.to_csv(OUT / "fp_by_flag.csv")

bins = [-1, 10, 15, 20, 21]
labels = ["0-10", "11-15", "16-20", "21"]
te["difficulty_bin"] = pd.cut(te["difficulty"], bins=bins, labels=labels)
diff_rows = []
for b in labels:
    d = te[te["difficulty_bin"] == b]
    k = int(d["correct"].sum())
    lo, hi = wilson_ci(k, len(d))
    diff_rows.append({"difficulty": b, "n": len(d), "n_attack": int(d[TARGET].sum()),
                      "correct": k, "accuracy": k / len(d) if len(d) else np.nan,
                      "ci_low": lo, "ci_high": hi})
pd.DataFrame(diff_rows).to_csv(OUT / "accuracy_by_difficulty.csv", index=False)

# Mức xác suất của các ca bỏ sót: FN gần ngưỡng hay rất tự tin?
fn_prob = te.loc[(te[TARGET] == 1) & (te["pred"] == 0), "prob"]
summary["fn_prob_median"] = float(fn_prob.median())
summary["fn_prob_share_below_0_1"] = float((fn_prob < 0.1).mean())

# --------------------------------------------------------------------------- #
# 6. Hình
# --------------------------------------------------------------------------- #
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "savefig.bbox": "tight"})

# 6.1 Confusion matrix Test (mô hình B)
cm = confusion_matrix(y_test, pred_b_te, labels=[0, 1])
fig, ax = plt.subplots(figsize=(4.6, 3.9))
im = ax.imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        pct = cm[i, j] / cm[i].sum() * 100
        ax.text(j, i, f"{cm[i, j]:,}".replace(",", ".") + f"\n({pct:.1f}%)".replace(".", ","),
                ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["Normal (0)", "Attack (1)"])
ax.set_yticks([0, 1], ["Normal (0)", "Attack (1)"])
ax.set_xlabel("Nhãn dự đoán")
ax.set_ylabel("Nhãn thực tế")
fig.colorbar(im, ax=ax, fraction=0.046)
fig.tight_layout()
fig.savefig(FIG / "task4_confusion_test.pdf"); fig.savefig(FIG / "task4_confusion_test.png", dpi=150)
plt.close(fig)

# 6.2 ROC Train vs Test (mô hình B) + mô hình A trên Test
fig, ax = plt.subplots(figsize=(4.8, 4.2))
for yt, pp, lab, ls in [(y_train, prob_b_tr, "Mô hình B – Train", "-"),
                        (y_test, prob_b_te, "Mô hình B – Test", "-"),
                        (y_test, prob_a_te, "Mô hình A – Test", "--")]:
    fpr, tpr, _ = roc_curve(yt, pp)
    auc = roc_auc_score(yt, pp)
    ax.plot(fpr, tpr, ls, label=f"{lab} (AUC = {auc:.3f})".replace(".", ","))
ax.plot([0, 1], [0, 1], ":", color="grey")
ax.set_xlabel("Tỷ lệ dương tính giả (1 − Specificity)")
ax.set_ylabel("Tỷ lệ dương tính thật (Recall)")
ax.legend(loc="lower right", fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "task4_roc.pdf"); fig.savefig(FIG / "task4_roc.png", dpi=150)
plt.close(fig)

# 6.3 Forest plot OR mô hình A
ca = coef_a[coef_a["variable"] != "const"].iloc[::-1]
fig, ax = plt.subplots(figsize=(6.0, 4.2))
ax.errorbar(ca["odds_ratio"], range(len(ca)),
            xerr=[ca["odds_ratio"] - ca["or_ci_low"], ca["or_ci_high"] - ca["odds_ratio"]],
            fmt="o", color="#123452", capsize=3)
ax.axvline(1, color="grey", ls="--")
ax.set_xscale("log")
ax.set_yticks(range(len(ca)), ca["variable"])
ax.set_xlabel("Tỷ số odds (thang log) và CI 95%")
fig.tight_layout()
fig.savefig(FIG / "task4_or_modelA.pdf"); fig.savefig(FIG / "task4_or_modelA.png", dpi=150)
plt.close(fig)

# 6.4 Recall theo nhóm (Train vs Test) với thanh sai số Wilson
fig, ax = plt.subplots(figsize=(6.0, 3.8))
w = 0.38
x = np.arange(len(CATEGORY_ORDER))
for k, (d, lab) in enumerate([(cat_tr, "Train"), (cat_te, "Test")]):
    d = d.set_index("category").loc[CATEGORY_ORDER]
    ax.bar(x + (k - 0.5) * w, d["rate"], w, label=lab,
           yerr=[d["rate"] - d["ci_low"], d["ci_high"] - d["rate"]], capsize=3)
ax.set_xticks(x, ["Normal\n(Specificity)", "DoS", "Probe", "R2L", "U2R"])
ax.set_ylabel("Tỷ lệ phân loại đúng")
ax.set_ylim(0, 1.05)
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "task4_recall_by_category.pdf"); fig.savefig(FIG / "task4_recall_by_category.png", dpi=150)
plt.close(fig)

# 6.5 Phân bố xác suất dự đoán trên Test theo nhãn thật
fig, ax = plt.subplots(figsize=(6.0, 3.6))
ax.hist(prob_b_te[y_test == 0], bins=50, alpha=0.6, label="Normal thực tế")
ax.hist(prob_b_te[y_test == 1], bins=50, alpha=0.6, label="Attack thực tế")
ax.axvline(THRESHOLD, color="k", ls="--", lw=1)
ax.set_xlabel("Xác suất dự đoán Attack $\\hat{p}$")
ax.set_ylabel("Số quan sát")
ax.legend()
fig.tight_layout()
fig.savefig(FIG / "task4_prob_hist_test.pdf"); fig.savefig(FIG / "task4_prob_hist_test.png", dpi=150)
plt.close(fig)

# --------------------------------------------------------------------------- #
# 7. Ghi tóm tắt
# --------------------------------------------------------------------------- #
with open(OUT / "summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2, default=float)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)
print(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
print("\nCV:\n", cv_res.round(4).to_string(index=False))
print("\nMetrics:\n", metrics.round(4).to_string(index=False))
print("\nModel A:\n", coef_a.round(4).to_string(index=False))
print("\nVIF:\n", vif.round(2).to_string(index=False))
print("\nRecall by category:\n", pd.concat([cat_tr, cat_te]).round(4).to_string(index=False))
print("\nSeen vs unseen:\n", seen_df.round(4).to_string(index=False))
print("\nTop missed names:\n", name_tab.head(15).round(4).to_string(index=False))
print("\nFP by service:\n", fp_service.head(10).round(4).to_string())
print("\nFP by flag:\n", fp_flag.round(4).to_string())
print("\nAcc by difficulty:\n", pd.DataFrame(diff_rows).round(4).to_string(index=False))
print("\nTop |beta| model B:\n", coef_b.head(15).round(4).to_string(index=False))
print("\nHigh corr pairs:\n", pd.DataFrame(pairs).round(4).to_string(index=False))
