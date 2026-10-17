"""
run_all.py - ONE command reproduces every headline number and figure.

    python -m src.run_all            # full run (~10-15 min on a laptop CPU)
    python -m src.run_all --quick    # smoke test on a thin grid (~2 min)

Outputs:  results/*.csv|json|md   (all numbers)    report/figures/*.pdf|png  (all figures)
Everything is seeded (SEED in data_prep.py); the model-selection protocol is
  fit on TRAIN -> choose hyper-parameters on VALIDATION -> report TEST once.
"""

from __future__ import annotations

import json
import os
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    # Executed directly as a script: python src/run_all.py
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    # Ensure working directory is project root so relative paths (data/..., results/...) work
    if not os.path.exists("data") and os.path.exists(os.path.join(project_root, "data")):
        os.chdir(project_root)
    from src import data_prep as dp
    from src import evaluate as ev
    from src.decision_tree import DecisionTree
    from src.svm import PegasosSVM, RFFPegasosSVM
    from src.bonuses import RidgeRegression, RandomForest, PCA, KMeans
    from src import make_report_tables
else:
    # Executed as a module: python -m src.run_all
    from . import data_prep as dp
    from . import evaluate as ev
    from .decision_tree import DecisionTree
    from .svm import PegasosSVM, RFFPegasosSVM
    from .bonuses import RidgeRegression, RandomForest, PCA, KMeans
    from . import make_report_tables

SEED = dp.SEED
QUICK = "--quick" in sys.argv
FIG_DIR, RES_DIR = "report/figures", "results"
EPOCHS, BATCH = 30, 256                       # minimum Pegasos budget (epochs) and mini-batch size


def epochs_linear(lam):
    """Pegasos needs ~1/lambda steps to converge, so the epoch budget grows as lambda shrinks."""
    return int(min(3000, max(EPOCHS, 0.01 / lam)))


def epochs_rff(lam):
    """Same idea for the RBF (random-Fourier-feature) SVM, with a smaller cap (each epoch is 500-d)."""
    return int(min(300, max(EPOCHS, 1e-3 / lam)))
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "axes.grid": True, "grid.alpha": 0.25,
                     "figure.dpi": 120, "savefig.bbox": "tight"})


# ------------------------------------------------------------------ utilities ----
def savefig(fig, name):
    fig.savefig(f"{FIG_DIR}/{name}.pdf")
    fig.savefig(f"{FIG_DIR}/{name}.png", dpi=200)
    plt.close(fig)
    print(f"   [fig] {name}")


def timed(fn, *a, **k):
    t = time.perf_counter()
    out = fn(*a, **k)
    return out, time.perf_counter() - t


def cls_metrics(y, pred, score):
    return ev.classification_report(y, pred, score)


def tree_cls_eval(m, X, y):
    return cls_metrics(y, m.predict(X), m.predict_proba(X)[:, 1])


def pick(tbl, score, size, higher=True, tol=0.002, ascending=True):
    """
    Validation-only selection rule: among all configs whose validation score is within
    `tol` of the best, take the SIMPLEST (smallest `size`, or largest if ascending=False).
    Prefers smaller trees / stronger regularisation when validation cannot tell them apart.
    """
    ok = tbl[tbl[score] >= tbl[score].max() - tol] if higher else tbl[tbl[score] <= tbl[score].min() + tol]
    return ok.sort_values(size, ascending=ascending).iloc[0]


def banner(s):
    print(f"\n{'=' * 78}\n{s}\n{'=' * 78}")


def shared_top_structure(ours, sk, max_depth=3):
    """Walk both trees together; count identical (feature, threshold) nodes down to `max_depth`."""
    t = sk.tree_
    matched = total = 0
    stack = [(ours.root, 0, 0)]
    while stack:
        nd, i, d = stack.pop()
        if nd.is_leaf() or d >= max_depth:
            continue
        total += 1
        if t.children_left[i] == -1:
            continue
        if nd.feature == t.feature[i] and np.isclose(nd.threshold, t.threshold[i], rtol=1e-5, atol=1e-6):
            matched += 1
            stack.append((nd.left, t.children_left[i], d + 1))
            stack.append((nd.right, t.children_right[i], d + 1))
    return matched, total


# ------------------------------------------------------------------------ main ----
def main() -> None:
    from sklearn.linear_model import SGDClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    np.random.seed(SEED)
    R: dict = {"seed": SEED, "quick": QUICK}          # everything we report goes in here
    t_start = time.time()

    # =========================================================== 1. data (Problem 1)
    banner("1. DATA: load, clean, features, split")
    raw = dp.load_raw()
    df = dp.clean(raw)
    log = df.attrs["cleaning_log"]
    for name, b, a in log:
        print(f"   {name:<66s}{b:>8d} -> {a:>8d}   (-{b - a})")
    R["cleaning_log"] = [{"step": n, "before": b, "after": a} for n, b, a in log]
    X, y, names = dp.make_features(df)
    ylog = np.log(y)
    bins = np.digitize(ylog, np.quantile(ylog, np.linspace(0, 1, 11)[1:-1]))      # price deciles
    tr, va, te = dp.split_indices(len(y), seed=SEED, strata=bins)
    ytier_tr, THR = dp.make_tier_label(y[tr])                                      # TRAIN median only
    ytier_va, _ = dp.make_tier_label(y[va], THR)
    ytier_te, _ = dp.make_tier_label(y[te], THR)
    print(f"   rows={len(y)}  features={X.shape[1]}  split train/val/test = {len(tr)}/{len(va)}/{len(te)}")
    print(f"   tier threshold (train median) = {THR:,.0f} AZN; premium share train/val/test = "
          f"{ytier_tr.mean():.3f}/{ytier_va.mean():.3f}/{ytier_te.mean():.3f}")
    R["data"] = {"rows": len(y), "n_features": X.shape[1], "n_train": len(tr), "n_val": len(va),
                 "n_test": len(te), "tier_threshold_azn": THR,
                 "premium_share": {"train": float(ytier_tr.mean()), "val": float(ytier_va.mean()),
                                   "test": float(ytier_te.mean())},
                 "price_median": float(np.median(y)), "price_mean": float(y.mean()),
                 "price_max": float(y.max()), "missing_share": dict(zip(names, np.isnan(X).mean(0).round(4).tolist()))}
    np.savez(f"{RES_DIR}/split_indices.npz", train=tr, val=va, test=te)

    # ---- EDA figure
    d = df.assign(log_price=ylog, tier=np.nan)
    d.loc[d.index[tr], "tier"] = ytier_tr
    fig, ax = plt.subplots(2, 2, figsize=(7.2, 5.6))
    ax[0, 0].hist(np.clip(y, 0, np.quantile(y, 0.99)) / 1e3, bins=60, color="#4C72B0")
    ax[0, 0].set(title="Price (clipped at 99th pct)", xlabel="price, thousand AZN", ylabel="listings")
    ax[0, 1].hist(ylog, bins=60, color="#55A868"); ax[0, 1].axvline(np.log(THR), color="k", ls="--", lw=1)
    ax[0, 1].set(title="log(price) with tier threshold (train median)", xlabel="log(price)")
    samp = np.random.default_rng(SEED).choice(len(y), 8000, replace=False)
    for c, col in zip(["new_build", "old_stock", "house_villa"], ["#4C72B0", "#DD8452", "#55A868"]):
        s = samp[df["category"].to_numpy()[samp] == c]
        ax[1, 0].scatter(df["area_m2"].to_numpy()[s], y[s], s=2, alpha=0.4, label=c, color=col)
    ax[1, 0].set(xscale="log", yscale="log", title="Area vs price", xlabel="area, m$^2$", ylabel="price, AZN")
    ax[1, 0].legend(markerscale=4, fontsize=7)
    s = np.random.default_rng(SEED).choice(len(y), 15000, replace=False)
    sc = ax[1, 1].scatter(df["lng"].to_numpy()[s], df["lat"].to_numpy()[s], c=ylog[s], s=2, cmap="viridis")
    ax[1, 1].set(title="Location effect (colour = log price)", xlabel="longitude", ylabel="latitude",
                 xlim=(49.6, 50.2), ylim=(40.25, 40.55))
    fig.colorbar(sc, ax=ax[1, 1], fraction=0.046)
    fig.tight_layout(); savefig(fig, "fig_eda_overview")

    fig, ax = plt.subplots(1, 3, figsize=(8.5, 2.7))
    d["rooms"].value_counts().sort_index().plot.bar(ax=ax[0], color="#4C72B0", rot=0)
    ax[0].set(title="Rooms distribution (capped at 10)", xlabel="rooms", ylabel="listings")
    d["category"].value_counts().plot.bar(ax=ax[1], color=["#4C72B0", "#DD8452", "#55A868"], rot=20)
    ax[1].set(title="Property categories", ylabel="listings")
    split_counts = pd.Series([len(tr), len(va), len(te)], index=["train", "val", "test"])
    split_counts.plot.bar(ax=ax[2], color=["#4C72B0", "#DD8452", "#55A868"], rot=0)
    ax[2].set(title="Stratified split row counts", ylabel="listings")
    fig.tight_layout(); savefig(fig, "fig_eda_distributions")

    # ---- preprocessing (fit on TRAIN only)
    Xtr, Xva, Xte = dp.impute(X[tr], X[va], X[te])                           # for the trees
    Ztr, Zva, Zte = dp.standardize(*[dp.log_transform(a, names) for a in (Xtr, Xva, Xte)])   # for SVMs
    ylog_tr, ylog_va, ylog_te = ylog[tr], ylog[va], ylog[te]

    # =========================================================== 4a. tree study
    banner("2. TREE: hyper-parameter study (validation only)")
    depths = [1, 2, 3, 4, 6, 8, 10, 12, 14, 18, None] if not QUICK else [2, 6, 10, None]
    rows = []
    for crit in ["gini", "entropy"]:
        for dep in depths:
            m, tf = timed(lambda: DecisionTree("classification", crit, max_depth=dep).fit(Xtr, ytier_tr))
            a, b = tree_cls_eval(m, Xtr, ytier_tr), tree_cls_eval(m, Xva, ytier_va)
            rows.append(dict(criterion=crit, max_depth=-1 if dep is None else dep, leaves=m.get_n_leaves(),
                             fit_s=tf, train_f1=a["f1"], val_f1=b["f1"], train_acc=a["accuracy"],
                             val_acc=b["accuracy"], val_auc=b["roc_auc"]))
        print(f"   {crit}: done")
    T_depth = pd.DataFrame(rows); T_depth.to_csv(f"{RES_DIR}/tree_cls_depth_sweep.csv", index=False)
    best = pick(T_depth, "val_f1", "leaves")
    best_crit, best_depth = best["criterion"], (None if best["max_depth"] < 0 else int(best["max_depth"]))
    print(f"   best (criterion, depth) by val F1: {best_crit}, {best_depth}  (val F1={best['val_f1']:.4f})")

    leaves_grid = [1, 3, 5, 10, 20, 50, 100, 200] if not QUICK else [1, 20, 100]
    rows = []
    for msl in leaves_grid:
        m = DecisionTree("classification", best_crit, max_depth=best_depth, min_samples_leaf=msl).fit(Xtr, ytier_tr)
        a, b = tree_cls_eval(m, Xtr, ytier_tr), tree_cls_eval(m, Xva, ytier_va)
        rows.append(dict(min_samples_leaf=msl, leaves=m.get_n_leaves(), train_f1=a["f1"], val_f1=b["f1"],
                         val_acc=b["accuracy"], val_auc=b["roc_auc"]))
    T_leaf = pd.DataFrame(rows); T_leaf.to_csv(f"{RES_DIR}/tree_cls_leaf_sweep.csv", index=False)
    bl = pick(T_leaf, "val_f1", "leaves")
    CLS_CFG = dict(criterion=best_crit, max_depth=best_depth, min_samples_leaf=int(bl["min_samples_leaf"]))
    print(f"   FINAL classification tree config: {CLS_CFG}")

    # regression tree (Task A) - validation RMSE on log price
    rows = []
    for dep in depths:
        m = DecisionTree("regression", "mse", max_depth=dep).fit(Xtr, ylog_tr)
        rows.append(dict(max_depth=-1 if dep is None else dep, leaves=m.get_n_leaves(),
                         train_rmse=ev.rmse(ylog_tr, m.predict(Xtr)), val_rmse=ev.rmse(ylog_va, m.predict(Xva)),
                         val_r2=ev.r2(ylog_va, m.predict(Xva))))
    T_rdepth = pd.DataFrame(rows); T_rdepth.to_csv(f"{RES_DIR}/tree_reg_depth_sweep.csv", index=False)
    rb = pick(T_rdepth, "val_rmse", "leaves", higher=False)
    reg_depth = None if rb["max_depth"] < 0 else int(rb["max_depth"])
    rows = []
    for msl in leaves_grid:
        m = DecisionTree("regression", "mse", max_depth=reg_depth, min_samples_leaf=msl).fit(Xtr, ylog_tr)
        rows.append(dict(min_samples_leaf=msl, leaves=m.get_n_leaves(), train_rmse=ev.rmse(ylog_tr, m.predict(Xtr)),
                         val_rmse=ev.rmse(ylog_va, m.predict(Xva))))
    T_rleaf = pd.DataFrame(rows); T_rleaf.to_csv(f"{RES_DIR}/tree_reg_leaf_sweep.csv", index=False)
    rl = pick(T_rleaf, "val_rmse", "leaves", higher=False)
    REG_CFG = dict(criterion="mse", max_depth=reg_depth, min_samples_leaf=int(rl["min_samples_leaf"]))
    print(f"   FINAL regression tree config: {REG_CFG}")

    # "every stopping rule actually changes the tree"
    base = DecisionTree("classification", "gini").fit(Xtr, ytier_tr)
    effect = [("none (full tree)", base)]
    for label, kw in [("max_depth=5", dict(max_depth=5)), ("min_samples_split=200", dict(min_samples_split=200)),
                      ("min_samples_leaf=50", dict(min_samples_leaf=50)),
                      ("min_impurity_decrease=1e-3", dict(min_impurity_decrease=1e-3))]:
        effect.append((label, DecisionTree("classification", "gini", **kw).fit(Xtr, ytier_tr)))
    T_eff = pd.DataFrame([dict(setting=l, depth=m.get_depth(), leaves=m.get_n_leaves(),
                               val_f1=tree_cls_eval(m, Xva, ytier_va)["f1"]) for l, m in effect])
    T_eff.to_csv(f"{RES_DIR}/tree_stopping_rules.csv", index=False)
    print(T_eff.to_string(index=False))

    # learning curves
    fr = [0.05, 0.1, 0.2, 0.4, 0.7, 1.0] if not QUICK else [0.1, 1.0]
    rng = np.random.default_rng(SEED); perm = rng.permutation(len(tr)); lc = []
    for f in fr:
        i = perm[:int(f * len(tr))]
        mc = DecisionTree("classification", **CLS_CFG).fit(Xtr[i], ytier_tr[i])
        mr = DecisionTree("regression", **REG_CFG).fit(Xtr[i], ylog_tr[i])
        lc.append(dict(n=len(i), cls_train=tree_cls_eval(mc, Xtr[i], ytier_tr[i])["f1"], cls_val=tree_cls_eval(mc, Xva, ytier_va)["f1"],
                       reg_train=ev.rmse(ylog_tr[i], mr.predict(Xtr[i])), reg_val=ev.rmse(ylog_va, mr.predict(Xva))))
    T_lc = pd.DataFrame(lc); T_lc.to_csv(f"{RES_DIR}/tree_learning_curve.csv", index=False)

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for crit, col in [("gini", "#4C72B0"), ("entropy", "#DD8452")]:
        s = T_depth[T_depth.criterion == crit]; xx = np.where(s.max_depth < 0, 26, s.max_depth)
        ax[0].plot(xx, s.train_f1, "--", color=col, label=f"{crit} train")
        ax[0].plot(xx, s.val_f1, "-o", ms=3, color=col, label=f"{crit} val")
    ax[0].set(title="Task B: depth sweep", xlabel="max_depth (26 = unlimited)", ylabel="F1 (premium)")
    ax[0].legend(fontsize=6)
    xx = np.where(T_rdepth.max_depth < 0, 26, T_rdepth.max_depth)
    ax[1].plot(xx, T_rdepth.train_rmse, "--", color="#55A868", label="train")
    ax[1].plot(xx, T_rdepth.val_rmse, "-o", ms=3, color="#55A868", label="val")
    ax[1].set(title="Task A: depth sweep", xlabel="max_depth (26 = unlimited)", ylabel="RMSE of log(price)"); ax[1].legend(fontsize=6)
    ax[2].plot(T_lc.n, T_lc.cls_train, "--", color="#4C72B0", label="B train F1"); ax[2].plot(T_lc.n, T_lc.cls_val, "-o", ms=3, color="#4C72B0", label="B val F1")
    ax[2].set(title="Learning curve (final configs)", xlabel="training rows", ylabel="F1", xscale="log"); ax[2].legend(fontsize=6)
    fig.tight_layout(); savefig(fig, "fig_tree_curves")

    fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.3))
    ax[0].semilogx(T_leaf.min_samples_leaf, T_leaf.train_f1, "--", label="train"); ax[0].semilogx(T_leaf.min_samples_leaf, T_leaf.val_f1, "-o", ms=3, label="val")
    ax[0].set(title="Task B: min_samples_leaf", xlabel="min_samples_leaf", ylabel="F1"); ax[0].legend(fontsize=6)
    ax[1].semilogx(T_rleaf.min_samples_leaf, T_rleaf.train_rmse, "--", label="train"); ax[1].semilogx(T_rleaf.min_samples_leaf, T_rleaf.val_rmse, "-o", ms=3, label="val")
    ax[1].set(title="Task A: min_samples_leaf", xlabel="min_samples_leaf", ylabel="RMSE log"); ax[1].legend(fontsize=6)
    fig.tight_layout(); savefig(fig, "fig_tree_leaf")

    # =========================================================== 4b. SVM study
    banner("3. SVM: lambda / gamma / schedule study (validation only)")
    lams = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1] if not QUICK else [1e-5, 1e-3, 1e-1]
    rows, hist = [], {}
    for lam in lams:
        m = PegasosSVM(lambda_=lam, n_epochs=epochs_linear(lam), batch_size=BATCH, random_state=SEED).fit(Ztr, ytier_tr)
        sc_va = m.decision_function(Zva); pr = m.predict(Zva)
        hist[lam] = m.history_
        gap = abs(m.last_iterate_objective_ - m.objective_) / m.objective_
        rows.append(dict(lambda_=lam, epochs=epochs_linear(lam), w_norm=m.w_norm_, margin_width=m.margin_width_, n_support=len(m.support_),
                         train_objective=m.objective_, last_iter_gap=gap, converged=bool(gap < 0.05), val_f1=ev.precision_recall_f1(ytier_va, pr)[2],
                         val_acc=ev.accuracy(ytier_va, pr), val_auc=ev.roc_auc(ytier_va, sc_va)))
    T_lam = pd.DataFrame(rows); T_lam.to_csv(f"{RES_DIR}/svm_lambda_sweep.csv", index=False)
    print(T_lam.to_string(index=False))
    LIN_LAM = float(pick(T_lam, "val_f1", "lambda_", ascending=False)["lambda_"])   # strongest lambda within 0.002 F1

    sched = {}
    for name, kw in [("pegasos 1/(lambda t)", dict(lr_schedule="pegasos")),
                     ("invscaling 0.5/sqrt(t)", dict(lr_schedule="invscaling", eta0=0.5)),
                     ("constant 0.01", dict(lr_schedule="constant", eta0=0.01))]:
        m = PegasosSVM(lambda_=LIN_LAM, n_epochs=epochs_linear(LIN_LAM), batch_size=BATCH, random_state=SEED, **kw).fit(Ztr, ytier_tr)
        sched[name] = (m.history_, m.objective_, ev.precision_recall_f1(ytier_va, m.predict(Zva))[2])
    ep = {}
    for e in [1, 5, 30]:
        m = PegasosSVM(lambda_=LIN_LAM, n_epochs=e, batch_size=BATCH, random_state=SEED).fit(Ztr, ytier_tr)
        ep[e] = (m.objective_, ev.precision_recall_f1(ytier_va, m.predict(Zva))[2])
    R["svm_schedules"] = {k: {"final_objective": v[1], "val_f1": v[2]} for k, v in sched.items()}
    R["svm_epochs"] = {str(k): {"final_objective": v[0], "val_f1": v[1]} for k, v in ep.items()}

    gammas = [0.005, 0.02, 0.05, 0.1] if not QUICK else [0.02, 0.1]
    lam_rff = [1e-5, 1e-4, 1e-3] if not QUICK else [1e-4]
    rows = []
    for g in gammas:
        for lam in lam_rff:
            m = RFFPegasosSVM(gamma=g, n_components=500, lambda_=lam, n_epochs=epochs_rff(lam), batch_size=BATCH,
                              random_state=SEED).fit(Ztr, ytier_tr)
            pr = m.predict(Zva)
            rows.append(dict(gamma=g, lambda_=lam, epochs=epochs_rff(lam), margin_width=m.margin_width_, train_objective=m.objective_,
                             val_f1=ev.precision_recall_f1(ytier_va, pr)[2], val_auc=ev.roc_auc(ytier_va, m.decision_function(Zva))))
    T_rff = pd.DataFrame(rows); T_rff.to_csv(f"{RES_DIR}/svm_rff_sweep.csv", index=False)
    print(T_rff.to_string(index=False))
    br = T_rff.sort_values("val_f1", ascending=False).iloc[0]
    RFF_G, RFF_LAM = float(br["gamma"]), float(br["lambda_"])
    EP_LIN, EP_RFF = epochs_linear(LIN_LAM), epochs_rff(RFF_LAM)
    print(f"   FINAL linear SVM lambda={LIN_LAM:g} ({EP_LIN} epochs);  FINAL RBF-SVM gamma={RFF_G}, lambda={RFF_LAM:g} ({EP_RFF} epochs)")

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    for lam, h in hist.items():
        st, ob, _ = zip(*h); ax[0].plot(st, ob, label=f"$\\lambda$={lam:g}")
    ax[0].set(title="Training objective J (log-log)", xlabel="Pegasos step", ylabel="J(w,b)", xscale="log", yscale="log"); ax[0].legend(fontsize=5)
    a2 = ax[1]; a2.semilogx(T_lam.lambda_, T_lam.margin_width, "-o", ms=3, color="#C44E52"); a2.set_yscale("log")
    a2.set(title="Margin vs validation score", xlabel="$\\lambda$", ylabel="margin width 2/||w||")
    a3 = a2.twinx(); a3.grid(False); a3.semilogx(T_lam.lambda_, T_lam.val_f1, "-s", ms=3, color="#4C72B0", label="val F1")
    a3.semilogx(T_lam.lambda_, T_lam.val_auc, "-^", ms=3, color="#55A868", label="val AUC"); a3.set_ylabel("validation score"); a3.legend(fontsize=6, loc="lower right")
    for name, (h, _, _) in sched.items():
        st, ob, _ = zip(*h); ax[2].plot(st, ob, label=name)
    ax[2].set(title="Learning-rate schedules", xlabel="Pegasos step", ylabel="J(w,b)", xscale="log", yscale="log"); ax[2].legend(fontsize=5)
    fig.tight_layout(); savefig(fig, "fig_svm_study")

    # ---------------------------------------------- final models, fit on TRAIN
    banner("4. FINAL MODELS (train) + sklearn baselines, evaluated on TEST")
    FIT, PRED = {}, {}
    mk_tc = lambda: DecisionTree("classification", **CLS_CFG)
    mk_tr = lambda: DecisionTree("regression", **REG_CFG)
    t_cls, t_reg = {}, {}
    ours_tc, t_cls["fit"] = timed(lambda: mk_tc().fit(Xtr, ytier_tr))
    ours_tr, t_reg["fit"] = timed(lambda: mk_tr().fit(Xtr, ylog_tr))
    sk_tc, sk_tc_fit = timed(lambda: DecisionTreeClassifier(criterion=CLS_CFG["criterion"], max_depth=CLS_CFG["max_depth"],
                                                           min_samples_leaf=CLS_CFG["min_samples_leaf"], random_state=SEED).fit(Xtr, ytier_tr))
    sk_tr, sk_tr_fit = timed(lambda: DecisionTreeRegressor(max_depth=REG_CFG["max_depth"], min_samples_leaf=REG_CFG["min_samples_leaf"],
                                                          random_state=SEED).fit(Xtr, ylog_tr))
    lin, lin_fit = timed(lambda: PegasosSVM(lambda_=LIN_LAM, n_epochs=EP_LIN, batch_size=BATCH, random_state=SEED).fit(Ztr, ytier_tr))
    rbf, rbf_fit = timed(lambda: RFFPegasosSVM(gamma=RFF_G, n_components=500, lambda_=RFF_LAM, n_epochs=EP_RFF,
                                              batch_size=BATCH, random_state=SEED).fit(Ztr, ytier_tr))
    sgd = Pipeline([("scale", StandardScaler()), ("clf", SGDClassifier(loss="hinge", penalty="l2", alpha=LIN_LAM, max_iter=EP_LIN,
                                                                     tol=None, random_state=SEED, average=True))])   # ASGD matches our suffix averaging
    sgd_plain = Pipeline([("scale", StandardScaler()), ("clf", SGDClassifier(loss="hinge", penalty="l2", alpha=LIN_LAM, max_iter=EP_LIN,
                                                                           tol=None, random_state=SEED))])
    Ltr, Lte, Lva = (dp.log_transform(a, names) for a in (Xtr, Xte, Xva))          # stateless log; scaling is INSIDE the pipeline
    _, sgd_fit = timed(lambda: sgd.fit(Ltr, ytier_tr))
    _, sgd_plain_fit = timed(lambda: sgd_plain.fit(Ltr, ytier_tr))
    n_sub = 15000 if not QUICK else 3000
    sub = np.random.default_rng(SEED).choice(len(tr), n_sub, replace=False)
    C_match = 1.0 / (RFF_LAM * n_sub)
    svc = Pipeline([("scale", StandardScaler()), ("clf", SVC(kernel="rbf", gamma=RFF_G, C=C_match))])
    _, svc_fit = timed(lambda: svc.fit(Ltr[sub], ytier_tr[sub]))
    rbf_sub, rbf_sub_fit = timed(lambda: RFFPegasosSVM(gamma=RFF_G, n_components=500, lambda_=RFF_LAM, n_epochs=EP_RFF,
                                                      batch_size=BATCH, random_state=SEED).fit(Ztr[sub], ytier_tr[sub]))

    # --- predictions on TEST (+ timing)
    def pred_t(fn, X_):
        return timed(fn, X_)
    P = {}
    P["tree_cls"] = dict(pred=ours_tc.predict(Xte), score=ours_tc.predict_proba(Xte)[:, 1], fit=t_cls["fit"], pred_s=timed(ours_tc.predict, Xte)[1])
    P["sk_tree_cls"] = dict(pred=sk_tc.predict(Xte.astype(np.float32)), score=sk_tc.predict_proba(Xte)[:, 1], fit=sk_tc_fit, pred_s=timed(sk_tc.predict, Xte)[1])
    P["svm_lin"] = dict(pred=lin.predict(Zte), score=lin.decision_function(Zte), fit=lin_fit, pred_s=timed(lin.predict, Zte)[1])
    P["sk_sgd"] = dict(pred=sgd.predict(Lte), score=sgd.decision_function(Lte), fit=sgd_fit, pred_s=timed(sgd.predict, Lte)[1])
    P["sk_sgd_plain"] = dict(pred=sgd_plain.predict(Lte), score=sgd_plain.decision_function(Lte), fit=sgd_plain_fit, pred_s=timed(sgd_plain.predict, Lte)[1])
    P["svm_rbf"] = dict(pred=rbf.predict(Zte), score=rbf.decision_function(Zte), fit=rbf_fit, pred_s=timed(rbf.predict, Zte)[1])
    P["svm_rbf_sub"] = dict(pred=rbf_sub.predict(Zte), score=rbf_sub.decision_function(Zte), fit=rbf_sub_fit, pred_s=timed(rbf_sub.predict, Zte)[1])
    P["sk_svc_sub"] = dict(pred=svc.predict(Lte), score=svc.decision_function(Lte), fit=svc_fit, pred_s=timed(svc.predict, Lte)[1])
    NAMES_B = {"tree_cls": "Ours: decision tree", "sk_tree_cls": "sklearn DecisionTreeClassifier",
               "svm_lin": "Ours: Pegasos linear SVM", "sk_sgd": "sklearn SGDClassifier (hinge, averaged)", "sk_sgd_plain": "sklearn SGDClassifier (hinge, no averaging)",
               "svm_rbf": "Ours: Pegasos + RFF (RBF) SVM", "svm_rbf_sub": f"Ours: RFF SVM (n={n_sub} subsample)",
               "sk_svc_sub": f"sklearn SVC-RBF (n={n_sub} subsample)"}
    tabB = {}
    for k, v in P.items():
        met = cls_metrics(ytier_te, v["pred"], v["score"])
        lo, hi = ev.bootstrap_ci(lambda yt, yp: ev.precision_recall_f1(yt, yp)[2], ytier_te, v["pred"], n_boot=200 if not QUICK else 20, seed=SEED)
        met.update(f1_ci_low=lo, f1_ci_high=hi, fit_s=v["fit"], predict_s=v["pred_s"])
        tabB[NAMES_B[k]] = met
    majority = np.zeros_like(ytier_te)
    tabB["Baseline: always 'standard'"] = {**cls_metrics(ytier_te, majority, np.zeros(len(majority))), "fit_s": 0.0, "predict_s": 0.0}

    # --- Task A on TEST
    pa = dict(tree=ours_tr.predict(Xte), sk=sk_tr.predict(Xte))
    tabA = {}
    for name, p, fit_s in [("Ours: decision tree", pa["tree"], t_reg["fit"]), ("sklearn DecisionTreeRegressor", pa["sk"], sk_tr_fit),
                           ("Baseline: train-mean log price", np.full(len(te), ylog_tr.mean()), 0.0)]:
        met = ev.regression_report(ylog_te, p)
        lo, hi = ev.bootstrap_ci(lambda a, b: ev.rmse(np.exp(a), np.exp(b)), ylog_te, p, n_boot=200 if not QUICK else 20, seed=SEED)
        met.update(rmse_azn_ci_low=lo, rmse_azn_ci_high=hi, fit_s=fit_s)
        tabA[name] = met
    print("\n--- TASK A (regression) on TEST ---"); ev.compare(tabA)
    print("\n--- TASK B (price tier) on TEST ---"); ev.compare({k: {m: v[m] for m in ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "fit_s", "predict_s"]} for k, v in tabB.items()})
    pd.DataFrame(tabA).T.to_csv(f"{RES_DIR}/test_results_taskA.csv")
    pd.DataFrame(tabB).T.to_csv(f"{RES_DIR}/test_results_taskB.csv")
    R["test_taskA"], R["test_taskB"] = tabA, tabB
    R["configs"] = {"tree_cls": CLS_CFG, "tree_reg": REG_CFG, "svm_linear_lambda": LIN_LAM,
                    "svm_rbf": {"gamma": RFF_G, "lambda": RFF_LAM, "n_components": 500}, "epochs_linear": EP_LIN, "epochs_rff": EP_RFF, "batch": BATCH,
                    "svc_C_matched": C_match}

    # --- how similar are our trees and sklearn's? (Problem 5a)
    agree = {}
    agree["cls_pred_agreement"] = float(np.mean(P["tree_cls"]["pred"] == P["sk_tree_cls"]["pred"]))
    agree["cls_depth"] = (ours_tc.get_depth(), int(sk_tc.get_depth())); agree["cls_leaves"] = (ours_tc.get_n_leaves(), int(sk_tc.get_n_leaves()))
    agree["cls_top3_nodes_matched"] = shared_top_structure(ours_tc, sk_tc, 3)
    agree["reg_pred_max_abs_diff_log"] = float(np.abs(pa["tree"] - pa["sk"]).max())
    agree["reg_pred_agreement_1e-6"] = float(np.mean(np.abs(pa["tree"] - pa["sk"]) < 1e-6))
    agree["reg_depth"] = (ours_tr.get_depth(), int(sk_tr.get_depth())); agree["reg_leaves"] = (ours_tr.get_n_leaves(), int(sk_tr.get_n_leaves()))
    agree["reg_top3_nodes_matched"] = shared_top_structure(ours_tr, sk_tr, 3)
    agree["svm_lin_vs_sgd_pred_agreement"] = float(np.mean(P["svm_lin"]["pred"] == P["sk_sgd"]["pred"]))
    agree["svm_rbf_sub_vs_svc_pred_agreement"] = float(np.mean(P["svm_rbf_sub"]["pred"] == P["sk_svc_sub"]["pred"]))
    rows = []
    for dep in [3, 6, 10, None]:
        a_ = DecisionTree("classification", CLS_CFG["criterion"], max_depth=dep).fit(Xtr, ytier_tr)
        b_ = DecisionTreeClassifier(criterion=CLS_CFG["criterion"], max_depth=dep, random_state=SEED).fit(Xtr, ytier_tr)
        c_ = DecisionTree("regression", "mse", max_depth=dep).fit(Xtr, ylog_tr)
        d_ = DecisionTreeRegressor(max_depth=dep, random_state=SEED).fit(Xtr, ylog_tr)
        rows.append(dict(max_depth=-1 if dep is None else dep,
                         cls_pred_agree=float(np.mean(a_.predict(Xte) == b_.predict(Xte))), cls_leaves_ours=a_.get_n_leaves(), cls_leaves_sk=int(b_.get_n_leaves()),
                         reg_pred_agree=float(np.mean(np.abs(c_.predict(Xte) - d_.predict(Xte)) < 1e-6)), reg_leaves_ours=c_.get_n_leaves(), reg_leaves_sk=int(d_.get_n_leaves())))
    T_agree = pd.DataFrame(rows); T_agree.to_csv(f"{RES_DIR}/agreement_by_depth.csv", index=False)
    print(T_agree.round(4).to_string(index=False))
    R["agreement_by_depth"] = T_agree.to_dict("records")
    R["agreement"] = agree
    print("\n--- agreement with scikit-learn ---"); [print(f"   {k}: {v}") for k, v in agree.items()]

    # =========================================================== 5c. CV mean +/- std
    banner("5. 5-FOLD CROSS-VALIDATION (train+val pool; preprocessing + tier threshold fitted inside each fold)")
    pool = np.concatenate([tr, va]); K = 5 if not QUICK else 3

    def prep(a, b):
        A, B = dp.impute(X[a], X[b]); return A, B
    def prep_s(a, b):
        A, B = dp.impute(X[a], X[b]); return dp.standardize(dp.log_transform(A, names), dp.log_transform(B, names))

    def cv_cls(kind):
        def f(i, j):
            a, b = pool[i], pool[j]; yt_a, thr = dp.make_tier_label(y[a]); yt_b, _ = dp.make_tier_label(y[b], thr)
            if kind == "tree":
                A, B = prep(a, b); m = DecisionTree("classification", **CLS_CFG).fit(A, yt_a); r = tree_cls_eval(m, B, yt_b)
            elif kind == "sk_tree":
                A, B = prep(a, b); m = DecisionTreeClassifier(criterion=CLS_CFG["criterion"], max_depth=CLS_CFG["max_depth"],
                                                             min_samples_leaf=CLS_CFG["min_samples_leaf"], random_state=SEED).fit(A, yt_a)
                r = cls_metrics(yt_b, m.predict(B), m.predict_proba(B)[:, 1])
            elif kind == "svm":
                A, B = prep_s(a, b); m = PegasosSVM(lambda_=LIN_LAM, n_epochs=EP_LIN, batch_size=BATCH, random_state=SEED).fit(A, yt_a)
                r = cls_metrics(yt_b, m.predict(B), m.decision_function(B))
            elif kind == "sgd":
                A, B = prep(a, b); A, B = dp.log_transform(A, names), dp.log_transform(B, names)
                m = Pipeline([("s", StandardScaler()), ("c", SGDClassifier(loss="hinge", alpha=LIN_LAM, max_iter=EP_LIN, tol=None, random_state=SEED, average=True))]).fit(A, yt_a)
                r = cls_metrics(yt_b, m.predict(B), m.decision_function(B))
            else:
                A, B = prep_s(a, b); m = RFFPegasosSVM(gamma=RFF_G, n_components=500, lambda_=RFF_LAM, n_epochs=EP_RFF, batch_size=BATCH, random_state=SEED).fit(A, yt_a)
                r = cls_metrics(yt_b, m.predict(B), m.decision_function(B))
            return {k: r[k] for k in ["accuracy", "f1", "roc_auc"]}
        return f

    def cv_reg(kind):
        def f(i, j):
            a, b = pool[i], pool[j]; A, B = prep(a, b)
            m = DecisionTree("regression", **REG_CFG).fit(A, ylog[a]) if kind == "tree" else \
                DecisionTreeRegressor(max_depth=REG_CFG["max_depth"], min_samples_leaf=REG_CFG["min_samples_leaf"], random_state=SEED).fit(A, ylog[a])
            r = ev.regression_report(ylog[b], m.predict(B)); return {k: r[k] for k in ["rmse_log", "r2_log", "rmse_azn", "mae_azn"]}
        return f
    st = np.digitize(ylog[pool], np.quantile(ylog[pool], np.linspace(0, 1, 6)[1:-1]))
    cvA = {"Ours: decision tree": ev.cross_validate(cv_reg("tree"), len(pool), K, SEED, st),
           "sklearn DecisionTreeRegressor": ev.cross_validate(cv_reg("sk"), len(pool), K, SEED, st)}
    cvB = {NAMES_B[k]: ev.cross_validate(cv_cls(c), len(pool), K, SEED, st) for k, c in
           [("tree_cls", "tree"), ("sk_tree_cls", "sk_tree"), ("svm_lin", "svm"), ("sk_sgd", "sgd"), ("svm_rbf", "rbf")]}
    print("\n--- CV Task A ---"); ev.compare(cvA); print("\n--- CV Task B ---"); ev.compare(cvB)
    R["cv_taskA"], R["cv_taskB"] = cvA, cvB
    pd.DataFrame({k: {m: f"{v[0]:.4g} +/- {v[1]:.2g}" for m, v in d_.items()} for k, d_ in cvA.items()}).T.to_csv(f"{RES_DIR}/cv_taskA.csv")
    pd.DataFrame({k: {m: f"{v[0]:.4g} +/- {v[1]:.2g}" for m, v in d_.items()} for k, d_ in cvB.items()}).T.to_csv(f"{RES_DIR}/cv_taskB.csv")

    # =========================================================== 5c. scaling
    banner("6. FIT-TIME SCALING")
    ns = [2000, 5000, 10000, 20000, len(tr)] if not QUICK else [2000, 5000]
    sc_rows = []
    for n_ in ns:
        i = perm[:n_]
        sc_rows.append(dict(n=n_,
            ours_tree=timed(lambda: DecisionTree("classification", **CLS_CFG).fit(Xtr[i], ytier_tr[i]))[1],
            sk_tree=timed(lambda: DecisionTreeClassifier(criterion=CLS_CFG["criterion"], max_depth=CLS_CFG["max_depth"], min_samples_leaf=CLS_CFG["min_samples_leaf"], random_state=SEED).fit(Xtr[i], ytier_tr[i]))[1],
            ours_pegasos=timed(lambda: PegasosSVM(lambda_=LIN_LAM, n_epochs=EP_LIN, batch_size=BATCH, random_state=SEED).fit(Ztr[i], ytier_tr[i]))[1],
            sk_sgd=timed(lambda: sgd.fit(Ltr[i], ytier_tr[i]))[1]))
    T_sc = pd.DataFrame(sc_rows); T_sc.to_csv(f"{RES_DIR}/fit_time_scaling.csv", index=False); print(T_sc.round(3).to_string(index=False))
    sgd.fit(Ltr, ytier_tr)
    fig, ax = plt.subplots(figsize=(3.4, 2.5))
    for c, lab, st_ in [("ours_tree", "ours: tree", "-o"), ("sk_tree", "sklearn: tree", "--o"), ("ours_pegasos", "ours: Pegasos", "-s"), ("sk_sgd", "sklearn: SGD", "--s")]:
        ax.loglog(T_sc.n, T_sc[c], st_, ms=3, label=lab)
    ax.set(title="Fit time vs training size", xlabel="training rows", ylabel="seconds"); ax.legend(fontsize=6)
    fig.tight_layout(); savefig(fig, "fig_fit_time")

    # =========================================================== 6. evaluation figures + error analysis
    banner("7. EVALUATION FIGURES AND ERROR ANALYSIS")
    # ROC + confusion matrices
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    ROC_LAB = {"tree_cls": "ours: tree", "svm_lin": "ours: linear SVM", "svm_rbf": "ours: RBF SVM", "sk_svc_sub": "sklearn SVC-RBF"}
    for k, col in [("tree_cls", "#4C72B0"), ("svm_lin", "#DD8452"), ("svm_rbf", "#55A868"), ("sk_svc_sub", "#8172B2")]:
        fpr, tpr = ev.roc_curve(ytier_te, P[k]["score"]); ax[0].plot(fpr, tpr, color=col, label=f"{ROC_LAB[k]} ({tabB[NAMES_B[k]]['roc_auc']:.3f})")
    ax[0].plot([0, 1], [0, 1], "k:", lw=1); ax[0].set(title="ROC (test)", xlabel="FPR", ylabel="TPR"); ax[0].legend(fontsize=5, loc="lower right")
    for a_, k in zip(ax[1:], ["tree_cls", "svm_lin"]):
        cm = ev.confusion_matrix(ytier_te, P[k]["pred"]); a_.imshow(cm, cmap="Blues"); a_.grid(False)
        for (r_, c_), v in np.ndenumerate(cm): a_.text(c_, r_, str(v), ha="center", va="center", color="w" if v > cm.max() / 2 else "k")
        a_.set(title=f"Confusion: {NAMES_B[k].split(':')[-1].strip()[:20]}", xticks=[0, 1], yticks=[0, 1], xticklabels=["standard", "premium"], yticklabels=["standard", "premium"], xlabel="predicted", ylabel="true")
    fig.tight_layout(); savefig(fig, "fig_roc_confusion")
    R["confusion"] = {k: ev.confusion_matrix(ytier_te, P[k]["pred"]).tolist() for k in P}

    # feature importance / weights
    imp_c, imp_r = ours_tc.feature_importances_, ours_tr.feature_importances_
    wl = lin.w
    fig, ax = plt.subplots(1, 3, figsize=(7.4, 3.0))
    for a_, vals, ttl, col in [(ax[0], imp_c, "Tree importance, Task B", "#4C72B0"), (ax[1], imp_r, "Tree importance, Task A", "#55A868")]:
        o = np.argsort(vals)[-10:]; a_.barh([names[i] for i in o], vals[o], color=col); a_.set(title=ttl, xlabel="impurity decrease share")
    o = np.argsort(np.abs(wl))[-10:]; ax[2].barh([names[i] for i in o], wl[o], color=["#C44E52" if v < 0 else "#4C72B0" for v in wl[o]])
    ax[2].set(title="Linear SVM weights (std. features)", xlabel="w")
    for a_ in ax: a_.tick_params(axis="y", labelsize=6)
    fig.tight_layout(); savefig(fig, "fig_importance")
    R["importance"] = {"tree_cls": dict(zip(names, imp_c.round(4).tolist())), "tree_reg": dict(zip(names, imp_r.round(4).tolist())),
                       "svm_weights": dict(zip(names, wl.round(4).tolist())), "svm_bias": lin.b,
                       "svm_linear_margin": lin.margin_width_, "svm_linear_w_norm": lin.w_norm_, "svm_linear_n_support": int(len(lin.support_)),
                       "svm_rbf_margin": rbf.margin_width_, "svm_rbf_n_support": int(len(rbf.support_))}
    print("   top tree-B features:", [names[i] for i in np.argsort(imp_c)[::-1][:5]])
    print("   top tree-A features:", [names[i] for i in np.argsort(imp_r)[::-1][:5]])
    print("   top |w| SVM features:", [names[i] for i in np.argsort(np.abs(wl))[::-1][:5]])
    print(f"   linear SVM: ||w||={lin.w_norm_:.3f} margin 2/||w||={lin.margin_width_:.3f} support vectors={len(lin.support_)}/{len(tr)}")

    # ---- Task A residual analysis
    dte = df.iloc[te].reset_index(drop=True).assign(y=y[te], pred=np.exp(pa["tree"]), ylog=ylog_te, plog=pa["tree"])
    dte["ape"] = np.abs(dte.pred - dte.y) / dte.y
    dte["dec"] = pd.qcut(dte.y, 10, labels=False, duplicates="drop")
    by_dec = dte.groupby("dec").agg(n=("y", "size"), price_med=("y", "median"), mdape=("ape", "median"),
                                    bias_pct=("pred", lambda s: 0.0)).drop(columns="bias_pct")
    by_dec["mean_signed_pct_err"] = dte.groupby("dec").apply(lambda g: float(((g.pred - g.y) / g.y).mean()), include_groups=False)
    by_cat = dte.groupby("category").agg(n=("y", "size"), mdape=("ape", "median"), rmse_log=("ylog", lambda s: 0.0)).drop(columns="rmse_log")
    by_cat["rmse_log"] = dte.groupby("category").apply(lambda g: ev.rmse(g.ylog, g.plog), include_groups=False)
    top1 = dte[dte.y >= dte.y.quantile(0.99)]
    lux = {"n": int(len(top1)), "mean_signed_pct_err": float(((top1.pred - top1.y) / top1.y).mean()), "mdape": float(top1.ape.median())}
    train_cnt = df.iloc[tr]["location"].value_counts()
    dte["loc_train_n"] = dte["location"].map(train_cnt).fillna(0).astype(int)
    sparse = dte[dte.loc_train_n < 30]; dense = dte[dte.loc_train_n >= 300]
    sp = {"sparse_lt30_n": int(len(sparse)), "sparse_mdape": float(sparse.ape.median()) if len(sparse) else None,
          "dense_ge300_n": int(len(dense)), "dense_mdape": float(dense.ape.median())}
    worst_loc = dte.groupby("location").agg(n=("y", "size"), mdape=("ape", "median"), train_n=("loc_train_n", "first")).query("n >= 15").sort_values("mdape", ascending=False).head(8)
    by_dec.to_csv(f"{RES_DIR}/errors_taskA_by_price_decile.csv"); by_cat.to_csv(f"{RES_DIR}/errors_taskA_by_category.csv"); worst_loc.to_csv(f"{RES_DIR}/errors_taskA_worst_locations.csv")
    R["errors_A"] = {"luxury_top1pct": lux, "sparse_vs_dense_locations": sp, "median_ape_overall": float(dte.ape.median())}
    print("   Task A median APE overall:", round(float(dte.ape.median()), 3)); print(by_dec.round(3).to_string()); print(by_cat.round(3).to_string())
    print("   luxury top-1%:", lux, "\n   sparse vs dense districts:", sp); print(worst_loc.round(3).to_string())

    fig, ax = plt.subplots(1, 3, figsize=(7.4, 2.5))
    ax[0].scatter(dte.ylog, dte.plog, s=2, alpha=0.3, color="#4C72B0"); lo_, hi_ = dte.ylog.min(), dte.ylog.max(); ax[0].plot([lo_, hi_], [lo_, hi_], "k--", lw=1)
    ax[0].set(title="Task A: predicted vs actual", xlabel="log price (actual)", ylabel="log price (tree)")
    ax[1].bar(by_dec.index + 1, by_dec.mean_signed_pct_err * 100, color=["#C44E52" if v < 0 else "#4C72B0" for v in by_dec.mean_signed_pct_err])
    ax[1].set(title="Bias by price decile", xlabel="price decile (10 = most expensive)", ylabel="mean signed error, %"); ax[1].axhline(0, color="k", lw=0.8)
    g = dte.groupby("location").agg(n=("loc_train_n", "first"), mdape=("ape", "median"), m=("y", "size")).query("m >= 5")
    ax[2].scatter(g.n.clip(lower=1), g.mdape * 100, s=8, color="#55A868"); ax[2].set(xscale="log", title="Error vs district size", xlabel="train listings in district", ylabel="median abs % error")
    fig.tight_layout(); savefig(fig, "fig_error_task_a")

    # ---- Task B error analysis: where do errors concentrate?
    dte["tier"] = ytier_te; dte["rel"] = np.log(dte.y / THR)
    bins_rel = pd.cut(dte.rel, [-5, -0.5, -0.25, -0.1, 0, 0.1, 0.25, 0.5, 5])
    err = {}
    for k in ["tree_cls", "svm_lin", "svm_rbf"]:
        dte[f"wrong_{k}"] = (P[k]["pred"] != ytier_te).astype(int)
        err[k] = dte.groupby(bins_rel, observed=True)[f"wrong_{k}"].mean()
    near = (dte.rel.abs() < 0.1)
    R["errors_B"] = {k: {"error_rate": float(dte[f"wrong_{k}"].mean()), "error_rate_within_10pct_of_threshold": float(dte.loc[near, f"wrong_{k}"].mean()),
                         "share_of_errors_within_10pct_of_threshold": float(dte.loc[near, f"wrong_{k}"].sum() / max(dte[f"wrong_{k}"].sum(), 1)),
                         "error_by_category": dte.groupby("category")[f"wrong_{k}"].mean().round(4).to_dict()} for k in err}
    R["errors_B"]["near_threshold_share_of_test"] = float(near.mean())
    print({k: (round(v["error_rate"], 4), round(v["error_rate_within_10pct_of_threshold"], 3), round(v["share_of_errors_within_10pct_of_threshold"], 3)) for k, v in R["errors_B"].items() if k != "near_threshold_share_of_test"})
    fig, ax = plt.subplots(1, 2, figsize=(5.4, 2.4))
    xs = np.arange(len(bins_rel.cat.categories)); w_ = 0.27
    for i, (k, col) in enumerate([("tree_cls", "#4C72B0"), ("svm_lin", "#DD8452"), ("svm_rbf", "#55A868")]):
        ax[0].bar(xs + (i - 1) * w_, err[k].values * 100, w_, color=col, label=NAMES_B[k].split(":")[-1].strip()[:16])
    ax[0].set_xticks(xs); ax[0].set_xticklabels([f"{c.left:g}..{c.right:g}" for c in bins_rel.cat.categories], rotation=45, fontsize=5)
    ax[0].set(title="Error rate vs distance to threshold", xlabel="log(price / threshold)", ylabel="% misclassified"); ax[0].legend(fontsize=5)
    cat_err = pd.DataFrame({k: dte.groupby("category")[f"wrong_{k}"].mean() * 100 for k in ["tree_cls", "svm_lin", "svm_rbf"]})
    cat_err.plot.bar(ax=ax[1], color=["#4C72B0", "#DD8452", "#55A868"], legend=False, rot=20); ax[1].set(title="Error rate by category", ylabel="% misclassified", xlabel="")
    fig.tight_layout(); savefig(fig, "fig_error_task_b")

    # =========================================================== 8. BONUS PROBLEMS (8, 9, 10)
    banner("8. BONUS PROBLEMS: Ridge Baseline, Random Forest Ensemble, Geospatial PCA & K-Means")
    # ---- Problem 8: From-Scratch Ridge Regression (Normal Equations)
    print("\n--- Problem 8 (Bonus): From-scratch Ridge Regression Baseline ---")
    ridge_lams = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_lam, best_val_rmse = 1.0, float("inf")
    for lam in ridge_lams:
        rr = RidgeRegression(lambda_=lam).fit(Ztr, ylog_tr)
        val_rmse = ev.rmse(ylog_va, rr.predict(Zva))
        if val_rmse < best_val_rmse:
            best_lam, best_val_rmse = lam, val_rmse
    print(f"   best Ridge lambda: {best_lam} (val RMSE={best_val_rmse:.4f})")
    rr_final = RidgeRegression(lambda_=best_lam).fit(Ztr, ylog_tr)
    rr_pred_te = rr_final.predict(Zte)
    rr_metrics = ev.regression_report(ylog_te, rr_pred_te)
    print(f"   Ridge Test RMSE_log: {rr_metrics['rmse_log']:.4f}, R2_log: {rr_metrics['r2_log']:.4f}, MAE (AZN): {rr_metrics['mae_azn']:,.0f}")
    R["bonus_problem_8_ridge"] = {"best_lambda": best_lam, "test_metrics": rr_metrics}

    # ---- Problem 9: From-Scratch Random Forest Ensemble from Own Trees
    print("\n--- Problem 9 (Bonus): From-scratch Random Forest Ensemble ---")
    n_trees_list = [1, 3, 5, 10, 15] if QUICK else [1, 5, 10, 20]
    rf_reg_rmse, rf_cls_f1 = [], []
    for B in n_trees_list:
        rf_r = RandomForest("regression", n_estimators=B, max_depth=10, min_samples_leaf=5, random_state=SEED).fit(Xtr, ylog_tr)
        rf_c = RandomForest("classification", n_estimators=B, max_depth=14, min_samples_leaf=1, random_state=SEED).fit(Xtr, ytier_tr)
        r_te = ev.rmse(ylog_te, rf_r.predict(Xte))
        c_te = ev.precision_recall_f1(ytier_te, rf_c.predict(Xte))[2]
        rf_reg_rmse.append(r_te)
        rf_cls_f1.append(c_te)
        print(f"   B={B:02d} trees -> Task A Test RMSE_log: {r_te:.4f} | Task B Test F1: {c_te:.4f}")

    fig_rf, ax_rf = plt.subplots(1, 2, figsize=(6.2, 2.4))
    ax_rf[0].plot(n_trees_list, rf_reg_rmse, "o-", color="#4C72B0", lw=1.5, ms=4)
    ax_rf[0].axhline(tabA["Ours: decision tree"]["rmse_log"], color="k", ls="--", lw=0.9, label="Single Tree")
    ax_rf[0].set(title="RF Task A: RMSE vs # Trees", xlabel="Number of Trees (B)", ylabel="Test RMSE_log")
    ax_rf[0].legend(fontsize=7)

    ax_rf[1].plot(n_trees_list, rf_cls_f1, "s-", color="#55A868", lw=1.5, ms=4)
    ax_rf[1].axhline(tabB["Ours: decision tree"]["f1"], color="k", ls="--", lw=0.9, label="Single Tree")
    ax_rf[1].set(title="RF Task B: F1 vs # Trees", xlabel="Number of Trees (B)", ylabel="Test F1")
    ax_rf[1].legend(fontsize=7)
    fig_rf.tight_layout()
    savefig(fig_rf, "fig_random_forest")
    R["bonus_problem_9_rf"] = {"n_estimators": n_trees_list, "test_rmse_log": rf_reg_rmse, "test_f1": rf_cls_f1}

    # ---- Problem 10: Unsupervised Geospatial K-Means Clustering & PCA
    print("\n--- Problem 10 (Bonus): Geospatial K-Means & PCA Market Segments ---")
    coords_tr = np.column_stack([df.iloc[tr]["lat"].values, df.iloc[tr]["lng"].values])
    km = KMeans(n_clusters=5, random_state=SEED).fit(coords_tr)
    coords_te = np.column_stack([dte["lat"].values, dte["lng"].values])
    dte["geo_cluster"] = km.predict(coords_te)

    # Compute cluster statistics
    cluster_stats = dte.groupby("geo_cluster").agg(
        listings=("y", "size"),
        median_price=("y", "median"),
        mdape=("ape", "median"),
        misclassified_pct=("wrong_tree_cls", lambda s: float(s.mean() * 100))
    )
    print("   Geospatial Cluster Summary (Test Set):")
    print(cluster_stats.round(2).to_string())

    # PCA on standardized features
    pca = PCA(n_components=2).fit(Ztr)
    pca_te = pca.transform(Zte)
    dte["pca_1"], dte["pca_2"] = pca_te[:, 0], pca_te[:, 1]
    print(f"   PCA Explained Variance (top 2): {pca.explained_variance_ratio_}")

    fig_geo, ax_geo = plt.subplots(1, 2, figsize=(7.2, 2.7))
    sc = ax_geo[0].scatter(dte["lng"], dte["lat"], c=dte["geo_cluster"], cmap="tab10", s=6, alpha=0.6)
    ax_geo[0].scatter(km.cluster_centers_[:, 1], km.cluster_centers_[:, 0], c="black", marker="X", s=50, label="Centroids")
    ax_geo[0].set(title="Geospatial K-Means Submarkets (k=5)", xlabel="Longitude", ylabel="Latitude")
    ax_geo[0].legend(fontsize=7)

    cluster_stats["mdape_pct"] = cluster_stats["mdape"] * 100
    cluster_stats[["mdape_pct", "misclassified_pct"]].plot.bar(ax=ax_geo[1], color=["#4C72B0", "#DD8452"], rot=0)
    ax_geo[1].set(title="Error Metrics Across Geospatial Clusters", xlabel="Cluster ID", ylabel="Error %")
    ax_geo[1].legend(["Task A MdAPE %", "Task B Misclass %"], fontsize=7)
    fig_geo.tight_layout()
    savefig(fig_geo, "fig_geospatial_clusters")
    R["bonus_problem_10_geo"] = cluster_stats.to_dict("index")

    # =========================================================== write summaries
    R["runtime_minutes"] = round((time.time() - t_start) / 60, 1)
    R["tree_stopping_rules"] = T_eff.to_dict("records")
    def _clean(o):
        if isinstance(o, dict): return {str(k): _clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, (np.integer,)): return int(o)
        return o
    with open(f"{RES_DIR}/results.json", "w") as fh:
        json.dump(_clean(R), fh, indent=2, ensure_ascii=False)
    make_report_tables.main()
    banner(f"DONE in {R['runtime_minutes']} min. Numbers: {RES_DIR}/  Figures: {FIG_DIR}/  Tables: report/tables/")


if __name__ == "__main__":
    main()
