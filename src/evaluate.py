"""
Evaluation and Walk-Forward Backtesting module for SirenCast UA.
Executes temporal validation across 2026 folds, comparing:
1. Heuristic Baseline
2. Logistic Regression
3. LightGBM V1 (Time Series Only)
4. LightGBM V2 (Multimodal: TS + Telegram NLP Signals)
Computes ROC-AUC, PR-AUC, Brier score, and exports diagnostic plots to reports/figures/.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import BASE_DIR
from src.features_text import get_multimodal_dataset
from src.models import (
    HeuristicCivilDefenseModel,
    create_lightgbm_model,
    create_logistic_baseline,
    compute_time_decay_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

FIGURES_DIR = BASE_DIR / "reports" / "figures"


def compute_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.35,
) -> Dict[str, float]:
    """
    Computes key performance metrics for probabilistic forecasts.
    """
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def run_walk_forward_evaluation(
    target_col: str = "target_kyiv_active_30m",
    save_plots: bool = True,
    figures_dir: Path = FIGURES_DIR,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Runs Walk-Forward expanding-window cross validation across 2026 folds
    with exponential time-decay sample weighting (T_half = 270 days).
    Folds:
    - Fold 1: Train 2023..2025, Test 2026 Q1 (Jan - Mar 2026)
    - Fold 2: Train 2023..2026-03, Test 2026 Q2 (Apr - Jun 2026)
    - Fold 3: Train 2023..2026-06, Test 2026 Q3 (Jul - Sep 2026)
    """
    df = get_multimodal_dataset()
    ignore_cols = ["timestamp", "target_kyiv_start_30m", "target_kyiv_active_30m"]
    text_cols = [c for c in df.columns if c.startswith("tg_")]
    ts_cols = [c for c in df.columns if c not in ignore_cols and not c.startswith("tg_")]
    multimodal_cols = ts_cols + text_cols

    splits = [
        ("Fold 1 (2026 Q1: Jan-Mar)", "2026-01-01", "2026-04-01"),
        ("Fold 2 (2026 Q2: Apr-Jun)", "2026-04-01", "2026-07-01"),
        ("Fold 3 (2026 Q3: Jul-Sep)", "2026-07-01", "2026-09-08"),
    ]

    records = []
    models_config = {
        "Heuristic Baseline": (HeuristicCivilDefenseModel(), ts_cols),
        "Logistic Regression": (create_logistic_baseline(), ts_cols),
        "LightGBM V1 (TS-only)": (create_lightgbm_model(), ts_cols),
        "LightGBM V2 (Multimodal)": (create_lightgbm_model(), multimodal_cols),
    }

    cached_predictions = {m: [] for m in models_config}
    cached_y_true = []

    for fold_name, t_start, t_end in splits:
        logger.info(f"\n=================== Evaluating {fold_name} ===================")
        train_mask = df["timestamp"] < t_start
        test_mask = (df["timestamp"] >= t_start) & (df["timestamp"] < t_end)

        y_train = df.loc[train_mask, target_col]
        y_test = df.loc[test_mask, target_col]
        cached_y_true.append(y_test.values)

        train_weights = compute_time_decay_weights(
            df.loc[train_mask, "timestamp"], half_life_days=270.0
        )

        logger.info(f"Train samples: {len(y_train):,} (positive: {y_train.sum():,})")
        logger.info(f"Test samples:  {len(y_test):,} (positive: {y_test.sum():,})")

        for m_name, (model, feat_cols) in models_config.items():
            X_train = df.loc[train_mask, feat_cols]
            X_test = df.loc[test_mask, feat_cols]

            if m_name == "Logistic Regression":
                model.fit(X_train, y_train, clf__sample_weight=train_weights)
            elif m_name.startswith("LightGBM"):
                model.fit(X_train, y_train, sample_weight=train_weights)
            else:
                model.fit(X_train, y_train)

            probs = model.predict_proba(X_test)[:, 1]
            cached_predictions[m_name].append(probs)

            metrics = compute_metrics(y_test.values, probs)
            records.append({
                "fold": fold_name,
                "model": m_name,
                **metrics,
            })
            logger.info(
                f"[{m_name}] ROC-AUC: {metrics['roc_auc']:.4f} | "
                f"PR-AUC: {metrics['pr_auc']:.4f} | "
                f"Brier: {metrics['brier_score']:.4f} | "
                f"F1: {metrics['f1']:.4f}"
            )

            # Quiet sky onset metrics (Early Warning evaluation: kyiv_is_active == 0)
            calm_test_mask = (df.loc[test_mask, "kyiv_is_active"] == 0).values
            if calm_test_mask.sum() > 0 and target_col == "target_kyiv_active_30m":
                y_onset_calm = df.loc[test_mask, "target_kyiv_start_30m"].values[calm_test_mask]
                probs_calm = probs[calm_test_mask]
                onset_metrics = compute_metrics(y_onset_calm, probs_calm, threshold=0.15)
                logger.info(
                    f"   ↳ [Quiet Sky Onset] ROC-AUC: {onset_metrics['roc_auc']:.4f} | "
                    f"PR-AUC: {onset_metrics['pr_auc']:.4f} (baseline: {y_onset_calm.mean():.3f}) | "
                    f"Recall@0.15: {onset_metrics['recall']:.2%}"
                )

    results_df = pd.DataFrame(records)

    # -------------------------------------------------------------
    # PLOTTING DIAGNOSTICS
    # -------------------------------------------------------------
    if save_plots:
        figures_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Generating diagnostic figures in {figures_dir}...")
        plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
        y_all_test = np.concatenate(cached_y_true)

        colors = {
            "Heuristic Baseline": "#888888",
            "Logistic Regression": "#2b5c8f",
            "LightGBM V1 (TS-only)": "#d95f02",
            "LightGBM V2 (Multimodal)": "#1b7837",
        }

        # Plot 1: ROC Curves
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        for m_name in models_config:
            all_probs = np.concatenate(cached_predictions[m_name])
            fpr, tpr, _ = roc_curve(y_all_test, all_probs)
            auc_val = roc_auc_score(y_all_test, all_probs)
            ax.plot(fpr, tpr, label=f"{m_name} (AUC = {auc_val:.3f})", color=colors.get(m_name), linewidth=2)

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance (0.50)")
        ax.set_title(f"ROC Curves - Walk-Forward Evaluation 2026 ({target_col})", fontsize=12, fontweight="bold")
        ax.set_xlabel("False Positive Rate", fontsize=10)
        ax.set_ylabel("True Positive Rate", fontsize=10)
        ax.legend(loc="lower right", frameon=True)
        fig.tight_layout()
        fig.savefig(figures_dir / "roc_curve_walkforward.png")
        plt.close(fig)

        # Plot 2: Precision-Recall Curves
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        baseline_pr = float(y_all_test.mean())
        for m_name in models_config:
            all_probs = np.concatenate(cached_predictions[m_name])
            prec, rec, _ = precision_recall_curve(y_all_test, all_probs)
            pr_val = average_precision_score(y_all_test, all_probs)
            ax.plot(rec, prec, label=f"{m_name} (PR-AUC = {pr_val:.3f})", color=colors.get(m_name), linewidth=2)

        ax.axhline(baseline_pr, color="black", linestyle="--", alpha=0.5, label=f"No Skill Baseline ({baseline_pr:.2%})")
        ax.set_title(f"Precision-Recall Curves - Walk-Forward 2026 ({target_col})", fontsize=12, fontweight="bold")
        ax.set_xlabel("Recall", fontsize=10)
        ax.set_ylabel("Precision", fontsize=10)
        ax.legend(loc="upper right", frameon=True)
        fig.tight_layout()
        fig.savefig(figures_dir / "pr_curve_walkforward.png")
        plt.close(fig)

        # Plot 3: Probability Calibration
        fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
        ax.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
        for m_name in ["Logistic Regression", "LightGBM V1 (TS-only)", "LightGBM V2 (Multimodal)"]:
            all_probs = np.concatenate(cached_predictions[m_name])
            prob_true, prob_pred = calibration_curve(y_all_test, all_probs, n_bins=10)
            ax.plot(prob_pred, prob_true, marker="o", label=f"{m_name}", color=colors.get(m_name), linewidth=2)

        ax.set_title("Probability Calibration (Reliability Curve)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Mean Predicted Probability", fontsize=10)
        ax.set_ylabel("Empirical Frequency", fontsize=10)
        ax.legend(loc="lower right", frameon=True)
        fig.tight_layout()
        fig.savefig(figures_dir / "calibration_reliability_curve.png")
        plt.close(fig)

        # Plot 4: Feature Importance for Multimodal LightGBM
        final_lgb_v2 = models_config["LightGBM V2 (Multimodal)"][0]
        imp_series = pd.Series(final_lgb_v2.feature_importances_, index=multimodal_cols).sort_values(ascending=True)
        top20 = imp_series.tail(20)

        fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
        bar_colors = ["#2b5c8f" if col.startswith("tg_") else "#1b7837" for col in top20.index]
        ax.barh(top20.index, top20.values, color=bar_colors, alpha=0.85)
        ax.set_title("Top 20 Features Importance - LightGBM V2 Multimodal (Blue = Telegram NLP)", fontsize=12, fontweight="bold")
        ax.set_xlabel("Split Gain Importance", fontsize=10)
        fig.tight_layout()
        fig.savefig(figures_dir / "feature_importance_ranking.png")
        plt.close(fig)

        # Plot 5: V1 vs V2 Comparison Bar Chart
        fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
        piv_auc = results_df[results_df["model"].isin(["LightGBM V1 (TS-only)", "LightGBM V2 (Multimodal)"])].pivot(
            index="fold", columns="model", values="roc_auc"
        )
        x = np.arange(len(piv_auc))
        width = 0.35
        ax.bar(x - width/2, piv_auc["LightGBM V1 (TS-only)"], width, label="V1 (TS-only)", color="#d95f02", alpha=0.9)
        ax.bar(x + width/2, piv_auc["LightGBM V2 (Multimodal)"], width, label="V2 (Multimodal: TS + Telegram)", color="#1b7837", alpha=0.9)
        ax.set_ylim(0.85, 0.95)
        ax.set_xticks(x)
        ax.set_xticklabels([f"Fold {i+1}" for i in range(len(piv_auc))], fontsize=11)
        ax.set_ylabel("ROC-AUC Score", fontsize=11)
        ax.set_title("Ablation Study: LightGBM V1 vs Multimodal V2 across 2026 Folds", fontsize=12, fontweight="bold")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(figures_dir / "v1_vs_v2_comparison.png")
        plt.close(fig)
        logger.info(f"All figures saved to {figures_dir}")

    return results_df, models_config


if __name__ == "__main__":
    results, _ = run_walk_forward_evaluation()
    print("\n\n=================== SUMMARY BENCHMARK TABLE ===================")
    pivot_auc = results.pivot(index="fold", columns="model", values="roc_auc")
    pivot_pr = results.pivot(index="fold", columns="model", values="pr_auc")
    print("\n--- ROC-AUC Comparison ---")
    print(pivot_auc.round(4))
    print("\n--- PR-AUC Comparison ---")
    print(pivot_pr.round(4))
