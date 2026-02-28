"""
Class 1: Technical confound audit.

Detects batch, donor, and assay-method leakage in GRN edge scores
by computing the Artifact Sensitivity Index (ASI) per edge and
training leakage classifiers on edge-derived features.
"""

import numpy as np
import pandas as pd
from typing import Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.preprocessing import LabelEncoder
import warnings


class TechnicalAudit:
    """Audit GRN edge scores for technical confounds (batch/donor/method).

    The audit works in two stages:
      1. Per-edge ASI: measures how much each edge score changes when
         a technical covariate is balanced out.
      2. Leakage classifiers: train classifiers to predict technical
         covariates from edge-product features, quantifying aggregate
         information leakage.

    Parameters
    ----------
    asi_threshold : float
        Edges with ASI above this value are flagged as blacklisted.
        Default 0.5, following the paper.
    n_top_features : int
        Number of top-variance edge features used for leakage classifiers.
    n_splits : int
        Number of CV folds for leakage classification.
    random_state : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        asi_threshold: float = 0.5,
        n_top_features: int = 200,
        n_splits: int = 5,
        random_state: int = 42,
    ):
        self.asi_threshold = asi_threshold
        self.n_top_features = n_top_features
        self.n_splits = n_splits
        self.random_state = random_state

    def compute_asi(
        self,
        scores_full: pd.Series,
        scores_balanced: pd.Series,
    ) -> pd.Series:
        """Compute Artifact Sensitivity Index for each edge.

        ASI = |r_full - r_balanced| / max(|r_full|, 0.01)

        Parameters
        ----------
        scores_full : pd.Series
            Edge scores from the full (unbalanced) dataset.
            Index should be edge identifiers (e.g., "TF->target").
        scores_balanced : pd.Series
            Edge scores from the covariate-balanced dataset.
            Must share the same index as scores_full.

        Returns
        -------
        pd.Series
            Per-edge ASI values, same index as input.
        """
        # Align indices
        common = scores_full.index.intersection(scores_balanced.index)
        full = scores_full.loc[common].astype(float)
        balanced = scores_balanced.loc[common].astype(float)

        denominator = np.maximum(np.abs(full), 0.01)
        asi = np.abs(full - balanced) / denominator
        return asi

    def flag_blacklist(self, asi: pd.Series) -> pd.Series:
        """Flag edges exceeding the ASI threshold.

        Parameters
        ----------
        asi : pd.Series
            Per-edge ASI values.

        Returns
        -------
        pd.Series
            Boolean series: True if edge is blacklisted.
        """
        return asi > self.asi_threshold

    def leakage_classification(
        self,
        edge_features: pd.DataFrame,
        covariate: pd.Series,
    ) -> dict:
        """Train leakage classifiers and return diagnostic metrics.

        Trains logistic regression and random forest classifiers to predict
        a technical covariate from cell-level edge-product features, using
        stratified k-fold cross-validation.

        Parameters
        ----------
        edge_features : pd.DataFrame
            Cell-by-edge feature matrix (rows = cells, columns = edges).
            Typically constructed as the outer product of gene expression
            for each TF-target pair.
        covariate : pd.Series
            Technical covariate labels for each cell (e.g., donor ID,
            batch ID, assay method). Must be aligned with edge_features rows.

        Returns
        -------
        dict
            Keys: 'auc_logreg', 'auc_rf', 'auc_best', 'balanced_acc_best',
                  'model_best', 'per_fold_auc'.
        """
        # Select top-variance features to keep computation tractable
        variances = edge_features.var(axis=0)
        top_cols = variances.nlargest(self.n_top_features).index
        X = edge_features[top_cols].values
        le = LabelEncoder()
        y = le.fit_transform(covariate.values)

        n_classes = len(np.unique(y))
        if n_classes < 2:
            return {
                "auc_logreg": np.nan,
                "auc_rf": np.nan,
                "auc_best": np.nan,
                "balanced_acc_best": np.nan,
                "model_best": "none",
                "per_fold_auc": [],
                "warning": "Fewer than 2 classes in covariate",
            }

        multi_class = "ovr" if n_classes > 2 else "raise"
        average = "macro" if n_classes > 2 else None

        skf = StratifiedKFold(
            n_splits=self.n_splits, shuffle=True,
            random_state=self.random_state,
        )

        results = {"logreg": [], "rf": []}

        for train_idx, test_idx in skf.split(X, y):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            # Logistic regression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                lr = LogisticRegression(
                    max_iter=1000, random_state=self.random_state,
                    solver="lbfgs", multi_class="multinomial" if n_classes > 2 else "auto",
                )
                lr.fit(X_train, y_train)
                if n_classes == 2:
                    lr_proba = lr.predict_proba(X_test)[:, 1]
                else:
                    lr_proba = lr.predict_proba(X_test)
                lr_auc = roc_auc_score(
                    y_test, lr_proba,
                    multi_class=multi_class, average=average,
                )
                results["logreg"].append(lr_auc)

            # Random forest
            rf = RandomForestClassifier(
                n_estimators=100, random_state=self.random_state,
                n_jobs=-1,
            )
            rf.fit(X_train, y_train)
            if n_classes == 2:
                rf_proba = rf.predict_proba(X_test)[:, 1]
            else:
                rf_proba = rf.predict_proba(X_test)
            rf_auc = roc_auc_score(
                y_test, rf_proba,
                multi_class=multi_class, average=average,
            )
            results["rf"].append(rf_auc)

        auc_lr = np.mean(results["logreg"])
        auc_rf = np.mean(results["rf"])
        best_model = "rf" if auc_rf >= auc_lr else "logreg"

        return {
            "auc_logreg": round(auc_lr, 4),
            "auc_rf": round(auc_rf, 4),
            "auc_best": round(max(auc_lr, auc_rf), 4),
            "balanced_acc_best": None,  # would require re-fitting
            "model_best": best_model,
            "per_fold_auc": {
                "logreg": [round(x, 4) for x in results["logreg"]],
                "rf": [round(x, 4) for x in results["rf"]],
            },
        }

    def run(
        self,
        scores_full: pd.Series,
        scores_balanced: Optional[pd.Series] = None,
        edge_features: Optional[pd.DataFrame] = None,
        covariates: Optional[dict] = None,
    ) -> dict:
        """Run the full Class 1 technical audit.

        Parameters
        ----------
        scores_full : pd.Series
            Edge scores from full dataset.
        scores_balanced : pd.Series, optional
            Edge scores from balanced dataset (for ASI computation).
        edge_features : pd.DataFrame, optional
            Cell-by-edge feature matrix (for leakage classification).
        covariates : dict, optional
            Mapping of covariate name to pd.Series of labels per cell.
            E.g., {"donor": donor_labels, "batch": batch_labels}.

        Returns
        -------
        dict
            Complete Class 1 audit results.
        """
        result = {"class": 1, "name": "Technical confound audit"}

        # ASI computation
        if scores_balanced is not None:
            asi = self.compute_asi(scores_full, scores_balanced)
            blacklist = self.flag_blacklist(asi)
            result["asi"] = {
                "values": asi,
                "mean": round(float(asi.mean()), 4),
                "median": round(float(asi.median()), 4),
                "blacklist_rate": round(float(blacklist.mean()), 4),
                "n_blacklisted": int(blacklist.sum()),
                "n_total": len(blacklist),
                "threshold": self.asi_threshold,
            }
            result["blacklist"] = blacklist
        else:
            result["asi"] = {"warning": "No balanced scores provided; ASI not computed."}

        # Leakage classification
        if edge_features is not None and covariates is not None:
            result["leakage"] = {}
            for cov_name, cov_labels in covariates.items():
                result["leakage"][cov_name] = self.leakage_classification(
                    edge_features, cov_labels,
                )
        else:
            result["leakage"] = {
                "warning": "No edge features or covariates provided; "
                           "leakage classification not run.",
            }

        return result
