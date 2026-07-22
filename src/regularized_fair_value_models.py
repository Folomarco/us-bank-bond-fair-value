from __future__ import annotations


import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from config_institutional import (
    REGRESSION_DIR,
    TABLES_DIR,
    FIGURES_DIR,
    ensure_directories,
)


GROUP_COL = "cusip_id"
DATE_COL = "date"
SPLIT_COL = "sample_split"
TARGET_COL = "final_vwap_return"

TRAIN_END_DATE = pd.Timestamp("2023-01-01")
VALIDATION_END_DATE = pd.Timestamp("2024-01-01")

PANEL_PATH = REGRESSION_DIR / "regression_panel_gap5_with_peer_factors.parquet"

OUTPUT_CANDIDATE_RESULTS = TABLES_DIR / "regularized_gap5_candidate_model_results.csv"
OUTPUT_SELECTED_RESULTS = TABLES_DIR / "regularized_gap5_selected_model_results.csv"
OUTPUT_SELECTED_SUMMARY = TABLES_DIR / "regularized_gap5_selected_model_summary.csv"
OUTPUT_COEFFICIENTS = TABLES_DIR / "regularized_gap5_model_coefficients.csv"
OUTPUT_RESIDUAL_DIAGNOSTICS = TABLES_DIR / "regularized_gap5_residual_diagnostics.csv"
OUTPUT_PCA_LOADINGS = TABLES_DIR / "regularized_gap5_pca_loadings.csv"
OUTPUT_PCA_EXPLAINED_VARIANCE = TABLES_DIR / "regularized_gap5_pca_explained_variance.csv"
OUTPUT_PREDICTIONS = REGRESSION_DIR / "regularized_gap5_model_predictions.parquet"
OUTPUT_MANIFEST = TABLES_DIR / "regularized_gap5_manifest.json"

FIG_TEST_OOS_R2 = FIGURES_DIR / "regularized_gap5_test_oos_r2.png"
FIG_VALIDATION_TEST_OOS_R2 = FIGURES_DIR / "regularized_gap5_validation_vs_test_oos_r2.png"

RATES_FEATURES = [
    "d_dgs2_interval",
    "d_dgs5_interval",
    "d_dgs10_interval",
    "d_dgs30_interval",
]

EQUITY_FEATURES = [
    "issuer_equity_log_return_interval",
]

VIX_FEATURES = [
    "d_vix_interval",
]

PEER_FEATURES_RAW = [
    "peer_raw_same_issuer_maturity",
    "peer_raw_other_bank_maturity",
    "peer_raw_bank_sector_maturity",
]

M4_FEATURES = RATES_FEATURES + EQUITY_FEATURES + VIX_FEATURES + PEER_FEATURES_RAW

RIDGE_ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
LASSO_ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
ELASTIC_NET_ALPHAS = [1e-6, 1e-5, 1e-4, 1e-3]
ELASTIC_NET_L1_RATIOS = [0.25, 0.50, 0.75]

PCR_RATES_COMPONENTS = [1, 2, 3]
PCR_PEER_COMPONENTS = [1, 2]
PCR_FULL_COMPONENTS = [3, 5, 7]

SAVE_PREDICTIONS_FOR_SPLITS = ["validation", "test"]

RANDOM_SEED = 42


def json_safe(x: Any) -> Any:
    if isinstance(x, Path):
        return str(x)
    if isinstance(x, pd.Timestamp):
        return x.isoformat()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [json_safe(v) for v in x]
    return x


def assign_sample_split(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")

    out[SPLIT_COL] = np.select(
        [
            out[DATE_COL] < TRAIN_END_DATE,
            (out[DATE_COL] >= TRAIN_END_DATE) & (out[DATE_COL] < VALIDATION_END_DATE),
            out[DATE_COL] >= VALIDATION_END_DATE,
        ],
        ["train", "validation", "test"],
        default="excluded",
    )

    return out


def prepare_panel(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out[DATE_COL] = pd.to_datetime(out[DATE_COL], errors="coerce")
    out[GROUP_COL] = out[GROUP_COL].astype(str).str.strip()

    for col in [TARGET_COL] + M4_FEATURES:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out


def assert_required_columns(df: pd.DataFrame, cols: list[str], context: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in {context}: {missing}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    residual = y_true - y_pred
    sse = float(np.sum(residual ** 2))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))

    return {
        "n_obs": int(len(y_true)),
        "rmse": float(np.sqrt(np.mean(residual ** 2))),
        "mae": float(np.mean(np.abs(residual))),
        "r2": float(1.0 - sse / sst) if sst > 0 else np.nan,
        "sse": sse,
        "mean_residual": float(np.mean(residual)),
        "std_residual": float(np.std(residual, ddof=1)),
        "p01_residual": float(np.quantile(residual, 0.01)),
        "p05_residual": float(np.quantile(residual, 0.05)),
        "p50_residual": float(np.quantile(residual, 0.50)),
        "p95_residual": float(np.quantile(residual, 0.95)),
        "p99_residual": float(np.quantile(residual, 0.99)),
    }


def oos_r2_from_sse(model_sse: float, benchmark_sse: float) -> float:
    if benchmark_sse > 0:
        return float(1.0 - model_sse / benchmark_sse)
    return np.nan


def residual_autocorr_by_cusip(df: pd.DataFrame, residual_col: str) -> float:
    values = []

    for _, g in df.sort_values([GROUP_COL, DATE_COL]).groupby(GROUP_COL):
        r = pd.to_numeric(g[residual_col], errors="coerce").dropna()

        if len(r) < 3:
            continue

        if float(r.std(ddof=1)) == 0.0:
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ac = r.autocorr(lag=1)

        if pd.notna(ac) and np.isfinite(ac):
            values.append(float(ac))

    if not values:
        return np.nan

    return float(np.mean(values))


def fit_fe_context(
    train: pd.DataFrame,
    features: list[str],
    target: str = TARGET_COL,
    group_col: str = GROUP_COL,
) -> dict[str, Any]:
    train = train.copy()

    y_means = train.groupby(group_col)[target].mean()
    x_means = train.groupby(group_col)[features].mean()

    context = {
        "features": features,
        "target": target,
        "group_col": group_col,
        "y_means": y_means,
        "x_means": x_means,
        "global_y_mean": float(train[target].mean()),
        "global_x_mean": train[features].mean(),
    }

    return context


def y_base(df: pd.DataFrame, context: dict[str, Any]) -> np.ndarray:
    group_col = context["group_col"]
    return (
        df[group_col]
        .map(context["y_means"])
        .fillna(context["global_y_mean"])
        .to_numpy(dtype=float)
    )


def center_y(df: pd.DataFrame, context: dict[str, Any]) -> np.ndarray:
    y = pd.to_numeric(df[context["target"]], errors="coerce").to_numpy(dtype=float)
    return y - y_base(df, context)


def center_x(df: pd.DataFrame, context: dict[str, Any]) -> pd.DataFrame:
    features = context["features"]
    group_col = context["group_col"]

    X = df[features].astype(float)

    X_bar = pd.DataFrame(index=df.index)

    for feature in features:
        X_bar[feature] = (
            df[group_col]
            .map(context["x_means"][feature])
            .fillna(context["global_x_mean"][feature])
        )

    X_centered = X - X_bar
    return X_centered


def fit_m0_context(
    train: pd.DataFrame,
    target: str = TARGET_COL,
    group_col: str = GROUP_COL,
) -> dict[str, Any]:
    return {
        "target": target,
        "group_col": group_col,
        "y_means": train.groupby(group_col)[target].mean(),
        "global_y_mean": float(train[target].mean()),
    }


def predict_m0(df: pd.DataFrame, context: dict[str, Any]) -> np.ndarray:
    return (
        df[context["group_col"]]
        .map(context["y_means"])
        .fillna(context["global_y_mean"])
        .to_numpy(dtype=float)
    )


class BaseWithinModel:
    def fit(self, X_train_centered: pd.DataFrame, y_train_centered: np.ndarray) -> "BaseWithinModel":
        raise NotImplementedError

    def predict_centered(self, X_centered: pd.DataFrame) -> np.ndarray:
        raise NotImplementedError

    def coefficient_rows(self, model_name: str) -> list[dict[str, Any]]:
        return []

    def pca_loading_rows(self, model_name: str) -> list[dict[str, Any]]:
        return []

    def pca_variance_rows(self, model_name: str) -> list[dict[str, Any]]:
        return []


class OLSWithinModel(BaseWithinModel):
    def __init__(self, features: list[str]):
        self.features = features
        self.model = LinearRegression(fit_intercept=False)

    def fit(self, X_train_centered: pd.DataFrame, y_train_centered: np.ndarray) -> "OLSWithinModel":
        self.model.fit(X_train_centered[self.features].to_numpy(dtype=float), y_train_centered)
        self.coef_ = pd.Series(self.model.coef_, index=self.features)
        return self

    def predict_centered(self, X_centered: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X_centered[self.features].to_numpy(dtype=float))

    def coefficient_rows(self, model_name: str) -> list[dict[str, Any]]:
        return [
            {
                "model": model_name,
                "coefficient_type": "original_feature",
                "feature": feature,
                "coefficient": float(coef),
            }
            for feature, coef in self.coef_.items()
        ]


class ScaledPenalisedWithinModel(BaseWithinModel):
    def __init__(
        self,
        features: list[str],
        estimator: Any,
        estimator_name: str,
    ):
        self.features = features
        self.estimator = estimator
        self.estimator_name = estimator_name
        self.scaler = StandardScaler(with_mean=True, with_std=True)

    def fit(self, X_train_centered: pd.DataFrame, y_train_centered: np.ndarray) -> "ScaledPenalisedWithinModel":
        X = X_train_centered[self.features].to_numpy(dtype=float)
        X_scaled = self.scaler.fit_transform(X)
        self.estimator.fit(X_scaled, y_train_centered)

        coef_scaled = np.asarray(self.estimator.coef_, dtype=float)
        coef_original = coef_scaled / self.scaler.scale_

        self.coef_scaled_ = pd.Series(coef_scaled, index=self.features)
        self.coef_original_ = pd.Series(coef_original, index=self.features)

        return self

    def predict_centered(self, X_centered: pd.DataFrame) -> np.ndarray:
        X = X_centered[self.features].to_numpy(dtype=float)
        X_scaled = self.scaler.transform(X)
        return self.estimator.predict(X_scaled)

    def coefficient_rows(self, model_name: str) -> list[dict[str, Any]]:
        rows = []

        for feature in self.features:
            rows.append(
                {
                    "model": model_name,
                    "coefficient_type": "scaled_feature",
                    "feature": feature,
                    "coefficient": float(self.coef_scaled_.loc[feature]),
                }
            )
            rows.append(
                {
                    "model": model_name,
                    "coefficient_type": "original_feature_backtransformed",
                    "feature": feature,
                    "coefficient": float(self.coef_original_.loc[feature]),
                }
            )

        return rows


class BlockPCRWithinModel(BaseWithinModel):

    def __init__(
        self,
        features: list[str],
        pca_blocks: dict[str, tuple[list[str], int]],
        model_label: str,
    ):
        self.features = features
        self.pca_blocks = pca_blocks
        self.model_label = model_label
        self.model = LinearRegression(fit_intercept=False)

        self.block_scalers: dict[str, StandardScaler] = {}
        self.block_pcas: dict[str, PCA] = {}
        self.transformed_feature_names: list[str] = []

    def _transform(self, X_centered: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        parts = []
        used_features = set()
        transformed_names = []

        for block_name, (block_features, n_components) in self.pca_blocks.items():
            block_features = [f for f in block_features if f in X_centered.columns]
            used_features.update(block_features)

            X_block = X_centered[block_features].to_numpy(dtype=float)

            if fit:
                scaler = StandardScaler(with_mean=True, with_std=True)
                X_scaled = scaler.fit_transform(X_block)

                pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
                X_pca = pca.fit_transform(X_scaled)

                self.block_scalers[block_name] = scaler
                self.block_pcas[block_name] = pca
            else:
                scaler = self.block_scalers[block_name]
                pca = self.block_pcas[block_name]
                X_scaled = scaler.transform(X_block)
                X_pca = pca.transform(X_scaled)

            pc_cols = [
                f"{block_name}_PC{k + 1}"
                for k in range(n_components)
            ]

            transformed_names.extend(pc_cols)
            parts.append(pd.DataFrame(X_pca, index=X_centered.index, columns=pc_cols))

        remaining_features = [f for f in self.features if f not in used_features]
        if remaining_features:
            parts.append(X_centered[remaining_features].copy())
            transformed_names.extend(remaining_features)

        out = pd.concat(parts, axis=1)

        if fit:
            self.transformed_feature_names = transformed_names

        return out[self.transformed_feature_names] if not fit else out

    def fit(self, X_train_centered: pd.DataFrame, y_train_centered: np.ndarray) -> "BlockPCRWithinModel":
        Z_train = self._transform(X_train_centered, fit=True)
        self.model.fit(Z_train.to_numpy(dtype=float), y_train_centered)
        self.coef_ = pd.Series(self.model.coef_, index=Z_train.columns)
        return self

    def predict_centered(self, X_centered: pd.DataFrame) -> np.ndarray:
        Z = self._transform(X_centered, fit=False)
        return self.model.predict(Z.to_numpy(dtype=float))

    def coefficient_rows(self, model_name: str) -> list[dict[str, Any]]:
        return [
            {
                "model": model_name,
                "coefficient_type": "transformed_feature",
                "feature": feature,
                "coefficient": float(coef),
            }
            for feature, coef in self.coef_.items()
        ]

    def pca_loading_rows(self, model_name: str) -> list[dict[str, Any]]:
        rows = []

        for block_name, pca in self.block_pcas.items():
            block_features, _ = self.pca_blocks[block_name]

            for component_idx, component in enumerate(pca.components_, start=1):
                for feature, loading in zip(block_features, component):
                    rows.append(
                        {
                            "model": model_name,
                            "pca_block": block_name,
                            "component": component_idx,
                            "feature": feature,
                            "loading": float(loading),
                        }
                    )

        return rows

    def pca_variance_rows(self, model_name: str) -> list[dict[str, Any]]:
        rows = []

        for block_name, pca in self.block_pcas.items():
            for component_idx, ratio in enumerate(pca.explained_variance_ratio_, start=1):
                rows.append(
                    {
                        "model": model_name,
                        "pca_block": block_name,
                        "component": component_idx,
                        "explained_variance_ratio": float(ratio),
                        "cumulative_explained_variance_ratio": float(
                            np.sum(pca.explained_variance_ratio_[:component_idx])
                        ),
                    }
                )

        return rows


class FullPCRWithinModel(BaseWithinModel):
    def __init__(self, features: list[str], n_components: int):
        self.features = features
        self.n_components = n_components
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        self.pca = PCA(n_components=n_components, random_state=RANDOM_SEED)
        self.model = LinearRegression(fit_intercept=False)

    def fit(self, X_train_centered: pd.DataFrame, y_train_centered: np.ndarray) -> "FullPCRWithinModel":
        X = X_train_centered[self.features].to_numpy(dtype=float)
        X_scaled = self.scaler.fit_transform(X)
        X_pca = self.pca.fit_transform(X_scaled)

        self.pc_names = [f"FullDesign_PC{k + 1}" for k in range(self.n_components)]
        self.model.fit(X_pca, y_train_centered)
        self.coef_ = pd.Series(self.model.coef_, index=self.pc_names)

        return self

    def predict_centered(self, X_centered: pd.DataFrame) -> np.ndarray:
        X = X_centered[self.features].to_numpy(dtype=float)
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        return self.model.predict(X_pca)

    def coefficient_rows(self, model_name: str) -> list[dict[str, Any]]:
        return [
            {
                "model": model_name,
                "coefficient_type": "full_pca_component",
                "feature": feature,
                "coefficient": float(coef),
            }
            for feature, coef in self.coef_.items()
        ]

    def pca_loading_rows(self, model_name: str) -> list[dict[str, Any]]:
        rows = []

        for component_idx, component in enumerate(self.pca.components_, start=1):
            for feature, loading in zip(self.features, component):
                rows.append(
                    {
                        "model": model_name,
                        "pca_block": "full_design",
                        "component": component_idx,
                        "feature": feature,
                        "loading": float(loading),
                    }
                )

        return rows

    def pca_variance_rows(self, model_name: str) -> list[dict[str, Any]]:
        rows = []

        for component_idx, ratio in enumerate(self.pca.explained_variance_ratio_, start=1):
            rows.append(
                {
                    "model": model_name,
                    "pca_block": "full_design",
                    "component": component_idx,
                    "explained_variance_ratio": float(ratio),
                    "cumulative_explained_variance_ratio": float(
                        np.sum(self.pca.explained_variance_ratio_[:component_idx])
                    ),
                }
            )

        return rows


def evaluate_fitted_model(
    model_name: str,
    model_group: str,
    model_family: str,
    hyperparameters: dict[str, Any],
    fitted_model: BaseWithinModel,
    fe_context: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    m0_predictions: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], dict[str, pd.DataFrame]]:
    rows = []
    pred_frames = {}

    for split_name, split_df in split_frames.items():
        y_true = split_df[TARGET_COL].to_numpy(dtype=float)

        Xc = center_x(split_df, fe_context)
        y_pred_centered = fitted_model.predict_centered(Xc)
        y_pred = y_base(split_df, fe_context) + y_pred_centered

        m0_pred = m0_predictions[split_name]
        metrics = compute_metrics(y_true, y_pred)
        m0_metrics = compute_metrics(y_true, m0_pred)

        row = {
            "model": model_name,
            "model_group": model_group,
            "model_family": model_family,
            "split": split_name,
            "target": TARGET_COL,
            "n_original_features": len(M4_FEATURES),
            "hyperparameters_json": json.dumps(json_safe(hyperparameters), sort_keys=True),
            **metrics,
            "m0_sse_same_sample": m0_metrics["sse"],
            "oos_r2_vs_m0_same_sample": oos_r2_from_sse(metrics["sse"], m0_metrics["sse"]),
        }

        rows.append(row)

        pred = split_df[["_sample_index", GROUP_COL, DATE_COL, SPLIT_COL]].copy()
        pred["model"] = model_name
        pred["model_group"] = model_group
        pred["model_family"] = model_family
        pred["target_value"] = y_true
        pred["fitted_value"] = y_pred
        pred["residual"] = y_true - y_pred

        pred_frames[split_name] = pred

    return rows, pred_frames


def fit_and_evaluate_candidate(
    model_name: str,
    model_group: str,
    model_family: str,
    hyperparameters: dict[str, Any],
    model_obj: BaseWithinModel,
    fe_context: dict[str, Any],
    split_frames: dict[str, pd.DataFrame],
    m0_predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    train = split_frames["train"]
    X_train_centered = center_x(train, fe_context)
    y_train_centered = center_y(train, fe_context)

    fitted = model_obj.fit(X_train_centered, y_train_centered)

    result_rows, pred_frames = evaluate_fitted_model(
        model_name=model_name,
        model_group=model_group,
        model_family=model_family,
        hyperparameters=hyperparameters,
        fitted_model=fitted,
        fe_context=fe_context,
        split_frames=split_frames,
        m0_predictions=m0_predictions,
    )

    return {
        "model_name": model_name,
        "model_group": model_group,
        "model_family": model_family,
        "hyperparameters": hyperparameters,
        "fitted_model": fitted,
        "result_rows": result_rows,
        "prediction_frames": pred_frames,
    }


def select_best_candidate(
    candidate_objects: list[dict[str, Any]],
    selection_split: str = "validation",
    metric: str = "rmse",
) -> dict[str, Any]:
    if not candidate_objects:
        raise ValueError("No candidate objects supplied for selection.")

    ranking_rows = []

    for obj in candidate_objects:
        rows = pd.DataFrame(obj["result_rows"])
        selected = rows.loc[rows["split"].eq(selection_split)].copy()

        if selected.empty:
            continue

        ranking_rows.append(
            {
                "model_name": obj["model_name"],
                "model_group": obj["model_group"],
                "selection_metric": metric,
                "selection_split": selection_split,
                "selection_value": float(selected.iloc[0][metric]),
            }
        )

    ranking = pd.DataFrame(ranking_rows)

    if ranking.empty:
        raise ValueError("No validation rows available for candidate selection.")

    best_name = (
        ranking.sort_values(["selection_value", "model_name"], ascending=[True, True])
        .iloc[0]["model_name"]
    )

    for obj in candidate_objects:
        if obj["model_name"] == best_name:
            obj["selection_ranking"] = ranking
            return obj

    raise RuntimeError("Could not recover best candidate object.")


def main() -> None:
    ensure_directories()

    print("\nREGULARISED AND PCA FAIR-VALUE MODELS")
    print(f"Reading panel: {PANEL_PATH}")

    panel = pd.read_parquet(PANEL_PATH)
    panel = prepare_panel(panel)
    panel = assign_sample_split(panel)
    panel["_sample_index"] = np.arange(len(panel))

    assert_required_columns(
        panel,
        [GROUP_COL, DATE_COL, TARGET_COL] + M4_FEATURES,
        context="regularised model panel",
    )

    required = [GROUP_COL, DATE_COL, TARGET_COL, SPLIT_COL] + M4_FEATURES
    sample = panel[required + ["_sample_index"]].dropna(subset=[TARGET_COL] + M4_FEATURES).copy()

    sample = sample.sort_values([GROUP_COL, DATE_COL]).reset_index(drop=True)

    split_frames = {
        "train": sample.loc[sample[SPLIT_COL].eq("train")].copy(),
        "validation": sample.loc[sample[SPLIT_COL].eq("validation")].copy(),
        "test": sample.loc[sample[SPLIT_COL].eq("test")].copy(),
    }

    print("\nCommon M4 complete-case sample:")
    print("Rows:", len(sample))
    print("CUSIPs:", sample[GROUP_COL].nunique())
    print("Date range:", sample[DATE_COL].min(), "to", sample[DATE_COL].max())
    print("Split counts:")
    print(sample[SPLIT_COL].value_counts(dropna=False).sort_index())

    for split_name, split_df in split_frames.items():
        if split_df.empty:
            raise ValueError(f"Empty split: {split_name}")

    fe_context = fit_fe_context(split_frames["train"], features=M4_FEATURES)

    m0_context = fit_m0_context(split_frames["train"])
    m0_predictions = {
        split_name: predict_m0(split_df, m0_context)
        for split_name, split_df in split_frames.items()
    }

    all_candidate_rows = []
    selected_objects = []
    coefficient_rows = []
    pca_loading_rows = []
    pca_variance_rows = []
    prediction_parts = []
    selection_rows = []

    print("\nFitting OLS_FE_M4")

    ols_obj = fit_and_evaluate_candidate(
        model_name="OLS_FE_M4",
        model_group="OLS_FE_M4",
        model_family="ols",
        hyperparameters={},
        model_obj=OLSWithinModel(features=M4_FEATURES),
        fe_context=fe_context,
        split_frames=split_frames,
        m0_predictions=m0_predictions,
    )

    selected_objects.append(ols_obj)

    ridge_candidates = []

    for alpha in RIDGE_ALPHAS:
        name = f"Ridge_FE_M4_alpha_{alpha:g}"
        print(f"Fitting {name}")

        obj = fit_and_evaluate_candidate(
            model_name=name,
            model_group="Ridge_FE_M4",
            model_family="ridge",
            hyperparameters={"alpha": alpha},
            model_obj=ScaledPenalisedWithinModel(
                features=M4_FEATURES,
                estimator=Ridge(alpha=alpha, fit_intercept=False, random_state=RANDOM_SEED),
                estimator_name="ridge",
            ),
            fe_context=fe_context,
            split_frames=split_frames,
            m0_predictions=m0_predictions,
        )

        ridge_candidates.append(obj)

    best_ridge = select_best_candidate(ridge_candidates, metric="rmse")
    best_ridge["model_name"] = "Ridge_FE_M4"
    selected_objects.append(best_ridge)

    lasso_candidates = []

    for alpha in LASSO_ALPHAS:
        name = f"Lasso_FE_M4_alpha_{alpha:g}"
        print(f"Fitting {name}")

        obj = fit_and_evaluate_candidate(
            model_name=name,
            model_group="Lasso_FE_M4",
            model_family="lasso",
            hyperparameters={"alpha": alpha},
            model_obj=ScaledPenalisedWithinModel(
                features=M4_FEATURES,
                estimator=Lasso(
                    alpha=alpha,
                    fit_intercept=False,
                    max_iter=20000,
                    random_state=RANDOM_SEED,
                    selection="cyclic",
                ),
                estimator_name="lasso",
            ),
            fe_context=fe_context,
            split_frames=split_frames,
            m0_predictions=m0_predictions,
        )

        lasso_candidates.append(obj)

    best_lasso = select_best_candidate(lasso_candidates, metric="rmse")
    best_lasso["model_name"] = "Lasso_FE_M4"
    selected_objects.append(best_lasso)

    elastic_candidates = []

    for alpha in ELASTIC_NET_ALPHAS:
        for l1_ratio in ELASTIC_NET_L1_RATIOS:
            name = f"ElasticNet_FE_M4_alpha_{alpha:g}_l1_{l1_ratio:g}"
            print(f"Fitting {name}")

            obj = fit_and_evaluate_candidate(
                model_name=name,
                model_group="ElasticNet_FE_M4",
                model_family="elastic_net",
                hyperparameters={"alpha": alpha, "l1_ratio": l1_ratio},
                model_obj=ScaledPenalisedWithinModel(
                    features=M4_FEATURES,
                    estimator=ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        fit_intercept=False,
                        max_iter=20000,
                        random_state=RANDOM_SEED,
                        selection="cyclic",
                    ),
                    estimator_name="elastic_net",
                ),
                fe_context=fe_context,
                split_frames=split_frames,
                m0_predictions=m0_predictions,
            )

            elastic_candidates.append(obj)

    best_elastic = select_best_candidate(elastic_candidates, metric="rmse")
    best_elastic["model_name"] = "ElasticNet_FE_M4"
    selected_objects.append(best_elastic)

    pcr_rates_objects = []

    for n_components in PCR_RATES_COMPONENTS:
        name = f"PCR_Rates_{n_components}PC_M4"
        print(f"Fitting {name}")

        obj = fit_and_evaluate_candidate(
            model_name=name,
            model_group=name,
            model_family="pcr_rates",
            hyperparameters={"rates_n_components": n_components},
            model_obj=BlockPCRWithinModel(
                features=M4_FEATURES,
                pca_blocks={"rates": (RATES_FEATURES, n_components)},
                model_label=name,
            ),
            fe_context=fe_context,
            split_frames=split_frames,
            m0_predictions=m0_predictions,
        )

        selected_objects.append(obj)
        pcr_rates_objects.append(obj)

    best_rates_pcr = select_best_candidate(pcr_rates_objects, metric="rmse")
    best_rates_n = int(best_rates_pcr["hyperparameters"]["rates_n_components"])

    print(f"\nBest rates PCA by validation RMSE: {best_rates_n} component(s)")

    for n_components in PCR_PEER_COMPONENTS:
        name = f"PCR_Peer_{n_components}PC_M4"
        print(f"Fitting {name}")

        obj = fit_and_evaluate_candidate(
            model_name=name,
            model_group=name,
            model_family="pcr_peer",
            hyperparameters={"peer_n_components": n_components},
            model_obj=BlockPCRWithinModel(
                features=M4_FEATURES,
                pca_blocks={"peer": (PEER_FEATURES_RAW, n_components)},
                model_label=name,
            ),
            fe_context=fe_context,
            split_frames=split_frames,
            m0_predictions=m0_predictions,
        )

        selected_objects.append(obj)

    for peer_components in PCR_PEER_COMPONENTS:
        name = f"PCR_RatesBest{best_rates_n}PC_Peer_{peer_components}PC_M4"
        print(f"Fitting {name}")

        obj = fit_and_evaluate_candidate(
            model_name=name,
            model_group=name,
            model_family="pcr_rates_best_peer",
            hyperparameters={
                "rates_n_components": best_rates_n,
                "peer_n_components": peer_components,
                "rates_n_selected_on": "validation_rmse",
            },
            model_obj=BlockPCRWithinModel(
                features=M4_FEATURES,
                pca_blocks={
                    "rates": (RATES_FEATURES, best_rates_n),
                    "peer": (PEER_FEATURES_RAW, peer_components),
                },
                model_label=name,
            ),
            fe_context=fe_context,
            split_frames=split_frames,
            m0_predictions=m0_predictions,
        )

        selected_objects.append(obj)

    for n_components in PCR_FULL_COMPONENTS:
        name = f"PCR_Full_{n_components}PC"
        print(f"Fitting {name}")

        obj = fit_and_evaluate_candidate(
            model_name=name,
            model_group=name,
            model_family="pcr_full",
            hyperparameters={"full_design_n_components": n_components},
            model_obj=FullPCRWithinModel(
                features=M4_FEATURES,
                n_components=n_components,
            ),
            fe_context=fe_context,
            split_frames=split_frames,
            m0_predictions=m0_predictions,
        )

        selected_objects.append(obj)

    for obj in [ols_obj] + ridge_candidates + lasso_candidates + elastic_candidates + pcr_rates_objects:
        all_candidate_rows.extend(obj["result_rows"])

    selected_names_already = {obj["model_name"] for obj in [ols_obj] + pcr_rates_objects}
    for obj in selected_objects:
        if obj["model_name"] not in selected_names_already and obj["model_family"].startswith("pcr"):
            all_candidate_rows.extend(obj["result_rows"])

    selected_result_rows = []

    for obj in selected_objects:
        selected_name = obj["model_name"]
        fitted_model = obj["fitted_model"]

        result_rows, pred_frames = evaluate_fitted_model(
            model_name=selected_name,
            model_group=obj["model_group"],
            model_family=obj["model_family"],
            hyperparameters=obj["hyperparameters"],
            fitted_model=fitted_model,
            fe_context=fe_context,
            split_frames=split_frames,
            m0_predictions=m0_predictions,
        )

        selected_result_rows.extend(result_rows)
        coefficient_rows.extend(fitted_model.coefficient_rows(selected_name))
        pca_loading_rows.extend(fitted_model.pca_loading_rows(selected_name))
        pca_variance_rows.extend(fitted_model.pca_variance_rows(selected_name))

        if "selection_ranking" in obj:
            ranking = obj["selection_ranking"].copy()
            ranking["selected_model"] = selected_name
            selection_rows.append(ranking)

        for split_name, pred in pred_frames.items():
            if split_name in SAVE_PREDICTIONS_FOR_SPLITS:
                prediction_parts.append(pred)

    candidate_results = pd.DataFrame(all_candidate_rows)
    selected_results = pd.DataFrame(selected_result_rows)

    residual_diag_rows = []

    if prediction_parts:
        predictions = pd.concat(prediction_parts, ignore_index=True)
    else:
        predictions = pd.DataFrame()

    if not predictions.empty:
        for (model_name, split_name), g in predictions.groupby(["model", SPLIT_COL]):
            residual_diag_rows.append(
                {
                    "model": model_name,
                    "split": split_name,
                    "n_obs": int(len(g)),
                    "residual_mean": float(g["residual"].mean()),
                    "residual_std": float(g["residual"].std(ddof=1)),
                    "residual_abs_mean": float(g["residual"].abs().mean()),
                    "residual_p01": float(g["residual"].quantile(0.01)),
                    "residual_p05": float(g["residual"].quantile(0.05)),
                    "residual_p50": float(g["residual"].quantile(0.50)),
                    "residual_p95": float(g["residual"].quantile(0.95)),
                    "residual_p99": float(g["residual"].quantile(0.99)),
                    "mean_cusip_lag1_autocorr": residual_autocorr_by_cusip(
                        g,
                        residual_col="residual",
                    ),
                }
            )

    residual_diagnostics = pd.DataFrame(residual_diag_rows)

    coefficients = pd.DataFrame(coefficient_rows)
    pca_loadings = pd.DataFrame(pca_loading_rows)
    pca_variance = pd.DataFrame(pca_variance_rows)

    if selection_rows:
        selected_summary = pd.concat(selection_rows, ignore_index=True)
    else:
        selected_summary = pd.DataFrame()

    print("\nSaving outputs...")

    candidate_results.to_csv(OUTPUT_CANDIDATE_RESULTS, index=False)
    selected_results.to_csv(OUTPUT_SELECTED_RESULTS, index=False)
    selected_summary.to_csv(OUTPUT_SELECTED_SUMMARY, index=False)
    coefficients.to_csv(OUTPUT_COEFFICIENTS, index=False)
    residual_diagnostics.to_csv(OUTPUT_RESIDUAL_DIAGNOSTICS, index=False)
    pca_loadings.to_csv(OUTPUT_PCA_LOADINGS, index=False)
    pca_variance.to_csv(OUTPUT_PCA_EXPLAINED_VARIANCE, index=False)

    if not predictions.empty:
        predictions.to_parquet(OUTPUT_PREDICTIONS, index=False)

    manifest = {
        "script": Path(__file__).name,
        "panel_path": str(PANEL_PATH),
        "target": TARGET_COL,
        "common_sample_rows": int(len(sample)),
        "common_sample_cusips": int(sample[GROUP_COL].nunique()),
        "split_counts": sample[SPLIT_COL].value_counts(dropna=False).to_dict(),
        "m4_features": M4_FEATURES,
        "rates_features": RATES_FEATURES,
        "peer_features": PEER_FEATURES_RAW,
        "ridge_alphas": RIDGE_ALPHAS,
        "lasso_alphas": LASSO_ALPHAS,
        "elastic_net_alphas": ELASTIC_NET_ALPHAS,
        "elastic_net_l1_ratios": ELASTIC_NET_L1_RATIOS,
        "pcr_rates_components": PCR_RATES_COMPONENTS,
        "pcr_peer_components": PCR_PEER_COMPONENTS,
        "pcr_full_components": PCR_FULL_COMPONENTS,
        "best_rates_pca_components_by_validation_rmse": best_rates_n,
        "train_end_date": str(TRAIN_END_DATE.date()),
        "validation_end_date": str(VALIDATION_END_DATE.date()),
        "outputs": {
            "candidate_results": str(OUTPUT_CANDIDATE_RESULTS),
            "selected_results": str(OUTPUT_SELECTED_RESULTS),
            "selected_summary": str(OUTPUT_SELECTED_SUMMARY),
            "coefficients": str(OUTPUT_COEFFICIENTS),
            "residual_diagnostics": str(OUTPUT_RESIDUAL_DIAGNOSTICS),
            "pca_loadings": str(OUTPUT_PCA_LOADINGS),
            "pca_explained_variance": str(OUTPUT_PCA_EXPLAINED_VARIANCE),
            "predictions": str(OUTPUT_PREDICTIONS),
        },
    }

    with open(OUTPUT_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(json_safe(manifest), fh, indent=2)

    print(f"Saved: {OUTPUT_CANDIDATE_RESULTS}")
    print(f"Saved: {OUTPUT_SELECTED_RESULTS}")
    print(f"Saved: {OUTPUT_SELECTED_SUMMARY}")
    print(f"Saved: {OUTPUT_COEFFICIENTS}")
    print(f"Saved: {OUTPUT_RESIDUAL_DIAGNOSTICS}")
    print(f"Saved: {OUTPUT_PCA_LOADINGS}")
    print(f"Saved: {OUTPUT_PCA_EXPLAINED_VARIANCE}")
    if not predictions.empty:
        print(f"Saved: {OUTPUT_PREDICTIONS}")
    print(f"Saved: {OUTPUT_MANIFEST}")

    make_figures(selected_results)

    print("\nSELECTED MODEL TEST PERFORMANCE")
    test_perf = (
        selected_results.loc[selected_results["split"].eq("test")]
        .sort_values("oos_r2_vs_m0_same_sample", ascending=False)
        [
            [
                "model",
                "model_family",
                "rmse",
                "mae",
                "r2",
                "oos_r2_vs_m0_same_sample",
                "std_residual",
            ]
        ]
    )
    print(test_perf.to_string(index=False))

    print("\nDONE.")


def make_figures(selected_results: pd.DataFrame) -> None:
    if selected_results.empty:
        return

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    test = selected_results.loc[selected_results["split"].eq("test")].copy()
    validation = selected_results.loc[selected_results["split"].eq("validation")].copy()

    if test.empty:
        return

    test = test.sort_values("oos_r2_vs_m0_same_sample", ascending=False).reset_index(drop=True)

    plt.figure(figsize=(11, 5.5))
    plt.bar(np.arange(len(test)), test["oos_r2_vs_m0_same_sample"].to_numpy())
    plt.axhline(0.0, linewidth=1)
    plt.xticks(np.arange(len(test)), test["model"], rotation=90)
    plt.ylabel("Test OOS R2 versus M0 fixed effects")
    plt.title("Regularised and PCA fair-value models: test OOS R2")
    plt.tight_layout()
    plt.savefig(FIG_TEST_OOS_R2, dpi=300, bbox_inches="tight")
    plt.close()

    if not validation.empty:
        merged = validation[["model", "oos_r2_vs_m0_same_sample"]].rename(
            columns={"oos_r2_vs_m0_same_sample": "validation_oos_r2"}
        ).merge(
            test[["model", "oos_r2_vs_m0_same_sample"]].rename(
                columns={"oos_r2_vs_m0_same_sample": "test_oos_r2"}
            ),
            on="model",
            how="inner",
        )

        plt.figure(figsize=(6.5, 5.5))
        plt.scatter(merged["validation_oos_r2"], merged["test_oos_r2"])

        for _, row in merged.iterrows():
            plt.annotate(
                row["model"],
                (row["validation_oos_r2"], row["test_oos_r2"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

        plt.axhline(0.0, linewidth=1)
        plt.axvline(0.0, linewidth=1)
        plt.xlabel("Validation OOS R2 versus M0")
        plt.ylabel("Test OOS R2 versus M0")
        plt.title("Validation versus test performance")
        plt.tight_layout()
        plt.savefig(FIG_VALIDATION_TEST_OOS_R2, dpi=300, bbox_inches="tight")
        plt.close()

    print(f"Saved: {FIG_TEST_OOS_R2}")
    print(f"Saved: {FIG_VALIDATION_TEST_OOS_R2}")


if __name__ == "__main__":
    main()