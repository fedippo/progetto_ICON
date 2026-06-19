"""Supervised learning evaluation for cluster prediction."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    CLASS_DISTRIBUTION_PATH,
    CLUSTERED_NORMALIZED_DATA_PATH,
    CV_REPEATS,
    CV_SPLITS,
    RANDOM_STATE,
    SUPERVISED_BEST_PARAMS_PATH,
    SUPERVISED_FEATURES,
    SUPERVISED_METRICS_PATH,
    TARGET_CLUSTER_COLUMN,
)


def import_ml_dependencies():
    """Import ML libraries lazily to keep the module inspectable without installs."""
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import make_scorer
        from sklearn.metrics import precision_score, recall_score, f1_score
        from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder
        from sklearn.svm import SVC
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing supervised-learning dependencies. Install requirements.txt before "
            "running this script: scikit-learn and imbalanced-learn are required."
        ) from exc

    return {
        "SMOTE": SMOTE,
        "ImbPipeline": ImbPipeline,
        "ColumnTransformer": ColumnTransformer,
        "RandomForestClassifier": RandomForestClassifier,
        "make_scorer": make_scorer,
        "precision_score": precision_score,
        "recall_score": recall_score,
        "f1_score": f1_score,
        "GridSearchCV": GridSearchCV,
        "RepeatedStratifiedKFold": RepeatedStratifiedKFold,
        "Pipeline": Pipeline,
        "OneHotEncoder": OneHotEncoder,
        "SVC": SVC,
    }


def load_clustered_dataset() -> pd.DataFrame:
    """Load the normalized dataset enriched with KMeans labels."""
    data_path = Path(CLUSTERED_NORMALIZED_DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Clustered dataset not found at {data_path}. Run clustering.py first."
        )
    return pd.read_csv(data_path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build X and y for supervised learning."""
    missing = [
        column
        for column in SUPERVISED_FEATURES + [TARGET_CLUSTER_COLUMN]
        if column not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing supervised-learning columns: {missing}")

    x = df[SUPERVISED_FEATURES].copy()
    y = df[TARGET_CLUSTER_COLUMN].copy()
    return x, y


def save_class_distribution(y: pd.Series) -> pd.DataFrame:
    """Save absolute and relative target class distribution."""
    distribution = (
        y.value_counts()
        .sort_index()
        .rename_axis(TARGET_CLUSTER_COLUMN)
        .reset_index(name="Count")
    )
    distribution["Share"] = distribution["Count"] / distribution["Count"].sum()
    Path(CLASS_DISTRIBUTION_PATH).parent.mkdir(parents=True, exist_ok=True)
    distribution.to_csv(CLASS_DISTRIBUTION_PATH, index=False)
    return distribution


def build_preprocessor(ml, x: pd.DataFrame):
    """Encode the categorical genre feature and pass numeric columns through."""
    categorical_columns = ["Primary_Genre"]
    numeric_columns = [column for column in x.columns if column not in categorical_columns]

    return ml["ColumnTransformer"](
        transformers=[
            (
                "genre",
                ml["OneHotEncoder"](handle_unknown="ignore"),
                categorical_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ]
    )


def build_model_specs(ml, preprocessor):
    """Define models, compact grids and SMOTE variants."""
    random_forest = ml["RandomForestClassifier"](random_state=RANDOM_STATE)
    svm = ml["SVC"](random_state=RANDOM_STATE)

    return [
        {
            "name": "RandomForest",
            "use_smote": False,
            "pipeline_cls": ml["Pipeline"],
            "steps": [("preprocess", preprocessor), ("model", random_forest)],
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 3],
            },
        },
        {
            "name": "RandomForest",
            "use_smote": True,
            "pipeline_cls": ml["ImbPipeline"],
            "steps": [
                ("preprocess", preprocessor),
                ("smote", ml["SMOTE"](random_state=RANDOM_STATE)),
                ("model", random_forest),
            ],
            "param_grid": {
                "model__n_estimators": [100, 200],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 3],
            },
        },
        {
            "name": "SVM",
            "use_smote": False,
            "pipeline_cls": ml["Pipeline"],
            "steps": [("preprocess", preprocessor), ("model", svm)],
            "param_grid": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale"],
            },
        },
        {
            "name": "SVM",
            "use_smote": True,
            "pipeline_cls": ml["ImbPipeline"],
            "steps": [
                ("preprocess", preprocessor),
                ("smote", ml["SMOTE"](random_state=RANDOM_STATE)),
                ("model", svm),
            ],
            "param_grid": {
                "model__C": [0.1, 1, 10],
                "model__kernel": ["rbf", "linear"],
                "model__gamma": ["scale"],
            },
        },
    ]


def build_scoring(ml) -> dict:
    """Create the scoring dictionary required by GridSearchCV."""
    return {
        "accuracy": "accuracy",
        "precision_macro": ml["make_scorer"](
            ml["precision_score"], average="macro", zero_division=0
        ),
        "recall_macro": ml["make_scorer"](
            ml["recall_score"], average="macro", zero_division=0
        ),
        "f1_macro": ml["make_scorer"](
            ml["f1_score"], average="macro", zero_division=0
        ),
    }


def summarize_search_results(search, model_name: str, use_smote: bool) -> dict:
    """Extract mean and standard deviation for the best F1-macro model."""
    best_index = search.best_index_
    row = search.cv_results_

    return {
        "Model": model_name,
        "SMOTE": use_smote,
        "Accuracy_Mean": row["mean_test_accuracy"][best_index],
        "Accuracy_Std": row["std_test_accuracy"][best_index],
        "Precision_Macro_Mean": row["mean_test_precision_macro"][best_index],
        "Precision_Macro_Std": row["std_test_precision_macro"][best_index],
        "Recall_Macro_Mean": row["mean_test_recall_macro"][best_index],
        "Recall_Macro_Std": row["std_test_recall_macro"][best_index],
        "F1_Macro_Mean": row["mean_test_f1_macro"][best_index],
        "F1_Macro_Std": row["std_test_f1_macro"][best_index],
    }


def run_supervised_learning() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run model comparison with repeated stratified cross-validation."""
    ml = import_ml_dependencies()

    df = load_clustered_dataset()
    x, y = split_features_target(df)
    class_distribution = save_class_distribution(y)

    cv = ml["RepeatedStratifiedKFold"](
        n_splits=CV_SPLITS,
        n_repeats=CV_REPEATS,
        random_state=RANDOM_STATE,
    )
    scoring = build_scoring(ml)
    preprocessor = build_preprocessor(ml, x)

    metric_rows = []
    parameter_rows = []

    for spec in build_model_specs(ml, preprocessor):
        pipeline = spec["pipeline_cls"](steps=spec["steps"])
        search = ml["GridSearchCV"](
            estimator=pipeline,
            param_grid=spec["param_grid"],
            scoring=scoring,
            refit="f1_macro",
            cv=cv,
            n_jobs=-1,
            return_train_score=False,
        )
        search.fit(x, y)

        metric_rows.append(
            summarize_search_results(search, spec["name"], spec["use_smote"])
        )
        parameter_rows.append(
            {
                "Model": spec["name"],
                "SMOTE": spec["use_smote"],
                "Best_F1_Macro": search.best_score_,
                "Best_Params": search.best_params_,
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    params_df = pd.DataFrame(parameter_rows)

    Path(SUPERVISED_METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(SUPERVISED_METRICS_PATH, index=False)
    params_df.to_csv(SUPERVISED_BEST_PARAMS_PATH, index=False)

    return class_distribution, metrics_df, params_df


if __name__ == "__main__":
    distribution, metrics, params = run_supervised_learning()
    print("Class distribution")
    print(distribution.to_string(index=False))
    print("\nMetrics")
    print(metrics.round(4).to_string(index=False))
    print("\nBest parameters")
    print(params.to_string(index=False))
