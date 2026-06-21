"""Valutazione supervisionata per predire il cluster commerciale."""

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
    """Importa le librerie ML solo quando viene eseguita la valutazione.

    Questo rende il file leggibile anche in ambienti in cui le dipendenze non sono
    ancora installate. La funzione restituisce un dizionario con classi e funzioni
    usate dagli altri blocchi dello script.
    """
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
    """Carica il dataset normalizzato con la colonna `Cluster_Label`.

    Questo e il dataset usato dai modelli supervisionati: le feature sono gia
    scalate e il target e stato prodotto nella fase di clustering.
    """
    data_path = Path(CLUSTERED_NORMALIZED_DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Clustered dataset not found at {data_path}. Run clustering.py first."
        )
    return pd.read_csv(data_path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa le feature supervisionate dal target da predire.

    Verifica prima che tutte le colonne attese siano presenti. Restituisce `X`, con
    le feature definite in configurazione, e `y`, cioe la colonna `Cluster_Label`.
    """
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
    """Calcola e salva la distribuzione delle classi target.

    La tabella prodotta serve a capire se i cluster sono sbilanciati e quindi se ha
    senso confrontare i modelli anche con SMOTE.
    """
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
    """Costruisce il preprocessing da applicare dentro la pipeline ML.

    `Primary_Genre` viene codificato con one-hot encoding, mentre le feature gia
    numeriche passano senza ulteriori trasformazioni. Il preprocessing resta dentro
    la pipeline per essere applicato correttamente in ogni fold della CV.
    """
    categorical_columns = ["Primary_Genre"]
    numeric_columns = [column for column in x.columns if column not in categorical_columns]

    # OneHotEncoder e dentro la pipeline per evitare trasformazioni fuori CV.
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
    """Definisce le configurazioni sperimentali dei modelli.

    Per ogni modello vengono create due varianti: una standard e una con SMOTE. Le
    griglie di iperparametri sono compatte per mantenere il progetto eseguibile nel
    vincolo temporale, ma coprono i parametri piu rilevanti.
    """
    random_forest = ml["RandomForestClassifier"](random_state=RANDOM_STATE)
    svm = ml["SVC"](random_state=RANDOM_STATE)

    # Ogni modello viene valutato sia senza SMOTE sia con SMOTE nel training fold.
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
    """Crea il dizionario delle metriche usate da GridSearchCV.

    Oltre all'accuracy vengono usate precision, recall e F1 macro. Le metriche
    macro pesano ogni classe allo stesso modo e sono quindi adatte quando i cluster
    non hanno la stessa numerosita.
    """
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
    """Estrae i risultati mediati della migliore configurazione.

    GridSearchCV valuta molte combinazioni di iperparametri. Questa funzione prende
    la combinazione scelta tramite F1 macro e raccoglie media e deviazione standard
    di tutte le metriche, cosi la relazione non dipende da un singolo run.
    """
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
    """Esegue l'intera fase supervisionata e salva risultati e parametri.

    Carica il dataset clusterizzato, prepara feature e target, imposta
    RepeatedStratifiedKFold, valuta Random Forest e SVM con/senza SMOTE e salva le
    tabelle finali nella cartella `results`.
    """
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
        # La pipeline impedisce data leakage: preprocessing e SMOTE restano dentro la CV.
        pipeline = spec["pipeline_cls"](steps=spec["steps"])
        search = ml["GridSearchCV"](
            estimator=pipeline,
            param_grid=spec["param_grid"],
            scoring=scoring,
            # Il refit usa F1 macro per privilegiare il bilanciamento tra classi.
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
