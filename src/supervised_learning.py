"""Valutazione supervisionata per predire il cluster commerciale.

Questo modulo addestra modelli di Machine Learning (Random Forest e SVM) per classificare
i giochi nei cluster precedentemente individuati. Valuta anche l'efficacia del
sovracampionamento (SMOTE) per gestire eventuali sbilanciamenti tra le dimensioni dei cluster.
"""

from __future__ import annotations

from pathlib import Path
import gc

import pandas as pd
from joblib import dump

from config import (
    CATEGORICAL_FEATURES,
    CLASS_DISTRIBUTION_PATH,
    CLUSTERED_NORMALIZED_DATA_PATH,
    CV_REPEATS,
    CV_SPLITS,
    RANDOM_STATE,
    SUPERVISED_N_JOBS,
    SUPERVISED_PRE_DISPATCH,
    SUPERVISED_BEST_PARAMS_PATH,
    SUPERVISED_FEATURES,
    SUPERVISED_METRICS_PATH,
    SUPERVISED_MODEL_PATH,
    TARGET_CLUSTER_COLUMN,
)


def import_ml_dependencies():
    """Importa dinamicamente le dipendenze pesanti di Machine Learning.

    Perché farlo in una funzione:
    Scikit-learn e imbalanced-learn sono librerie voluminose. Importarle a livello globale
    rallenterebbe l'avvio dello script se si volesse solo ispezionare il modulo o eseguirne
    funzioni minori. Inoltre, questo blocco 'try-except' centralizzato restituisce un
    dizionario di dipendenze, garantendo un messaggio di errore chiaro e bloccante se
    l'ambiente Python non è configurato correttamente.
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
    """Carica il dataset preparato per i modelli supervisionati.

    Si utilizza specificamente la versione 'Normalized' generata in fase di preprocessing
    e arricchita con la label in fase di clustering. La normalizzazione è strettamente
    necessaria per l'algoritmo SVM (Support Vector Machine), che si basa sul calcolo delle
    distanze spaziali (margini) per tracciare gli iperpiani di separazione tra le classi.
    """
    data_path = Path(CLUSTERED_NORMALIZED_DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Clustered dataset not found at {data_path}. Run clustering.py first."
        )
    return pd.read_csv(data_path)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Isola le variabili indipendenti (X) dalla variabile dipendente/target (y).

    Esegue prima una validazione di integrità: verifica che tutte le colonne indicate
    in `SUPERVISED_FEATURES` e il `TARGET_CLUSTER_COLUMN` siano effettivamente presenti
    nel DataFrame. In caso contrario, solleva un'eccezione preventiva.
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
    """Calcola la numerosità di ciascun cluster per evidenziare sbilanciamenti.

    Logica analitica:
    L'algoritmo KMeans non garantisce che i cluster abbiano la stessa dimensione.
    Questa funzione calcola il conteggio assoluto ('Count') e la percentuale relativa ('Share')
    di ogni classe. Salvare questo dato è cruciale per la relazione finale, poiché
    giustifica l'utilizzo di metriche 'Macro' e l'introduzione di SMOTE.
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
    """Costruisce il trasformatore di colonne per le feature miste.

    Logica di Encoding:
    I modelli matematici non sanno leggere stringhe come 'Action' o 'RPG'.
    - `Primary_Genre`: Viene trasformato usando il 'OneHotEncoder'. Vengono create N colonne
      binarie indipendenti (una per ogni genere), evitando di introdurre relazioni ordinali fittizie
      (es. RPG=1, Action=2 non significa che Action > RPG).
    - `Multiplayer`: Essendo già una variabile binaria (0/1), viene bypassata ('passthrough')
      insieme alle restanti feature numeriche per evitare trasformazioni ridondanti.
    """
    one_hot_columns = [column for column in CATEGORICAL_FEATURES if column != "Multiplayer"]
    numeric_columns = [column for column in x.columns if column not in one_hot_columns]

    return ml["ColumnTransformer"](
        transformers=[
            (
                "genre",
                ml["OneHotEncoder"](handle_unknown="ignore"),
                one_hot_columns,
            ),
            ("numeric", "passthrough", numeric_columns),
        ]
    )


def build_model_specs(ml, preprocessor):
    """Progetta gli esperimenti: definisce modelli, iperparametri e architetture di pipeline.

    Strategia metodologica:
    Vengono testati due algoritmi diversi per approccio (Ensemble vs Margine Spaziale):
    - Random Forest: Robusto, gestisce bene relazioni non lineari.
    - SVM: Sensibile alla normalizzazione, eccellente nei confini decisionali complessi.

    Prevenzione del Data Leakage:
    Quando si usa SMOTE (creazione di dati sintetici), è fondamentale generare questi
    nuovi record SOLO sul set di addestramento (Train fold) e mai sul set di test (Validation fold).
    L'uso di `ImbPipeline` (da imbalanced-learn) garantisce matematicamente che lo SMOTE
    venga applicato unicamente durante la fase di '.fit()', mantenendo il test set immacolato.
    """
    random_forest = ml["RandomForestClassifier"](
        random_state=RANDOM_STATE,
        n_jobs=SUPERVISED_N_JOBS,
    )
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
    """Configura l'arsenale di metriche di valutazione per il modello.

    Scelta delle metriche:
    In presenza di cluster di dimensioni diverse, la semplice Accuracy è ingannevole
    (predire sempre il cluster maggiore fornirebbe un punteggio alto ma inutile).
    Si utilizzano le metriche 'Macro', che calcolano il punteggio (Precision/Recall/F1)
    per ogni singola classe separatamente e ne fanno poi la media non pesata.
    In questo modo, classificare correttamente il cluster più piccolo è importante
    tanto quanto classificare quello più grande.
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
    """Consolida i risultati della Cross-Validation per il modello migliore.

    Dopo l'esplorazione esaustiva della griglia da parte di GridSearchCV, questa funzione
    intercetta la combinazione di iperparametri considerata ottimale. Estrae la media
    e la deviazione standard (std) calcolata sui vari fold di validazione per ciascuna metrica.
    La deviazione standard è importante perché indica quanto il modello è stabile
    su porzioni di dati differenti.
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
    """Motore esecutivo (Entry Point) della fase supervisionata.

    Orchestra l'intero esperimento in modo riproducibile:
    1. Importa le librerie e carica il dataset normalizzato.
    2. Registra la distribuzione nativa dei cluster.
    3. Imposta la "Repeated Stratified K-Fold Cross Validation": una validazione rigorosa
       che mantiene costante la percentuale delle classi in ogni suddivisione e ripete il
       processo più volte per annullare fluttuazioni statistiche.
    4. Sottopone tutti i modelli configurati in `build_model_specs` a ottimizzazione
       tramite GridSearch, utilizzando l'F1 Macro come criterio definitivo ('refit')
       per eleggere la variante vincente.
    5. Esporta un report CSV dettagliato con metriche e migliori parametri per la stesura della relazione.
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
    best_model = None
    best_model_score = float("-inf")
    best_model_name = None

    for spec in build_model_specs(ml, preprocessor):
        print(f"Avvio esperimento: {spec['name']} | SMOTE={spec['use_smote']}")
        pipeline = spec["pipeline_cls"](steps=spec["steps"])
        search = ml["GridSearchCV"](
            estimator=pipeline,
            param_grid=spec["param_grid"],
            scoring=scoring,
            refit="f1_macro",
            cv=cv,
            n_jobs=SUPERVISED_N_JOBS,
            pre_dispatch=SUPERVISED_PRE_DISPATCH,
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
        if search.best_score_ > best_model_score:
            best_model_score = search.best_score_
            best_model = search.best_estimator_
            best_model_name = f"{spec['name']} | SMOTE={spec['use_smote']}"

        print(f"Completato esperimento: {spec['name']} | SMOTE={spec['use_smote']}")
        del search
        del pipeline
        gc.collect()

    metrics_df = pd.DataFrame(metric_rows)
    params_df = pd.DataFrame(parameter_rows)

    Path(SUPERVISED_METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(SUPERVISED_METRICS_PATH, index=False)
    params_df.to_csv(SUPERVISED_BEST_PARAMS_PATH, index=False)
    Path(SUPERVISED_MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    dump(best_model, SUPERVISED_MODEL_PATH)
    print(f"Miglior modello salvato: {best_model_name} -> {SUPERVISED_MODEL_PATH}")

    return class_distribution, metrics_df, params_df


if __name__ == "__main__":
    distribution, metrics, params = run_supervised_learning()
    print("Class distribution")
    print(distribution.to_string(index=False))
    print("\nMetrics")
    print(metrics.round(4).to_string(index=False))
    print("\nBest parameters")
    print(params.to_string(index=False))
