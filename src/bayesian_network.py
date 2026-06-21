"""Apprendimento e inferenza di una rete bayesiana per il rischio commerciale."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from config import (
    BAYESIAN_EDGES_PATH,
    BAYESIAN_FEATURES,
    BAYESIAN_MAX_INDEGREE,
    BAYESIAN_QUERIES_PATH,
    CLUSTERED_DISCRETIZED_DATA_PATH,
    RANDOM_STATE,
    TARGET_CLUSTER_COLUMN,
)


def import_pgmpy_dependencies():
    """Importa pgmpy gestendo differenze tra versioni della libreria.

    Alcune versioni usano `BayesianNetwork`, altre `DiscreteBayesianNetwork`; anche
    gli estimatori dei parametri hanno API diverse. La funzione isola queste
    differenze e restituisce un dizionario uniforme usato dal resto dello script.
    """
    try:
        from pgmpy.estimators import BicScore, HillClimbSearch, MaximumLikelihoodEstimator
        from pgmpy.inference import VariableElimination
        try:
            # Nelle versioni recenti BayesianNetwork e deprecato.
            from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
        except ImportError:
            from pgmpy.models import BayesianNetwork as BayesianModel
        try:
            # Nuova API per stimare CPD discrete.
            from pgmpy.parameter_estimator import DiscreteMLE
        except ImportError:
            DiscreteMLE = None
    except ImportError:
        try:
            from pgmpy.estimators import BIC as BicScore
            from pgmpy.estimators import HillClimbSearch, MaximumLikelihoodEstimator
            from pgmpy.inference import VariableElimination
            try:
                from pgmpy.models import DiscreteBayesianNetwork as BayesianModel
            except ImportError:
                from pgmpy.models import BayesianNetwork as BayesianModel
            try:
                from pgmpy.parameter_estimator import DiscreteMLE
            except ImportError:
                DiscreteMLE = None
        except ImportError as exc:
            raise ModuleNotFoundError(
                "Missing pgmpy dependency. Install requirements.txt before running this script."
            ) from exc

    return {
        "BayesianModel": BayesianModel,
        "BicScore": BicScore,
        "HillClimbSearch": HillClimbSearch,
        "MaximumLikelihoodEstimator": MaximumLikelihoodEstimator,
        "DiscreteMLE": DiscreteMLE,
        "VariableElimination": VariableElimination,
    }


def load_bayesian_dataset() -> pd.DataFrame:
    """Carica il dataset adatto alla rete bayesiana.

    Usa il dataset discretizzato e gia arricchito con `Cluster_Label`. Controlla che
    tutte le variabili previste siano presenti e converte ogni colonna in interi,
    per rappresentare stati discreti.
    """
    data_path = Path(CLUSTERED_DISCRETIZED_DATA_PATH)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Discretized clustered dataset not found at {data_path}. Run clustering.py first."
        )

    df = pd.read_csv(data_path)
    missing = [column for column in BAYESIAN_FEATURES if column not in df.columns]
    if missing:
        raise ValueError(f"Missing Bayesian Network columns: {missing}")

    bayesian_df = df[BAYESIAN_FEATURES].copy()
    for column in bayesian_df.columns:
        # Tutte le variabili della rete devono essere stati discreti interi.
        bayesian_df[column] = bayesian_df[column].astype(int)

    return bayesian_df


def learn_structure(df: pd.DataFrame, pgmpy):
    """Apprende automaticamente la struttura della rete bayesiana.

    HillClimbSearch esplora strutture candidate e il punteggio BIC penalizza reti
    troppo complesse. `BAYESIAN_MAX_INDEGREE` limita il numero massimo di genitori
    per nodo, rendendo le CPD piu leggibili e meno costose.
    """
    search = pgmpy["HillClimbSearch"](df)
    try:
        return search.estimate(
            scoring_method=pgmpy["BicScore"](df),
            max_indegree=BAYESIAN_MAX_INDEGREE,
        )
    except TypeError:
        # Compatibilita con API pgmpy che accettano il nome dello score come stringa.
        return search.estimate(
            scoring_method="bic-d",
            max_indegree=BAYESIAN_MAX_INDEGREE,
        )


def fit_model(df: pd.DataFrame, structure, pgmpy):
    """Stima le tabelle di probabilita condizionata della rete.

    A partire dalla struttura appresa, il modello calcola le CPD sui dati
    discretizzati. Alla fine `check_model` verifica che la rete sia formalmente
    consistente prima di procedere con le query.
    """
    model = pgmpy["BayesianModel"](structure.edges())
    if pgmpy["DiscreteMLE"] is not None:
        # API recente: fit richiede un estimatore discreto gia inizializzato.
        model.fit(df, estimator=pgmpy["DiscreteMLE"]())
    else:
        # API precedente: fit accetta la classe MaximumLikelihoodEstimator.
        model.fit(df, estimator=pgmpy["MaximumLikelihoodEstimator"])
    model.check_model()
    return model


def save_edges(model) -> pd.DataFrame:
    """Esporta gli archi della rete appresa in un CSV.

    Questo output serve per documentare le dipendenze trovate dall'algoritmo e per
    commentarle nella relazione senza dover visualizzare direttamente il grafo.
    """
    edges_df = pd.DataFrame(list(model.edges()), columns=["Source", "Target"])
    Path(BAYESIAN_EDGES_PATH).parent.mkdir(parents=True, exist_ok=True)
    edges_df.to_csv(BAYESIAN_EDGES_PATH, index=False)
    return edges_df


def most_common_state(df: pd.DataFrame, column: str) -> int:
    """Restituisce lo stato discreto piu frequente di una variabile.

    Viene usato per costruire query dimostrative realistiche, scegliendo un valore
    osservato spesso nel dataset.
    """
    return int(df[column].mode().iloc[0])


def highest_state(df: pd.DataFrame, column: str) -> int:
    """Restituisce lo stato discreto massimo osservato.

    Per le variabili discretizzate del progetto corrisponde in genere allo stato
    'alto', ad esempio prezzo alto o playtime alto.
    """
    return int(df[column].max())


def lowest_state(df: pd.DataFrame, column: str) -> int:
    """Restituisce lo stato discreto minimo osservato.

    Per le variabili discretizzate del progetto corrisponde in genere allo stato
    'basso', ad esempio prezzo basso.
    """
    return int(df[column].min())


def query_cluster_distribution(inference, evidence: dict) -> list[dict]:
    """Calcola la distribuzione del cluster dato uno scenario osservato.

    L'evidenza rappresenta informazioni disponibili prima della pubblicazione, come
    genere o prezzo. Il risultato indica quale profilo commerciale risulta piu
    probabile secondo la rete bayesiana.
    """
    result = inference.query(
        variables=[TARGET_CLUSTER_COLUMN],
        evidence=evidence,
        show_progress=False,
    )

    rows = []
    for state, probability in enumerate(result.values):
        rows.append(
            {
                "Query": f"P({TARGET_CLUSTER_COLUMN} | {evidence})",
                "Variable": TARGET_CLUSTER_COLUMN,
                "State": state,
                "Probability": float(probability),
            }
        )
    return rows


def query_low_review_risk(inference, evidence: dict) -> list[dict]:
    """Calcola la distribuzione del review score dato uno scenario osservato.

    Viene usata per stimare il rischio di ricezione bassa degli utenti. In
    particolare, uno stato 0 di `Review_Score_Pct` viene interpretato come segnale
    di rischio commerciale.
    """
    result = inference.query(
        variables=["Review_Score_Pct"],
        evidence=evidence,
        show_progress=False,
    )

    rows = []
    for state, probability in enumerate(result.values):
        rows.append(
            {
                "Query": f"P(Review_Score_Pct | {evidence})",
                "Variable": "Review_Score_Pct",
                "State": state,
                "Probability": float(probability),
            }
        )
    return rows


def run_business_queries(df: pd.DataFrame, model, pgmpy) -> pd.DataFrame:
    """Esegue le query probabilistiche usate nella relazione.

    Le query sono costruite per rappresentare scenari decisionali del publisher:
    prezzo alto, prezzo basso con durata alta, multiplayer e rischio di review
    score basso. I risultati vengono salvati in `bayesian_queries.csv`.
    """
    inference = pgmpy["VariableElimination"](model)

    # Gli stati sono discreti: 0=basso, 1=medio, 2=alto per le feature discretizzate.
    common_genre = most_common_state(df, "Primary_Genre")
    high_price = highest_state(df, "Price")
    low_price = lowest_state(df, "Price")
    long_playtime = highest_state(df, "Playtime_Hours")
    multiplayer = 1 if 1 in set(df["Multiplayer"]) else most_common_state(df, "Multiplayer")

    query_rows = []
    query_rows.extend(
        query_cluster_distribution(
            inference,
            evidence={"Primary_Genre": common_genre, "Price": high_price},
        )
    )
    query_rows.extend(
        query_cluster_distribution(
            inference,
            evidence={"Price": low_price, "Playtime_Hours": long_playtime},
        )
    )
    query_rows.extend(
        query_low_review_risk(
            inference,
            evidence={"Price": high_price, "Multiplayer": multiplayer},
        )
    )

    queries_df = pd.DataFrame(query_rows)
    queries_df.to_csv(BAYESIAN_QUERIES_PATH, index=False)
    return queries_df


def run_bayesian_network() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Esegue l'intera fase bayesiana.

    Carica i dati discretizzati, apprende la struttura, stima le CPD, salva gli
    archi e produce le query probabilistiche usate come supporto decisionale.
    """
    pgmpy = import_pgmpy_dependencies()
    df = load_bayesian_dataset()
    structure = learn_structure(df, pgmpy)
    model = fit_model(df, structure, pgmpy)
    edges_df = save_edges(model)
    queries_df = run_business_queries(df, model, pgmpy)
    return edges_df, queries_df


if __name__ == "__main__":
    edges, queries = run_bayesian_network()
    print("Learned edges")
    print(edges.to_string(index=False))
    print("\nInference queries")
    print(queries.to_string(index=False))
